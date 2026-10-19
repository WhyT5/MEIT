#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_mini_demo_data.py —— 构建 MEIT 指令数据集（ECG 波形 → 文本报告）

【v2 变更】默认**不再翻译、不调用任何 LLM**，直接使用 PTB-XL 原始报告（`report` 字段）。
            **注意：原文并非纯德文** —— 全库约 61% 德文 / 31% 英文 / 6% 混排 / 2% 其它，
            也就是说默认语料是**德英混合**的。需要纯英文时可取原文里本来就是英文的子集（约 6712 条）。
            翻译改成可选开关 `--translate`（默认关闭）。

为什么默认用原文：
  1. 零成本、零延迟、零网络依赖；
  2. 避免"翻译时把 SCP codes 喂给 LLM"造成的**标签泄漏**（旧脚本的 prompt 就是给了 codes）；
  3. 验证"ECG 能否条件化"（L2）只需要 loss / 边界 token / 嵌入可分，与语言无关；
  4. ECG 编码器学的是 "ECG → 报告内容"，与语言无关，可先用混合语料把编码器训到能条件化，再用纯语言微调。
  注意：GPT-2 的德文生成能力弱，德文部分的输出**只能用于验证管线，不能当报告质量证据**。

数据源：`ptbxl_database.csv`（官方库，ecg_id 为索引；`report` 为德/英混合原文，`filename_hr` 为 records500 路径）
输出：  <out-dir>/ptbxl_ecg_{train,val,test}.jsonl

输出格式对齐 MEIT finetune 脚本（load_dataset("json") + encode_with_messages_format）：
    {
      "dataset": "ptbxl",
      "id": "ptbxl_<ecg_id>",
      "ecg_path": "<ecg-root>/records500/00000/00001_hr",
      "messages": [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "<report>"}
      ]
    }
  注意：`messages[0].content` 会被训练脚本的 `_concat_messages()` **完全忽略**
  （prompt 是写死的字符串），所以这里留空即可；不要指望往它里面塞东西能生效。

用法：
    # 默认：20 条、原文（德英混合）、不联网
    python build_mini_demo_data.py

    # 2k 条、随机抽样（推荐：head 抽样会继承数据集的排序偏斜）
    python build_mini_demo_data.py --num 2000 --sample random --out-dir demo_data_de_2k

    # 打开翻译（需要 DEEPSEEK_API_KEY；会先查缓存，缺失的才调 API）
    python build_mini_demo_data.py --num 20 --translate --out-dir demo_data_en

分片：`--split` 给三段的**比例**，默认 0.7/0.15/0.15；num=20 时正好得到 14/3/3。
"""
import argparse
import csv
import json
import os
import random
import sys

import pandas as pd

# ---- 默认路径（相对本文件定位，换机器一般不用改） ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.abspath(os.path.join(_HERE, "..", "..", "ecg_ptbxl_benchmarking"))
DEFAULT_DB = os.path.join(_PROJ, "data", "ptbxl", "ptbxl_database.csv")
DEFAULT_ECG_ROOT = os.path.join(_PROJ, "data", "ptbxl")
DEFAULT_TRANSLATION_CACHE = os.path.join(_PROJ, "output", "ptbxl_report2_top20.csv")

# 数据里用于标记"这条报告不可用"的占位文本
PLACEHOLDER_MARKERS = ("something wrong with the data",)


# ============================================================
# 1. 读库与筛选
# ============================================================
def load_database(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[错误] 找不到 {path}\n       用 --db 指定 ptbxl_database.csv 的路径")
    df = pd.read_csv(path, index_col="ecg_id")
    for col in ("report", "filename_hr"):
        if col not in df.columns:
            sys.exit(f"[错误] {path} 缺少列 '{col}'")
    return df


def is_usable(report: str) -> bool:
    if report is None:
        return False
    s = str(report).strip()
    if not s or s.lower() == "nan":
        return False
    low = s.lower()
    return not any(m in low for m in PLACEHOLDER_MARKERS)


def iter_records(df: pd.DataFrame, num: int, sample: str, seed: int, ecg_root: str,
                 check_files: bool):
    """按 ecg_id 升序（或随机）挑出前 num 条可用记录。"""
    ids = sorted(df.index.tolist())
    if sample == "random":
        rng = random.Random(seed)
        rng.shuffle(ids)

    out = []
    for ecg_id in ids:
        row = df.loc[ecg_id]
        report = row["report"]
        if not is_usable(report):
            continue
        fn = str(row["filename_hr"]).strip()
        if not fn or fn.lower() == "nan":
            continue
        ecg_path = os.path.join(ecg_root, fn)
        if check_files and not os.path.exists(ecg_path + ".hea"):
            continue
        out.append({"ecg_id": int(ecg_id), "report": str(report).strip(),
                    "ecg_path": ecg_path})
        if len(out) >= num:
            break

    if len(out) < num:
        print(f"[警告] 只找到 {len(out)} 条可用记录（要求 {num} 条）")
    return out


# ============================================================
# 2. 可选：把德文翻成英文（默认关闭）
# ============================================================
TRANSLATE_SYSTEM = (
    "You are a meticulous medical translator. You will receive ONE German ECG report. "
    "Translate it into English faithfully. Hard rules: "
    "(1) Do NOT add, infer, complete, or explain any finding that is not literally in the input; "
    "(2) Do NOT omit anything; "
    "(3) Do NOT mention or guess SCP codes; "
    "(4) Output ONLY the translated sentence(s), no preamble, no quotes, no code fence."
)


def load_translation_cache(path: str) -> dict:
    """读取 {ecg_id: english} 缓存；不存在返回空 dict。"""
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[警告] 翻译缓存读取失败（忽略）: {e}")
        return {}
    if "ecg_id" not in df.columns or "report2" not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        try:
            eid = int(r["ecg_id"])
        except Exception:
            continue
        txt = r["report2"]
        if isinstance(txt, str) and txt.strip():
            out[eid] = txt.strip()
    return out


def translate_one(german: str, api_key: str, base_url: str, model: str,
                  timeout: int = 60) -> str:
    """调 OpenAI 兼容的 chat/completions。仅 --translate 时才会被调用。"""
    import requests  # 延迟导入：默认路径不需要它

    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": TRANSLATE_SYSTEM},
                {"role": "user", "content": german},
            ],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def translate_records(records, cache_path, api_key, base_url, model):
    """先查缓存，缺失的才调 API；结果写回缓存。"""
    cache = load_translation_cache(cache_path)
    hit = sum(1 for r in records if r["ecg_id"] in cache)
    print(f"[翻译] 缓存命中 {hit}/{len(records)}，其余将调用 {model}")

    rows = []
    for i, r in enumerate(records, 1):
        eid = r["ecg_id"]
        if eid in cache:
            en = cache[eid]
        else:
            try:
                en = translate_one(r["report"], api_key, base_url, model)
            except Exception as e:
                # 翻译失败就退回德文，**不要**静默丢弃这条
                print(f"  [{i}/{len(records)}] ecg_id={eid} 翻译失败，退回德文: {e}")
                en = r["report"]
            cache[eid] = en
        rows.append(en)
        if i % 50 == 0:
            print(f"  ... {i}/{len(records)}")

    # 写回缓存（只追加/更新，不破坏原文件结构）
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ecg_id", "report2"])
                for k, v in sorted(cache.items()):
                    w.writerow([k, v])
            print(f"[翻译] 已写回缓存 {cache_path}")
        except Exception as e:
            print(f"[警告] 缓存写回失败（不影响本次输出）: {e}")
    return rows


# ============================================================
# 3. 切分与落盘
# ============================================================
def split_sizes(num: int, ratios) -> tuple:
    """按比例算三段大小，保证总和等于 num。"""
    r = [float(x) for x in ratios]
    if len(r) != 3 or abs(sum(r) - 1.0) > 1e-6:
        sys.exit(f"[错误] --split 必须是 3 个和为 1 的比例，收到 {ratios}")
    n_train = int(round(num * r[0]))
    n_val = int(round(num * r[1]))
    n_test = num - n_train - n_val
    if n_test < 0:
        sys.exit(f"[错误] 比例与 num 不匹配：{n_train}+{n_val}>{num}")
    return n_train, n_val, n_test


def write_jsonl(path: str, items) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="构建 MEIT 指令数据集（默认德文原文、不调用 LLM）")
    ap.add_argument("--db", default=DEFAULT_DB, help="ptbxl_database.csv 路径")
    ap.add_argument("--ecg-root", default=DEFAULT_ECG_ROOT,
                    help="PTB-XL 根目录（其下有 records500/）")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "demo_data_de"),
                    help="输出目录（默认 demo_data_de，避免覆盖旧的英文 demo_data/）")
    ap.add_argument("--num", type=int, default=20, help="总条数（筛选后计数）")
    ap.add_argument("--split", default="0.7,0.15,0.15",
                    help="train,val,test 三段比例（默认 0.7,0.15,0.15；num=20 时为 14/3/3）")
    ap.add_argument("--sample", choices=["head", "random"], default="head",
                    help="head=按 ecg_id 取前 N 条；random=随机（2k 建议用 random，见下）")
    ap.add_argument("--seed", type=int, default=42, help="--sample random 的种子")
    ap.add_argument("--check-files", action="store_true",
                    help="跳过 records500 下 .hea 文件不存在的记录")
    # 翻译开关（默认关闭）
    ap.add_argument("--translate", action="store_true",
                    help="把德文翻成英文（默认关闭）。需要 DEEPSEEK_API_KEY。")
    ap.add_argument("--translation-cache", default=DEFAULT_TRANSLATION_CACHE,
                    help="翻译缓存 csv（ecg_id,report2）；命中则不调 API")
    ap.add_argument("--api-base", default=os.environ.get("DEEPSEEK_BASE_URL",
                                                         "https://api.deepseek.com"))
    ap.add_argument("--api-model", default=os.environ.get("DEEPSEEK_MODEL",
                                                          "deepseek-v4-flash"))
    args = ap.parse_args()

    if args.num <= 0:
        sys.exit("[错误] --num 必须为正")

    df = load_database(args.db)
    records = iter_records(df, args.num, args.sample, args.seed,
                           args.ecg_root, args.check_files)
    if not records:
        sys.exit("[错误] 没有可用记录")

    # ---- 目标文本 ----
    if args.translate:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            sys.exit("[错误] --translate 需要环境变量 DEEPSEEK_API_KEY。\n"
                     "       若只想用德文原文，去掉 --translate 即可。")
        targets = translate_records(records, args.translation_cache,
                                    api_key, args.api_base, args.api_model)
        lang = "en(translated)"
    else:
        targets = [r["report"] for r in records]
        lang = "de(original)"

    # ---- 切分 ----
    n_tr, n_va, n_te = split_sizes(len(records), args.split.split(","))
    print(f"[切分] train {n_tr} / val {n_va} / test {n_te}  (共 {len(records)})")

    def build(start, end):
        items = []
        for r, rep in zip(records[start:end], targets[start:end]):
            items.append({
                "dataset": "ptbxl",
                "id": f"ptbxl_{r['ecg_id']}",   # 用 ecg_id，便于回查原始记录
                "ecg_path": r["ecg_path"],
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": rep},
                ],
            })
        return items

    os.makedirs(args.out_dir, exist_ok=True)
    parts = {
        "train": build(0, n_tr),
        "val": build(n_tr, n_tr + n_va),
        "test": build(n_tr + n_va, len(records)),
    }
    for name, items in parts.items():
        p = os.path.join(args.out_dir, f"ptbxl_ecg_{name}.jsonl")
        write_jsonl(p, items)
        print(f"[{name}] {len(items)} 条 -> {p}")

    # ---- 摘要（只用字符长度，不引入 tokenizer 依赖） ----
    lens = [len(t) for t in targets]
    print(f"\n语言: {lang} | 总条数: {len(records)} | "
          f"报告字符长度 min/mean/max = {min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}")
    print(f"采样方式: {args.sample}"
          + (f" (seed={args.seed})" if args.sample == "random" else ""))
    if args.sample == "head":
        print("提示：head 会继承数据集的排序偏斜。实测「含 normal 的报告」比例："
              "前 20 条 85.0% / 前 2000 条 53.1% / 全库 61.3%（SCP 含 NORM：80.0% / 52.8% / 43.6%）。")
        print("      要更接近全库分布，2k 时建议加 --sample random。")


if __name__ == "__main__":
    main()
