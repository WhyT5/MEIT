#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_l2_quick.py —— 判断模型有没有"读懂心电图"（L2 快检，不需要生成完整报告）

两个判据，**必须都通过**才算条件化：

  A. 嵌入可分性
     不同 ECG 经过 encoder+projector 后的嵌入，两两余弦相似度。
     实测参考：完全没训练时 ≈ 1.000（所有心电图看起来一样）；能区分时应显著下降（< 0.9 更好）。

  B. 边界 token 正确率（关键对照）
     给模型 prompt，看它预测的"报告第一个 token"是不是参考报告的第一个 token。
     同时跑一遍 **把 ECG 置零**（等价于"不看心电图"）作为基线。
     只有 A 通过、且 B 明显高于零基线，才算真的在用 ECG。

用法：
  python check_l2_quick.py --ckpt_dir demo_output_2k/epoch_4 --num 32
  python check_l2_quick.py --ckpt_dir ... --split train --num 32   # 对照：训练集
"""
import argparse
import json
import os
import sys

import torch
from torch.nn.functional import normalize
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_data_collator import DataCollatorForSeq2Seq  # noqa: E402
from ecg_llm import ECG_LLM_ForCausalLM  # noqa: E402

FIXED_USER_PROMPT = "generate a description of the ECG embeddings: "


def _default_base_model():
    """优先用同级目录下的 gpt2_medium_local（离线可用），否则回退到 HF 名字。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "gpt2_medium_local"),
                 os.path.join(here, "gpt2_medium_local")):
        cand = os.path.abspath(cand)
        if os.path.exists(os.path.join(cand, "config.json")):
            return cand
    return "gpt2-medium"


def build_args(ckpt_dir, base_model, cache_dir, llm_type, ecg_model_type):
    cfg = AutoConfig.from_pretrained(base_model, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"unk_token": "<unk>", "pad_token": "[PAD]"})

    class A:
        pass

    a = A()
    for k, v in dict(
        model_name_or_path=base_model, cache_dir=cache_dir, llm_type=llm_type,
        ecg_model_type=ecg_model_type, use_lora=True, use_qlora=False,
        lora_rank=64, lora_alpha=128, lora_dropout=0.1,
        use_flash_attn=False, low_cpu_mem_usage=False, train_or_test="test",
        ecg_model_ckpt_path=os.path.join(ckpt_dir, "ecg_model.pth"),
        lora_model_name_or_path=ckpt_dir,
    ).items():
        setattr(a, k, v)
    return a, cfg, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--demo_dir", default="demo_data_de_2k")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=32)
    ap.add_argument("--base_model", default=_default_base_model(),
                    help="默认自动探测同级的 gpt2_medium_local")
    ap.add_argument("--llm_type", default="gpt2_medium")
    ap.add_argument("--ecg_model_type", default="ResNet50")
    _hf = os.environ.get("HF_HOME")
    ap.add_argument("--cache_dir", default=(os.path.join(_hf, "hub") if _hf else None),
                    help="默认取 $HF_HOME/hub；不设则用 HF 默认缓存")
    a_cli = ap.parse_args()

    path = os.path.join(a_cli.demo_dir, f"ptbxl_ecg_{a_cli.split}.jsonl")
    if not os.path.exists(path):
        sys.exit(f"[错误] 找不到 {path}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][: a_cli.num]
    print(f"[数据] {path}  取 {len(rows)} 条")

    a, cfg, tokenizer = build_args(a_cli.ckpt_dir, a_cli.base_model,
                                   a_cli.cache_dir, a_cli.llm_type, a_cli.ecg_model_type)
    print(f"[模型] {a_cli.ckpt_dir}")
    model = ECG_LLM_ForCausalLM(args=a, model_name_or_path=a_cli.base_model, config=cfg,
                                tokenizer=tokenizer,
                                ecg_layer_idxs=list(range(cfg.num_hidden_layers)),
                                accelerator=None)
    model.eval()

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None,
                                      padding="longest", ecg_data_name="ptbxl")
    prefix = f"<|user|>\n{FIXED_USER_PROMPT}\n<|assistant|>\n"
    # 设备必须统一：ECG_model.forward 用 input_ids.device 去搬特征张量，
    # 若 input_ids 在 CPU、ecg 在 GPU，就会把特征搬回 CPU 而 proj 层在 GPU → 报 mixed device。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_ids = tokenizer(prefix, return_tensors="pt", max_length=128,
                           truncation=True).input_ids.to(device)

    embs, acc_ecg, acc_zero, top5_ecg, n = [], 0, 0, 0, 0
    with torch.no_grad():
        for r in rows:
            ref = r["messages"][1]["content"]
            ref_ids = tokenizer(ref.strip(), add_special_tokens=False).input_ids
            if not ref_ids:
                continue
            target = ref_ids[0]

            feat = {"input_ids": prompt_ids.flatten(), "labels": prompt_ids.flatten(),
                    "attention_mask": torch.ones_like(prompt_ids).flatten(),
                    "ecg": r["ecg_path"]}
            ecg = collator([feat])["ecg"].to(device)

            # A. 嵌入
            e = model.ecg_model(ecg, prompt_ids)
            embs.append(normalize(e, dim=-1)[0].float().cpu())

            # B. 边界 token（有 ECG）
            lg = model(input_ids=prompt_ids, ecg=ecg, use_cache=False).logits[0, -1]
            acc_ecg += int(lg.argmax().item() == target)
            top5_ecg += int(target in lg.topk(5).indices.tolist())

            # B'. 边界 token（ECG 置零 = 不看图）
            lg0 = model(input_ids=prompt_ids, ecg=torch.zeros_like(ecg),
                        use_cache=False).logits[0, -1]
            acc_zero += int(lg0.argmax().item() == target)
            n += 1

    E = torch.stack(embs)
    S = E @ E.T
    off = [S[i, j].item() for i in range(n) for j in range(i + 1, n)]
    mean_cos = sum(off) / max(1, len(off))
    mean_std = E.std(dim=0).mean().item()

    print("\n" + "=" * 62)
    print(f"样本数: {n}    split={a_cli.split}")
    print("-" * 62)
    print(f"A. 嵌入可分性")
    print(f"   跨样本平均余弦 = {mean_cos:.4f}   （越接近 1 越糟；未训练≈1.000）")
    print(f"   跨样本 std     = {mean_std:.5f}")
    print(f"B. 边界 token 正确率")
    print(f"   有 ECG : top-1 {acc_ecg/n:6.1%}   top-5 {top5_ecg/n:6.1%}")
    print(f"   无 ECG : top-1 {acc_zero/n:6.1%}   （置零基线）")
    print(f"   差值   : {(acc_ecg-acc_zero)/n:+.1%}   ← 这个差值才是「用上了心电图」的证据")
    print("=" * 62)

    ok_a = mean_cos < 0.95
    ok_b = (acc_ecg - acc_zero) / n > 0.10
    print(f"判定: A={'通过' if ok_a else '未通过'} | B={'通过' if ok_b else '未通过'}"
          f" → L2 {'通过' if (ok_a and ok_b) else '未通过'}")
    print("注意：随机猜的 top-1 约 1/50257；但若所有报告都以同一 token 开头，")
    print("      零基线也会很高——所以**只看差值**，不要看绝对准确率。")


if __name__ == "__main__":
    main()
