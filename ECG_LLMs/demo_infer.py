#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_infer.py —— MEIT 最小推理脚本（本地 CPU 可跑 / 服务器 GPU 同样可用）

作用：加载 `finetune_ecgllm_with_lora_ptbxl.py` 训练产出的 checkpoint，
      对 demo_data 里的样本逐条生成报告，并和参考报告并排打印。

为什么需要它：训练脚本里那段生成/评测被 `if step > 9 and step % test_step == 0`
挡住，20 条 demo 永远触发不了，所以「拿几条生成样例」必须单独有个入口。

用法：
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=<可写缓存目录>
  python demo_infer.py --ckpt_dir demo_output/epoch_2 --split val,test

设计要点（避免重犯 OpenTSLM 那次的「训练-推理分布偏移」）：
  1. ECG 预处理直接调用 `DataCollatorForSeq2Seq`，不在这里重写一份；
  2. prompt 前缀用与训练完全相同的拼接方式构造，且用 labels 掩码边界做交叉校验；
  3. 逐条 batch=1 推理（避免 padding 引入 [PAD]，测试分支没有 resize embedding）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import wfdb
from torchvision.transforms import transforms
from transformers import AutoConfig, AutoTokenizer, GPT2Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecg_data_collator import DataCollatorForSeq2Seq  # noqa: E402
from ecg_llm import ECG_LLM_ForCausalLM  # noqa: E402

# 与 finetune_ecgllm_with_lora_ptbxl.py 的 _concat_messages() 保持一致
FIXED_USER_PROMPT = "generate a description of the ECG embeddings: "


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="训练产出的目录，如 demo_output/epoch_2（内含 adapter_config.json + ecg_model.pth）")
    p.add_argument("--base_model", default="gpt2-medium")
    p.add_argument("--llm_type", default="gpt2_medium")
    p.add_argument("--ecg_model_type", default="ResNet50")
    p.add_argument("--cache_dir", default=os.environ.get("HF_HOME", "./.hf_cache"))
    p.add_argument("--demo_dir", default="demo_data")
    p.add_argument("--split", default="val,test", help="逗号分隔，可选 train,val,test")
    p.add_argument("--num", type=int, default=0, help="每个 split 最多取几条，0=全部")
    p.add_argument("--out", default=None, help="结果写入的 jsonl 路径")
    return p.parse_args()


def build_prompt_ids(tokenizer, reference_report=None, max_seq_length=128):
    """返回 (prompt_ids, boundary_source)。

    训练时模型看到的文本是（见 _concat_messages）：
        <|user|>\n{PROMPT}\n<|assistant|>\n{report}{eos}\n
    生成时的输入是「第一个未被 mask 的 token 之前」的部分，即到 <|assistant|>\n 为止。
    这里同时用两种方式算边界并交叉校验，防止训练/推理不一致。
    """
    prefix = f"<|user|>\n{FIXED_USER_PROMPT}\n<|assistant|>\n"
    ids_direct = tokenizer(prefix, return_tensors="pt",
                           max_length=max_seq_length, truncation=True).input_ids

    if reference_report is None:
        return ids_direct, "prefix"

    # 走一遍训练侧的完整拼接 + 掩码，取出边界做校验
    full = prefix + str(reference_report).strip() + tokenizer.eos_token + "\n"
    enc = tokenizer(full.strip(), return_tensors="pt",
                    max_length=max_seq_length, truncation=True)
    input_ids = enc.input_ids
    labels = input_ids.clone()
    messages_so_far = f"<|user|>\n{FIXED_USER_PROMPT}\n<|assistant|>\n"
    boundary = tokenizer(messages_so_far, return_tensors="pt",
                         max_length=max_seq_length, truncation=True).input_ids.shape[1]
    labels[:, boundary:] = input_ids[:, boundary:]
    labels[:, :boundary] = -100
    first_unmasked = int((labels != -100).long().argmax(dim=1)[0])
    ids_train_style = input_ids[:, :first_unmasked]

    same = ids_direct.shape == ids_train_style.shape and bool(
        torch.equal(ids_direct, ids_train_style))
    tag = "prefix" if same else "MISMATCH"
    if not same:
        print(f"  [警告] 推理 prompt 与训练掩码边界不一致！"
              f"direct={ids_direct.shape} train_style={ids_train_style.shape}")
    return ids_direct, tag


def main():
    args_cli = parse_args()

    # ---- 组装 ECG_LLM_ForCausalLM 需要的 args ----
    cfg = AutoConfig.from_pretrained(args_cli.base_model, cache_dir=args_cli.cache_dir)
    # 与训练保持一致：训练命令用了 --use_slow_tokenizer。这很关键 ——
    # AutoTokenizer 默认返回 GPT2TokenizerFast，而代码里的 isinstance(tokenizer, GPT2Tokenizer)
    # 对 Fast 版本是 False，会导致 [PAD] 没被加入、词表没被 resize（训练时就会 IndexError）。
    tokenizer = AutoTokenizer.from_pretrained(
        args_cli.base_model, cache_dir=args_cli.cache_dir, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"unk_token": "<unk>", "pad_token": "[PAD]"})

    ecg_layer_idxs = [idx for idx in range(cfg.num_hidden_layers)]

    class A:
        pass

    a = A()
    a.model_name_or_path = args_cli.base_model
    a.cache_dir = args_cli.cache_dir
    a.llm_type = args_cli.llm_type
    a.ecg_model_type = args_cli.ecg_model_type
    a.use_lora = True
    a.use_qlora = False
    a.lora_rank = 16
    a.lora_alpha = 32
    a.lora_dropout = 0.05
    a.use_flash_attn = False
    a.low_cpu_mem_usage = False
    a.train_or_test = "test"
    a.ecg_model_ckpt_path = os.path.join(args_cli.ckpt_dir, "ecg_model.pth")
    a.lora_model_name_or_path = args_cli.ckpt_dir

    for f in [a.ecg_model_ckpt_path, os.path.join(args_cli.ckpt_dir, "adapter_config.json")]:
        if not os.path.exists(f):
            sys.exit(f"[错误] 找不到 {f}；--ckpt_dir 应指向训练产出的 epoch_N 目录")

    print(f"[1/3] 加载模型：base={args_cli.base_model} ckpt={args_cli.ckpt_dir}")
    model = ECG_LLM_ForCausalLM(args=a, model_name_or_path=args_cli.base_model,
                                config=cfg, tokenizer=tokenizer,
                                ecg_layer_idxs=ecg_layer_idxs, accelerator=None)
    model.eval()
    print(f"      ECG 预处理设备: {next(model.ecg_model.parameters()).device}")

    # 用真实 collator 做 ECG 预处理，保证与训练完全一致
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None,
                                      padding="longest", ecg_data_name="ptbxl")

    splits = [s.strip() for s in args_cli.split.split(",") if s.strip()]
    results = []
    print(f"[2/3] 开始生成（splits={splits}）")
    for split in splits:
        path = os.path.join(args_cli.demo_dir, f"ptbxl_ecg_{split}.jsonl")
        if not os.path.exists(path):
            print(f"      [跳过] {path} 不存在")
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if args_cli.num:
            rows = rows[:args_cli.num]
        for n, row in enumerate(rows):
            ref = row["messages"][1]["content"]
            prompt_ids, tag = build_prompt_ids(tokenizer, ref)

            # 借 collator 跑同一条预处理路径（需要 labels 存在才会处理 ecg）
            feat = {"input_ids": prompt_ids.flatten(),
                    "labels": prompt_ids.flatten(),
                    "attention_mask": torch.ones_like(prompt_ids).flatten(),
                    "ecg": row["ecg_path"]}
            batch = collator([feat])
            ecg = batch["ecg"]

            # 走仓库自己的接口（已在 ecg_llm.ECG_LLM_ForCausalLM.generate() 里
            # 换成自包含的贪心解码，不再依赖 modeling_ecg_gpt2 里那份旧版拷贝）
            out_ids = model.generate(input_ids=batch["input_ids"], ecg=ecg)
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            # 去掉 prompt 前缀，只留正文
            gen = text.split("<|assistant|>")[-1].strip()
            gen = gen.replace(FIXED_USER_PROMPT, "").strip()

            results.append({"id": row["id"], "split": split, "ecg_path": row["ecg_path"],
                            "prompt_boundary": tag, "reference": ref, "generated": gen})
            print(f"      [{split} {n+1}/{len(rows)}] {row['id']} ok")

    print("[3/3] 结果")
    print("=" * 150)
    for r in results:
        print(f"\n### {r['id']}  ({r['split']})")
        print(f"  参考: {r['reference']}")
        print(f"  生成: {r['generated'] if r['generated'] else '<空>'}")
    print("=" * 150)

    if args_cli.out:
        with open(args_cli.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"已写入 {args_cli.out}")
    return results


if __name__ == "__main__":
    main()
