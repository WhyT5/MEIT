#!/usr/bin/env bash
# MEIT 最小 demo —— 服务器环境一键探测
#
# 用途：在服务器上跑一次，把 MINI_DEMO.md 里所有 <占位符> 的值一次性问出来。
# 用法：  bash probe_server_env.sh            （或 bash probe_server_env.sh 2>&1 | tee probe.log）
#
# 设计原则：只读、不改动任何东西；每一步失败都不中断（set +e），
#           目的是拿信息，不是拿成功退出码。

set +e

hr() { printf '\n========== %s ==========\n' "$1"; }

hr "0. 主机与用户"
echo "host    : $(hostname)"
echo "user    : $(whoami)"
echo "pwd     : $(pwd)"
echo "date    : $(date)"
echo "home    : $HOME"

hr "1. GPU（找一张空闲卡；注意 CUDA_VISIBLE_DEVICES 用哪张）"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
             --format=csv,noheader 2>&1
else
  echo "!! 没有 nvidia-smi：这台机器没有可用 GPU，或驱动未装"
fi

hr "2. conda 环境（确认 <meit环境名>）"
if command -v conda >/dev/null 2>&1; then
  conda env list 2>&1
else
  echo "!! PATH 里没有 conda。若装了，试试：source ~/anaconda3/bin/activate /path/to/env"
  ls -d "$HOME"/*conda*/envs/* 2>/dev/null
fi

hr "3. 关键依赖版本（在【已激活 meit 环境】下跑才有意义）"
echo "python  : $(command -v python)  $(python -V 2>&1)"
python - <<'PY' 2>&1
mods = ["torch","torchvision","transformers","tokenizers","accelerate","peft",
        "datasets","evaluate","sklearn","matplotlib","numpy","pandas","wfdb",
        "pycocoevalcap","bert_score","rouge","nltk","tqdm","sentencepiece","yaml","ecg_plot"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  OK   {m:16s} {getattr(mod,'__version__','(no __version__)')}")
    except Exception as e:
        print(f"  MISS {m:16s} {type(e).__name__}: {e}")
try:
    import torch
    print(f"  -> torch.cuda.is_available() = {torch.cuda.is_available()}, "
          f"device_count = {torch.cuda.device_count()}")
except Exception:
    pass
# MEIT 需要 transformers 4.43.x
try:
    import transformers
    v = transformers.__version__
    print("  -> transformers 版本检查:", "OK (4.43.x)" if v.startswith("4.43") else f"⚠️ 是 {v}，MEIT 要求 4.43.x")
except Exception:
    pass
PY

hr "4. java（只有训练中触发 pycocoevalcap Meteor 才需要；本 demo 用不到）"
if command -v java >/dev/null 2>&1; then java -version 2>&1 | head -2; else echo "未安装（本 demo 不需要，可忽略）"; fi

hr "5. 定位 MEIT 代码目录（确认 <服务器MEIT路径>）"
find "$HOME" /root /data /mnt /home -maxdepth 5 \
     -name "finetune_ecgllm_with_lora_ptbxl.py" 2>/dev/null | head -5

hr "6. 定位 PTB-XL 数据根目录（确认 <PTBXL根目录>）"
find "$HOME" /root /data /mnt /home -maxdepth 6 \
     -type d -name "records500" 2>/dev/null | head -5
echo "--- 顺带找 ptbxl_database.csv ---"
find "$HOME" /root /data /mnt /home -maxdepth 6 \
     -name "ptbxl_database.csv" 2>/dev/null | head -5

hr "7. 定位报告 csv（确认 <服务器上报告csv路径>）"
find "$HOME" /root /data /mnt /home -maxdepth 6 \
     -name "ptbxl_report2_top20.csv" 2>/dev/null | head -5

hr "8. 抽样确认一条 PTB-XL 记录可读（record 00001）"
REC=$(find "$HOME" /root /data /mnt /home -maxdepth 6 -name "00001_hr.dat" 2>/dev/null | head -1)
echo "found: ${REC:-<没有找到 00001_hr.dat>}"
[ -n "$REC" ] && ls -la "$REC" "${REC%.dat}.hea" 2>&1

hr "9. 缓存目录与磁盘（--cache_dir 必须可写；默认值 /local/scratch0/... 通常不存在）"
echo "[ -w /local/scratch0 ] ->"; [ -w /local/scratch0 ] && echo "  可写" || echo "  不可写/不存在  ← 必须显式传 --cache_dir"
echo "--- df -h ---"; df -h "$HOME" /tmp 2>&1 | head -6
echo "--- 现有 HF 缓存里有没有 gpt2-medium ---"
ls -d "$HOME"/.cache/huggingface/hub/models--gpt2* 2>/dev/null || echo "  没有 gpt2 缓存（首次运行会下载；需能访问 HF）"

hr "10. HuggingFace 可达性（决定要不要设 HF_ENDPOINT / HF_HUB_OFFLINE）"
for u in https://huggingface.co https://hf-mirror.com; do
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$u" 2>/dev/null)
  echo "  $u -> ${code:-timeout}"
done

hr "11. 待同步文件是否已就位 + 代码修复是否已同步"
MEITDIR=$(find "$HOME" /root /data /mnt /home -maxdepth 5 \
          -name "finetune_ecgllm_with_lora_ptbxl.py" 2>/dev/null | head -1)
if [ -n "$MEITDIR" ]; then
  D=$(dirname "$MEITDIR")
  echo "MEIT 目录: $D"
  for f in ecg_llm.py ecg_data_collator.py build_mini_demo_data.py demo_data/ptbxl_ecg_train.jsonl; do
    [ -e "$D/$f" ] && echo "  有   $f" || echo "  缺   $f   ← 需要 scp 同步"
  done
  echo "--- ecg_llm.py 里 breakpoint 是否已注释（应无未注释的 breakpoint）---"
  grep -n "^\s*breakpoint()" "$D/ecg_llm.py" 2>/dev/null || echo "  OK：没有未注释的 breakpoint()"
  echo "--- ecg_data_collator.py 的 ptbxl 分支是否已修复 ---"
  if grep -q "nan_to_num" "$D/ecg_data_collator.py" 2>/dev/null; then
    echo "  OK：已含 nan_to_num（修复已同步）"
  else
    echo "  ⚠️ 未修复！仍是老代码 → 训练第一个 batch 必然 TypeError，必须同步 ecg_data_collator.py"
  fi
  echo "--- ecg_data_collator.py 是否还在 import ecg_plot（会多一个脆弱依赖）---"
  grep -n "^import ecg_plot" "$D/ecg_data_collator.py" 2>/dev/null && echo "  ⚠️ 仍在 import ecg_plot" || echo "  OK：已移除"
else
  echo "没找到 MEIT 目录，先确认代码传上去了没有。"
fi

hr "完成"
echo "把上面全部输出（可 tee 到 probe.log）回贴即可，我据此把 MINI_DEMO.md 的占位符填死。"
