# MEIT 最小 demo —— 执行手册与实跑记录（v3）

> 更新：2026-10-15
> **状态一句话：已在本地 CPU 上完整跑通（训练 → checkpoint → 推理 → 生成报告），
> 生成结果已落盘；但输出退化，根因已用数据定位。**
> v2 的定位是"怎么在服务器上把它跑起来"；v3 记录了"实际跑起来之后发现的 8 处源码 bug、
> 1 个必需参数、2 条环境约束，以及输出为什么退化"。
>
> 产物体积：`demo_output_cpu/` 1.9GB ｜ `.venvs/meit_cpu` 1.7GB ｜ `.hf_cache` 1.5GB

---

## 0. 现状摘要（先看这一节）

| 项 | 结果 |
|---|---|
| 训练 | ✅ 跑通。3 epoch / 12 step / **42 秒**（3.55 s/it），loss 80.21 → 68.77 |
| checkpoint | ✅ `demo_output_cpu/epoch_{0,1,2}/`（adapter + `ecg_model.pth`） |
| 推理 | ✅ 跑通。6 条样本全部返回（`demo_generations.jsonl`） |
| 生成质量 | ❌ **退化**：6 条全部输出 `<|ass` 后立即 EOS |
| 根因 | ✅ 已定位（见 §5）：ECG 编码器仅从随机初始化训了 12 步 + `proj_ecg` 的 BatchNorm eval 统计量把信号压缩约 1000 倍 |
| 判定 | **管线可用，但此规模下的输出不可用于任何质量结论**（v2 里"验证报告质量"的说法作废） |

---

## 1. 八处源码修复（已落地，`git diff` 可查）

前五处是训练/运行阻塞，后三处是推理路径。**第 1 条会让训练在第一个 batch 必崩，
第 4 条会让每次 import 白等 7 分钟。**

### 1.1 训练/运行阻塞（5 处）

| # | 文件 | 症状 | 根因 | 修法 |
|---|---|---|---|---|
| 1 | `ecg_data_collator.py`<br>（两份 collator 类，共 2 处） | `TypeError: isnan(): argument 'input' must be Tensor, not numpy.ndarray`，训练第一个 batch 必崩 | `ptbxl` 分支对 **numpy 数组** 调 `torch.isnan()`。已在 torch **1.4.0 与 2.13.0** 双版本复现 → 该分支在任何 torch 下都跑不通 | 改用 `np.nan_to_num(ecg, nan=0., posinf=0., neginf=0.)`，语义不变（mimic 分支无需改，那里已是 Tensor） |
| 2 | `ecg_data_collator.py` | 启动即 `ModuleNotFoundError: ecg_plot` | 顶层 `import ecg_plot` / `import yaml`，但**本文件从未使用**（真实用法在 `draw_figures/draw_ecg.py`）；`ecg_plot` 会连带拉入 opencv | 删除这两个 import |
| 3 | `ecg_data_collator.py`（2 处） | 无 GPU 环境直接崩；多卡/非 0 卡会放错设备 | 硬编码 `feature['ecg'] = ecg.to("cuda")` | 新增 `_ecg_device()`：有 GPU 用 GPU、否则 CPU。**服务器 GPU 行为完全不变**（`ecg_llm.py:450` 本来就会再 `.to(input_ids.device)`） |
| 4 | `eval_metrics.py` | **import 训练脚本要 7 分 4 秒** | 模块级无条件调用 `nltk.download('wordnet'/'punkt')`；nltk 默认源 `raw.githubusercontent.com` 不可达时会长时间超时。而这两个包只在算 METEOR 时用到 | 改为默认不联网（`MEIT_DOWNLOAD_NLTK=1` 才下载）。**import 耗时 7m04s → 4.6s** |
| 5 | `modeling_ecg_llama.py` | `ImportError: cannot import name '_get_unpad_data' from 'transformers.models.llama.modeling_llama'` | 这三个符号在较新 transformers 里已搬家：`_get_unpad_data` → `modeling_flash_attention_utils`；`_make_causal_mask`/`_expand_mask` → `modeling_attn_mask_utils.AttentionMaskConverter` | 加 `try/except ImportError` 双向兼容（旧版走原路径）。本 demo 走 GPT-2 路径，这些符号只在 LLaMA 分支用 |

### 1.2 推理路径（3 处）

| # | 文件 | 症状 | 修法 |
|---|---|---|---|
| 6 | `ecg_llm.py`（test 分支） | `load_state_dict` 报 `Missing key(s) / Unexpected key(s)` | 训练侧 `save_with_accelerate()→get_state_dict()` 保存的 `ecg_model.pth` 保留了 `ecg_model.` 前缀，而加载端期望无前缀 → 加载时识别并剥离 |
| 7 | `ecg_llm.py`（test 分支） | `device_map` 被当非法值 | 原文 `device_map = "..." if ... else "auto",` **末尾多一个逗号**，实际是 tuple `("auto",)`；且 CPU 环境应取 `None` |
| 8 | `ecg_llm.py`（`ECG_LLM_ForCausalLM.generate`） | `AttributeError: 'GPT2LMHeadModel' object has no attribute '_get_generation_mode'`<br>`AttributeError: 'GenerationConfig' object has no attribute '_eos_token_tensor'` | `models/modeling_ecg_gpt2.py` 里的自定义 `generate()` 是从 **transformers 4.37/4.38** 整段拷贝的（调用 `GenerationMixin._get_generation_mode`），而本仓库 `requirements.txt` 写的是 4.43.4（4.39 起该逻辑迁到 `GenerationConfig.get_generation_mode`）。**拷贝版 generate 与新版 mixin 混用，不是打补丁能修好的** |

> **第 8 条的处置（方案 A，已落地并验证）**：把 `ECG_LLM_ForCausalLM.generate()` 改成
> **自包含的贪心解码**——直接复用模型自己的 `forward`（ECG 注入本来就在 forward 里），
> 不再调用 `self.llm_model.generate(...)`，因此既不依赖那份 250 行拷贝，也不依赖
> transformers 各版本的生成内部实现。语义等价于原 `do_sample=False / num_beams=1 / max_length=128`。
>
> **等价性已实测**：同一份 checkpoint、同 6 条样本，改造前后两次输出**逐条完全一致**
> （`demo_generations.jsonl` vs `demo_generations_via_model_generate.jsonl`）。
> 收益：仓库自己的 `generate()` 接口现在可用，`semanic_test.py` / `zeroshot_test.py`
> 等上游脚本也能直接调；`demo_infer.py` 里那份临时解码已删除。
>
> **版本归属已查实**（下载 4.37–4.42 源码逐个比对）：
>
> | 版本 | `GenerationMixin._get_generation_mode` | `GenerationConfig.get_generation_mode` |
> |---|---|---|
> | 4.37.2 / 4.38.2 | ✅ | ❌ |
> | 4.39.3 – 4.43.4 | ❌ | ✅ |
>
> 若坚持"用仓库自带的那份 generate"，需钉 `transformers==4.38.2` + `tokenizers<0.19`
> （4.38 要求 tokenizers 0.18.x），并重跑全部验证。**不推荐**：训练路径根本不用
> `generate()`，降级只服务推理，还会连带降 tokenizers。

### 1.3 需要同步到服务器的文件（5 改 + 3 新）

| 文件 | 类型 | 说明 |
|---|---|---|
| `ECG_LLMs/ecg_data_collator.py` | 覆盖 | 修复 1/2/3 |
| `ECG_LLMs/ecg_llm.py` | 覆盖 | `breakpoint()` 注释 + 修复 6/7 + 修复 8（自包含 generate） |
| `ECG_LLMs/eval_metrics.py` | 覆盖 | 修复 4（否则每次 import 白等 7 分钟） |
| `ECG_LLMs/modeling_ecg_llama.py` | 覆盖 | 修复 5 |
| `ECG_LLMs/build_mini_demo_data.py` | 覆盖 | 数据脚本：默认德文、`--translate` 可选 |
| `ECG_LLMs/demo_infer.py` | 新增 | 最小推理入口（现走仓库自己的 `model.generate()`） |
| `probe_server_env.sh` | 新增 | 服务器环境一键探测 |

> `models/modeling_ecg_gpt2.py` **不在同步清单里**：方案 A 之后不再调用它内部那份
> 旧版 `generate()`，因此那个文件保持上游原样（试过的半截补丁已还原，避免掩盖问题）。

```bash
# 端口 1897；<服务器MEIT路径> 用 probe_server_env.sh 第 5 节查出来的值
for f in ecg_data_collator.py ecg_llm.py eval_metrics.py modeling_ecg_llama.py \
         build_mini_demo_data.py demo_infer.py; do
  scp -P 1897 MEIT/ECG_LLMs/$f lmx@10.130.128.54:<服务器MEIT路径>/ECG_LLMs/
done
scp -P 1897 MEIT/probe_server_env.sh lmx@10.130.128.54:<服务器MEIT路径>/
```

---

## 2. **一个必需的命令行参数：`--use_slow_tokenizer`**

漏了它，训练会在第一个 batch 报 `IndexError: index out of range in self`（在 `wte` 词嵌入）。

**机理**：`AutoTokenizer.from_pretrained('gpt2-medium')` 默认返回 **`GPT2TokenizerFast`**，而
`isinstance(GPT2TokenizerFast_instance, GPT2Tokenizer)` 是 **False**（两者是兄弟类，不是父子）。
于是 `ecg_llm.py:273` 那段

```python
if isinstance(tokenizer, GPT2Tokenizer):
    tokenizer.add_special_tokens({'unk_token': '<unk>', 'pad_token': '[PAD]'})
embedding_size = self.llm_model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    self.llm_model.resize_token_embeddings(len(tokenizer))
```

**整段被跳过** → 但另一处 `else` 分支仍把 `[PAD]` 加进了 tokenizer（len → 50259，`pad_id=50258`），
模型词表却仍是 50257 → padding 出来的 id 越界。

上游 `run_osc_gpt_2_medium.sh` / `scripts/run_osc_gpt_2_medium_ptbxl.sh` **都带着这个 flag**，
MINI_DEMO v1/v2 漏了。**推论：`GPT2Tokenizer` 的 isinstance 判断在这个代码库里是一处隐藏地雷，
换任何别的 tokenizer 加载方式都会再踩一次。** 更彻底的修法是把判断改成
`isinstance(tokenizer, (GPT2Tokenizer, GPT2TokenizerFast))`，本次未做（保持改动最小）。

---

## 3. 环境搭建（本机实测可复现）

本地 GPU **不可用**（工作区 `/dev/nvidia*` 不存在，`torch.cuda` 报 `Error 304`），
所以用纯 CPU venv。**实测 CPU 完全够跑这个 demo。**

```bash
V=/root/work/S/.venvs/meit_cpu
/root/miniconda3/envs/meit/bin/python -m venv $V      # 复用 meit 环境的 python 3.10

# ① torch 走官方 CPU 源，避免拖 2.5GB 的 nvidia 依赖（该源实测可达）
$V/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.0 torchvision==0.19.0

# ② 其余依赖：这几个版本号是钉死的，装"最新版"必炸
$V/bin/pip install "huggingface_hub==0.24.6" "transformers==4.43.4" "tokenizers==0.19.1" \
    "accelerate==0.31.0" "peft==0.12.0" "datasets==2.21.0" \
    wfdb sentencepiece protobuf pandas scikit-learn matplotlib \
    "evaluate==0.4.2" nltk "bert-score==0.3.13" "rouge==1.0.1" "pycocoevalcap==1.2" tqdm
```

**为什么必须钉版本（都是实测结论）：**
- `transformers==4.43.4`：`models/modeling_ecg_gpt2.py` 依赖 4.x 的
  `transformers.pytorch_utils.Conv1D` / `SequenceSummary` / `utils.model_parallel_utils`，
  这些在 5.x 已被移除。
- `datasets==2.21.0` / `peft==0.12.0` / `huggingface_hub==0.24.6`：装最新版会把
  `huggingface_hub` 拉到 1.x，与 transformers 4.43 的 API 冲突。
- **不要装** `flash-attn` / `deepspeed` / `vllm` / `bitsandbytes`（本 demo 都不需要）。
- **不要装 `rouge_score`**：全仓库没有任何代码 import 它，而它是唯一需要现场编译的包
  —— 会因 pip 的 wheel 缓存目录 `/root/.cache/pip/wheels` 不可写而**整个安装失败**。
  真正被 import 的是 `rouge`（`utils_ecg.py: from rouge import Rouge`），别搞混。

**权重下载（hf-mirror，实测 ~1.37 MB/s）：**

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/work/S/.hf_cache
export HF_HUB_DISABLE_XET=1        # ← 必须！见下
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('gpt2-medium', cache_dir='$HF_HOME/hub', \
  allow_patterns=['*.json','*.txt','*.model','model.safetensors'])"
```

> ⚠️ **`HF_HUB_DISABLE_XET=1` 是必须的。** 新版 `huggingface_hub` 默认走 **Xet** 协议，
> 而 hf-mirror 不支持该端点，会报
> `ConnectionError: HTTP status client error (404 Not Found), domain: https://hf-mirror.com/api/models/gpt2-medium/xet-read-token/...`。
> 小文件（config/tokenizer）能下成功，**只有大权重文件会失败**，很容易误判成"网络不通"。
> 禁用 Xet 后走普通 HTTP，1.42GB 约 9 分钟下完。

**关于 `--cache_dir`**：脚本默认值是作者集群的 `/local/scratch0/data_and_checkpoint/models`，
会被直接传给 `from_pretrained(cache_dir=...)`。本机用 `--cache_dir /root/work/S/.hf_cache/hub`
（即 `$HF_HOME/hub`，与环境变量保持一致，避免两份缓存）。

**其他实测体量**：`pycocoevalcap` wheel **104MB**、`pyarrow` 50MB、`scipy` 37MB。
`pycocoevalcap` 大是因为打包了 METEOR 的 jar 与数据。

---

## 4. ⚠️ 上游真实配方修正（重要）

查 `scripts/run_osc_gpt_2_medium_ptbxl.sh`（PTB-XL 专用启动脚本）发现，
**MINI_DEMO v1/v2 的配方与上游有多处不一致**，其中前两条直接影响可复现性：

| 项 | 上游实际做法 | MINI_DEMO v1/v2 |
|---|---|---|
| 入口脚本 | **`semanic_test.py`** | `finetune_ecgllm_with_lora_ptbxl.py` |
| `--ecg_data_name` | **不传** → 用该脚本的默认值 **`"mimic"`** | 显式传 `ptbxl` |
| LoRA | `rank 64 / alpha 128 / dropout 0.1` | `rank 16 / alpha 32 / dropout 0.05` |
| 学习率 | `2e-5` | `1e-4` |
| epochs | `5` | `3` |
| 有效 batch | `16 × 4 = 64` | `4 × 1 = 4` |
| 其他 | `--with_tracking --report_to tensorboard --preprocessing_num_workers 16 --dev_ratio 0.1 --val_test_ratio 0.1` | 均未传 |

### 4.1 这一条回答了 v2 §6 悬而未决的"归一化歧义"

- `finetune_ecgllm_with_lora_ptbxl.py` 的 `--ecg_data_name` 默认值是 **`"ptbxl"`**；
- 但上游真正用的 **`semanic_test.py` 默认值是 `"mimic"`**（两个脚本 diff 有 112 行）；
- 上游启动脚本不传该参数 → **上游对 PTB-XL 数据走的是 `mimic` 分支，即逐记录 min-max 归一化到 [0,1]，不是原始 mV。**

**这与"`ptbxl` 分支是死代码"完全自洽**：作者从未用过那条分支，所以它坏了也没人发现。
**结论：要复现 MEIT 的 PTB-XL 行为，必须显式用 `--ecg_data_name mimic`。**
（patch 过 `ptbxl` 分支后也能跑，但那是偏离上游的选择，必须在实验记录里写明。）

> 本次 3-epoch 实跑用的是**已修复的 `ptbxl` 分支（原始 mV）**，与上游不一致 —— 这是一个混杂因素，
> 记在此处备查。

---

## 5. 本地实跑记录与根因诊断

### 5.0 数据构建（v3.1：默认原文、不调 LLM）

`build_mini_demo_data.py` 已重写：**默认直接用 PTB-XL 原始的 `report` 字段，不联网**；
翻译改成可选开关 `--translate`（默认关）。

```bash
# 默认：20 条原文 → demo_data_de/（14/3/3）
python build_mini_demo_data.py

# 2k 条：建议用 random（head 会继承排序偏斜，见下）
python build_mini_demo_data.py --num 2000 --sample random --out-dir demo_data_de_2k

# 需要英文时才打开（要求 DEEPSEEK_API_KEY；先查缓存，缺失才调 API）
python build_mini_demo_data.py --num 20 --translate --out-dir demo_data_en
```

**已实测的三条：**

| 检查 | 结果 |
|---|---|
| 默认路径（20 条德文） | ✅ train 14 / val 3 / test 3；`demo_data_de/` |
| 2k 随机抽样 | ✅ 1400/300/300；`demo_data_de_2k/` |
| `--translate` 缺 key | ✅ 明确报错退出，提示去掉 `--translate` |

> ⚠️ **实测更正：所谓「原文」其实是德英混合，不是纯德文。** 全库 21799 条判定：
> 德文 61.3%（13357）/ **英文 30.8%（6712）** / 混排 5.8%（1259）/ 其它 2.2%（471，多为 `trace only requested.`）。
> 2k 训练集里出现频率最高的报告就是英文的 `sinus rhythm. normal ecg.`（120/1400）。
> 含义：默认语料是混合语言，不是「德文预训练」。

### 5.0.1 如果需要纯英文：直接取原文里的英文子集

PTB-XL 自带 6712 条**本来就是英文**的报告，够 2k 用，**零翻译、零泄漏、零成本**。
但它是**有选择偏倚的子集**（实测）：

| | 德文子集 | 英文子集 |
|---|---|---|
| 条数 | 13357 | 6712 |
| SCP 含 NORM | 44.7% | 46.7% |
| 平均年龄 | **58.5** | **69.4** |
| 女性比例 | **54.8%** | **44.9%** |
| strat_fold 分布 | 各 fold 内比例一致（非批次性差异） | 同左 |

结论：英文子集更老、更偏男性，但各 fold 内分布一致，说明不是按批次划分的。
用于 **L2（能否条件化）不受影响**；用于论文的正式对比时，必须报告这一偏倚，
或改用「全量 + 去 SCP codes 的翻译」以消除选择偏倚。

**与旧英文数据集的关系**：两条数据集用的是**完全相同的 20 条 ECG 记录**（已逐条比对
`ecg_path` 集合），所以可以直接做"德文 vs 英文"对照，不引入样本差异。
`id` 从原来的下标（`ptbxl_14`）改成了 **ecg_id**（`ptbxl_15` 等），便于回查原始记录。

**为什么要改成默认原文（四个理由）：**
1. 零成本、零延迟、零网络依赖；
2. **避免标签泄漏**——旧脚本的翻译 prompt 是
   *"You will receive an original ECG report together with its SCP codes … use
   translate_scp_codes to look up the codes"*，等于把答案告诉了写目标报告的人。
   20 条时侥幸没出问题（已逐条核过），**2k 条批量做会把偶发问题放大成系统偏差**；
3. 验证 L2（ECG 能否条件化）只需要 loss / 边界 token / 嵌入可分，**与语言无关**；
4. ECG 编码器学的是 "ECG → 报告内容"，与语言无关，德文预训练后可再用英文微调。
   **但注意**：GPT-2 的德文生成能力弱，德文输出**只能验证管线，不能当报告质量证据**。

**2k 条的两个实测提醒：**

- **抽样偏斜**（"含 normal 的报告"比例）：前 20 条 **85.0%** / 前 2000 条 **53.1%** / 全库 **61.3%**
  （SCP 含 NORM：80.0% / 52.8% / 43.6%）。所以 `--sample head` 取 2000 条会让 NORM 偏多约 9 个百分点，
  **建议 2k 用 `--sample random`**。
- **序列长度**：抽 1000 条测得 report token 长度 min/median/p95/max = 1/**17**/73/**223**。
  prompt 固定开销约 27 token，所以 `--max_seq_length 128` 会截断约 **1%** 的报告
  （>64 token 的占 8.1%，>128 的占 0.3%）。2k 规模建议把 `--max_seq_length` 提到 **192**。

### 5.1 训练（3 epoch，与 v2 手册相同的超参）

```bash
cd MEIT/ECG_LLMs
export HF_HOME=/root/work/S/.hf_cache HF_ENDPOINT=https://hf-mirror.com NLTK_DATA=/root/work/S/.nltk_data
/root/work/S/.venvs/meit_cpu/bin/python finetune_ecgllm_with_lora_ptbxl.py \
  --model_name_or_path gpt2-medium --tokenizer_name gpt2-medium --use_slow_tokenizer \
  --use_ecg_llm --use_lora --llm_type gpt2_medium --ecg_data_name ptbxl \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --train_data_path demo_data/ptbxl_ecg_train.jsonl \
  --val_data_path   demo_data/ptbxl_ecg_val.jsonl \
  --test_data_path  demo_data/ptbxl_ecg_test.jsonl \
  --max_seq_length 128 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 --num_train_epochs 3 \
  --learning_rate 1e-4 --lr_scheduler_type linear --warmup_ratio 0.03 \
  --weight_decay 0.0 --logging_steps 1 --checkpointing_steps epoch --seed 42 \
  --cache_dir /root/work/S/.hf_cache/hub --output_dir demo_output_cpu
```

**关键日志事实**（用于确认配置真的生效）：
- `Distributed environment: NO` / `Device: cpu` / `Mixed precision type: no`
- `trainable params: 1,572,864 || all params: 356,396,032 || trainable%: 0.4413`（LoRA 只占 0.44%）
- `the ecg layers include: [0..23]`（24 层全部注入）
- `####### save ecg model ok ############`（v2 判据命中）
- `Loading ecg_encoder success`（ECG 分支被真正构造）

**Loss 曲线（12 步全程，batch=4）：**

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 80.21 | 74.03 | 75.00 | 72.92 | 67.54 | 73.33 | 73.90 | 73.27 | 75.99 | **62.14** | 72.82 | 68.77 |

有噪声但整体下降（80.2 → 68.8）。注意 loss 量级在 60–80，是 **sum over tokens** 而非 mean
（见 §5.3 的换算）——不要误读成"loss 爆炸"。

**产物结构**（每 epoch 一份）：
```
demo_output_cpu/epoch_{0,1,2}/
  adapter_config.json          r=16, alpha=32, target_modules=["c_attn"]
  adapter_model.safetensors    418MB
  ecg_model.pth                81MB
```
> `adapter_model.safetensors` 418MB / 50 个张量 / 104.5M 参数，**不是**纯 LoRA：
> 48 个是 LoRA（合计 1.57M 参数），另外 2 个是 `wte.weight` 与 `lm_head.weight`
> （各 50259×1024 ≈ 51.5M）。**这其实是好事**——被 resize 过的词表嵌入被一起保存了，
> 推理时不会出现 `[PAD]` 行是随机初始化的对齐问题。
>
> ⚠️ 每 epoch 存一次 = 每 epoch 约 500MB。要跑多 epoch 时建议
> `--checkpointing_steps 999999`（只在最后存一次），否则 60 epoch 会写 ~30GB。

### 5.2 推理与生成结果

```bash
python demo_infer.py --ckpt_dir demo_output_cpu/epoch_2 \
  --cache_dir /root/work/S/.hf_cache/hub --split val,test --out demo_generations.jsonl
```

| # | split | 参考报告（截断） | 生成 |
|---|---|---|---|
| ptbxl_14 | val | sinus arrhythmia; otherwise normal ecg. | `<|ass` |
| ptbxl_15 | val | sinus rhythm. normal ecg. | `<|ass` |
| ptbxl_16 | val | atrial fibrillation with a rapid ventricular response… | `<|ass` |
| ptbxl_17 | test | atrial flutter with 2:1 av conduction… | `<|ass` |
| ptbxl_18 | test | sinus rhythm with p-wave broadening. | `<|ass` |
| ptbxl_19 | test | supraventricular escape beats… | `<|ass` |

`prompt_boundary=prefix`，即推理 prompt 与训练侧掩码边界**交叉校验一致**（`demo_infer.py` 内置断言），
所以这不是"训练-推理 prompt 不一致"导致的。

### 5.3 三条诊断证据（本次最有价值的产出）

**证据 1 —— 报告被背下来了。** 用训练好的模型在**训练样本**上算 loss：

```
loss（20 个监督 token 求和） = 6.973
折算每 token 交叉熵         = 0.349 nats     （随机基线 ln(50257) = 10.82）
```

**证据 2 —— 但它不知道是哪一条。** 在"应输出报告第一个词"的位置，模型 top-5 预测：

| rank | token | logit |
|---|---|---|
| 1 | `<` | −61.04 |
| 2 | `\|` | −63.13 |
| 3 | `The` | −63.57 |
| 4 | `A` | −64.29 |
| 5 | `I` | −64.44 |

而参考报告的起始 token 是 **`sin`** —— 不在 top-5 里。
（这也解释了 0.349 的算术：第一个 token 贡献约 7 nats，其余 19 个接近 0 → 平均 0.35。
即"除了不知道选哪条，其余全部背熟"。）

**证据 3 —— ECG 条件信息为零。** 对 6 条**不同**心电图算投影后的嵌入：

| 检查 | eval 模式（= 推理实际用的模式） | train 模式 |
|---|---|---|
| `ecg_encoder` 输出跨样本 std | 0.00133 | — |
| `proj_ecg` 最终输出跨样本 std | **0.00108** | **1.0954** |
| 跨样本平均余弦相似度 | **0.99999** | −0.111 |

6 条完全不同的 ECG，嵌入两两余弦 = 1.000 → **塌成常数**。
根因：`proj_ecg` 里有两个 `BatchNorm1d`，其 `num_batches_tracked` 只有 **12**
（12 个 batch 的统计量），eval 模式下用它做归一化，把信号**压缩约 1000 倍**
（train 模式 1.0954 → eval 模式 0.00108）。

### 5.4 结论

输出退化**不是**"MEIT 方法不行"，而是两个因素叠加：

1. **ECG 编码器是 16M 参数、随机初始化的 ResNet50，只训了 12 步** —— 其特征本就近乎与输入无关；
2. **`proj_ecg` 的 BatchNorm 在推理时用 12 个 batch 估出的统计量，把残存信号又压掉约 1000 倍**。

上游是在**全量 PTB-XL（约 21k 条）× 5 epoch**、有效 batch 64 下训练的 —— 步数是这里的
**上千倍**。所以在 20 条规模上，这个架构在推理侧必然无法条件化。**这是规模的必然结果，不是 bug。**

---

## 6. 两个必须记住的推论

### 6.1 BatchNorm 的 train/eval 不一致 = 第 7 类分布偏移

在 OpenTSLM 阶段建立过一张"6 处训练-推理分布偏移"对照表。**MEIT 架构自带第 7 类，
而且不属于使用失误：**

> `ECG_model.proj_ecg` 的 `BatchNorm1d` 使模型在 train / eval 下行为不同。
> 训练步数不足时 running stats 不可用，**推理侧会把 ECG 条件整段抹掉**（本文实测约 1000 倍）。

对研究的影响：任何在小规模上做的"注入 vs 不注入"对比，都可能因为这一项而**测不出差异**
——因为 ECG 条件本身已经被吃掉了。**做 C2/C3/C4 对比前，必须先确认 ECG 通路有效。**

### 6.2 ★对 C4 / AH 的关键混杂因素（比上面全部加起来更重要）

核心指标 AH 度量的是"锚点错了，报告会不会照抄"。但如果 **ECG 条件本身没有信息**，
模型"照抄锚点"只是因为它**别无信息可依** —— 此时 AH 会虚高，
而"模型把锚点当作不可质疑的事实"这个归因**根本不成立**。

> **因此 C4（假阳性污染）之前必须先做一个前置检查：
> 证明 C0 模型能从 ECG 生成具有判别力的报告**
> （例如 ECG-only 的标签级召回显著高于随机/多数类基线；或 §5.3 证据 3 那种嵌入可分性检查）。
> 否则 AH 不可解释，B 组消融（针对 ECG-LENS 门控的批评）也失去立足点。

---

## 7. 未完成事项（等指令，不自行推进）

**v3.1 已完成（2026-10-15 第二轮）：**

| 事项 | 状态 |
|---|---|
| 方案 A：`generate()` 改为自包含贪心解码 | ✅ 完成，且**逐条输出与改造前完全一致**（等价性已实测） |
| `models/modeling_ecg_gpt2.py` 的半截补丁 | ✅ 已还原（不再调用它，留半截修复反而掩盖问题） |
| 数据脚本重写：默认原文、`--translate` 可选 | ✅ 完成；20 条/2k 两条路径均验证 |
| 新数据能否被训练器消费 | ✅ 1 epoch 冒烟通过（4 步，loss 83.3→78.6，checkpoint 正常） |
| 新数据能否走 `model.generate()` 推理 | ✅ 通过（2 条，输出仍退化，与 §5.4 诊断一致） |

**仍待做：**

| 事项 | 状态 |
|---|---|
| 长训练验证"多训能否从 ECG 条件化"（建议：**上游配方 `mimic` + rank64/alpha128/lr2e-5**，1–2k 条起步） | ⏸ 待做 |
| 归一化对照：`mimic`（min-max，上游）vs `ptbxl`（原始 mV）各跑一次 | ⏸ 待做 |
| **C0 前置检查：ECG 条件有效性**（见 §6.2，是 C4/AH 的必要前置） | ⏸ 待做 |
| 把 BatchNorm 这条补进"分布偏移"对照表 | ⏸ 待做 |
| 服务器路径占位符（环境名 / MEIT 路径 / PTBXL 根目录） | ⏸ 未填，需在服务器上跑 `probe_server_env.sh` |
| 德文 vs 英文的语言对照（两条数据集 ECG 完全相同，可直接配对） | ⏸ 待做 |
| `--translate` 的实际 API 调用路径 | ⚠️ **本轮未验证**（无 key，只验证了缺 key 的报错分支） |
| 3-epoch 旧实跑用的是 ptbxl 分支（原始 mV），与上游 mimic 配方不一致 | ⚠️ 混杂因素（新冒烟已改用 mimic） |

---

## 8. 报错 → 症状对照表（v3，全部为实测）

| 症状 | 原因 | 处理 |
|---|---|---|
| `TypeError: isnan(): argument 'input' must be Tensor, not numpy.ndarray` | `ecg_data_collator.py` 未同步修复版 | 同步该文件；或临时改 `--ecg_data_name mimic` |
| `IndexError: index out of range in self`（在 `wte`） | **漏了 `--use_slow_tokenizer`** | 加上该参数（见 §2） |
| `ImportError: cannot import name '_get_unpad_data'` | transformers 版本与 `modeling_ecg_llama.py` 不匹配 | `pip install transformers==4.43.4 tokenizers==0.19.1`；或同步已修的 `modeling_ecg_llama.py` |
| `AttributeError: ... no attribute '_get_generation_mode'`<br>`AttributeError: 'GenerationConfig' object has no attribute '_eos_token_tensor'` | `models/modeling_ecg_gpt2.py` 里那份旧版 `generate()` 与新版 transformers 不兼容 | **不要调 `self.llm_model.generate(...)`**；用 `ECG_LLM_ForCausalLM.generate()`（已改为自包含贪心解码，见 §1.2 第 8 条）。若坚持用上游那份，需降级 `transformers==4.38.2` + `tokenizers<0.19` |
| `ModuleNotFoundError: ecg_plot` / `yaml` | 服务器上是未修复版 collator | 同步修复版；**不要真的去装 `ecg_plot`** |
| `ModuleNotFoundError: wfdb / torchvision / rouge / PIL` | 装了 v1 的兜底清单 | 按 §3 补齐 |
| `RuntimeError: Error(s) in loading state_dict for ECG_model: Missing/Unexpected key(s)` | `ecg_model.pth` 的 `ecg_model.` 前缀未剥离 | 同步已修的 `ecg_llm.py` |
| `ConnectionError ... xet-read-token ... 404` | 新版 hub 走 Xet，hf-mirror 不支持 | `export HF_HUB_DISABLE_XET=1` |
| pip 装到一半 `Failed to build rouge_score` / `Permission denied: '/root/.cache/pip/wheels'` | `rouge_score` 需要现场编译，且 wheel 缓存目录不可写 | **不要装 `rouge_score`**（无人 import 它） |
| import 训练脚本卡 7 分钟无输出 | `eval_metrics.py` 模块级 `nltk.download` 等不可达服务器 | 同步已修的 `eval_metrics.py`；或预置 `NLTK_DATA` |
| `CUDA out of memory` | batch 偏大 | 降 `--per_device_train_batch_size` |
| **生成结果退化（如输出 `<|ass` 后 EOS）** | **ECG 条件信息在推理侧被 BatchNorm 抹掉 + 编码器训练步数严重不足** | **不是 bug；见 §5.4。需按 §6.2 先做 C0 前置检查，或按 §7 加大规模/改用上游配方** |

---

## 9. 命令速查

```bash
# 一次性环境变量（每个新 shell 都要设）
export HF_HOME=/root/work/S/.hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export NLTK_DATA=/root/work/S/.nltk_data
PY=/root/work/S/.venvs/meit_cpu/bin/python

# ① 构建数据（默认原文、不联网；--translate 才翻译）
cd /root/work/S/MEIT/ECG_LLMs
$PY build_mini_demo_data.py                      # 20 条 → demo_data_de/
# $PY build_mini_demo_data.py --num 2000 --sample random --out-dir demo_data_de_2k

# ② 训练（本地 CPU 版；--use_slow_tokenizer / --ecg_data_name 见 §2、§4）
$PY finetune_ecgllm_with_lora_ptbxl.py --model_name_or_path gpt2-medium \
    --tokenizer_name gpt2-medium --use_slow_tokenizer \
    --use_ecg_llm --use_lora --llm_type gpt2_medium --ecg_data_name mimic \
    --train_data_path demo_data_de/ptbxl_ecg_train.jsonl \
    --val_data_path   demo_data_de/ptbxl_ecg_val.jsonl \
    --test_data_path  demo_data_de/ptbxl_ecg_test.jsonl \
    --cache_dir /root/work/S/.hf_cache/hub --output_dir demo_output_de

# ③ 推理（走仓库自己的 ECG_LLM_ForCausalLM.generate()）
$PY demo_infer.py --ckpt_dir demo_output_de --demo_dir demo_data_de \
    --cache_dir /root/work/S/.hf_cache/hub --split val,test \
    --out demo_generations_de.jsonl

# 服务器侧环境探测（填占位符用）
bash /root/work/S/MEIT/probe_server_env.sh 2>&1 | tee probe.log
```

---

## 10. 与 v2 的差异一览

| v2 的说法 | v3 的修正 |
|---|---|
| "修复 1 处 bug（`torch.isnan`）+ 补 `--cache_dir`" | 实际需要 **8 处源码修复** + **1 个必需参数** `--use_slow_tokenizer` |
| "`--ecg_data_name ptbxl`" | 上游用的是 **`mimic`**（min-max 归一化）；`ptbxl` 分支是死代码 |
| "LoRA rank 16 / alpha 32 / lr 1e-4 / 3 epochs" | 上游是 **rank 64 / alpha 128 / lr 2e-5 / 5 epochs / 有效 batch 64** |
| "dependency 清单补 5 个包" | 还需钉死 `datasets/peft/huggingface_hub` 版本、**不装 `rouge_score`**、**必须设 `HF_HUB_DISABLE_XET=1`** |
| "P0 判据：loss 下降 + 保存 checkpoint" | 判据确实命中（12 步、42 秒、3 个 epoch 目录），**但"通过判据"≠"输出可用"** |
| "通过后验证报告质量" | ❌ **作废**。此规模下推理侧必然无法条件化（§5.4），只能验证管线连通性 |
| 未提及 | **BatchNorm 的 train/eval 不一致**（第 7 类分布偏移）与 **C4/AH 的前置混杂因素**（§6.2） |

---

## 附：本地验证环境存档

| 项 | 值 |
|---|---|
| Python | 3.10.20（venv 复用 `/root/miniconda3/envs/meit` 的解释器） |
| venv | `/root/work/S/.venvs/meit_cpu`（1.7GB） |
| torch / torchvision | `2.4.0+cpu` / `0.19.0+cpu` |
| transformers / tokenizers | `4.43.4` / `0.19.1` |
| accelerate / peft / datasets / hub | `0.31.0` / `0.12.0` / `2.21.0` / `0.24.6` |
| 权重缓存 | `/root/work/S/.hf_cache/hub/models--gpt2-medium`（1.5GB） |
| 硬件 | 本工作区 GPU 不可见（`/dev/nvidia*` 缺失，torch 报 Error 304）；训练全程 CPU，10 线程 |
| 实测速度 | 训练 3.55 s/it（batch=4）｜import 4.6s｜hf-mirror 下载 ~1.37 MB/s |
