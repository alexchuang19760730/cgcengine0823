# CGC Engine Models

All models are hosted on HuggingFace: [`Alexchuang/cgcengine-models`](https://huggingface.co/Alexchuang/cgcengine-models)

Total: 79.5 GB (7 files)

---

## Model Catalog

### 🏆 Ornith-1.5-35B-A3B (Recommended for Code/Agent)

| 量化 | 大小 | 文件名 | 说明 |
|------|------|--------|------|
| IQ3_XXS | 15.3 GB | `Ornith-1.5-35B-A3B-IQ3_XXS.gguf` | **推荐** — 代码/Agent 能力最强，16GB Mac 可跑 |

```bash
# 下载
hf download Alexchuang/cgcengine-models Ornith-1.5-35B-A3B-IQ3_XXS.gguf

# 运行 (MTP)
llama-speculative-simple \
  -m Ornith-1.5-35B-A3B-IQ3_XXS.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 -c 3072 --temp 0 \
  -expert-cache 4294967296
```

### 🔧 Nail-Qwen3.6-35B-A3B-MTP (Production MTP)

| 量化 | 大小 | 文件名 | 说明 |
|------|------|--------|------|
| denseIQ4X | 13.7 GB | `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf` | **生产版** — Q6_K head，26 t/s on Mac M4 |
| IQ3_XXS | 14.1 GB | `Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS.gguf` | 原始 MTP 版 |

```bash
# 下载生产版
hf download Alexchuang/cgcengine-models Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf

# 运行
llama-speculative-simple \
  -m Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 -c 3072 --temp 0
```

### 📦 Qwen3.6-35B-A3B (Base Model)

| 量化 | 大小 | 文件名 | 说明 |
|------|------|--------|------|
| IQ3_XXS | 13.2 GB | `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | 原始 Qwen3.6，无 MTP |

### 🤖 Gemma4-26B-A4B (MoE Sub-Agent)

| 量化 | 大小 | 文件名 | 说明 |
|------|------|--------|------|
| IQ3_S | 11.3 GB | `gemma-4-26B-A4B-it-UD-IQ3_S.gguf` | 多模态 MoE，用作子代理 |

### 🔬 Huihui-Qwen3.8-27B (Experimental)

| 量化 | 大小 | 文件名 | 说明 |
|------|------|--------|------|
| IQ3_S | 12.0 GB | `Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf` | Qwen3.8 27B abliterated 版本 |

---

## Quick Download (All Models)

```bash
# 下载全部 (79.5 GB)
hf download Alexchuang/cgcengine-models --local-dir ./models/gguf

# 只下载生产模型 (13.7 GB)
hf download Alexchuang/cgcengine-models \
  Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf \
  --local-dir ./models/gguf

# 只下载 Ornith-1.5 (15.3 GB)
hf download Alexchuang/cgcengine-models \
  Ornith-1.5-35B-A3B-IQ3_XXS.gguf \
  --local-dir ./models/gguf
```

---

## Model Selection Guide

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| **代码生成 / Agent** | Ornith-1.5 IQ3_XXS | 代码能力最强，自我改进训练 |
| **MTP 推理 (Mac M4)** | Nail-denseIQ4X | 26 t/s，expert-cache 优化 |
| **通用问答** | Qwen3.6 IQ3_XXS | 基础模型，稳定 |
| **多模态子代理** | Gemma4 IQ3_S | 图文理解，MoE 架构 |
| **实验性测试** | Huihui-Qwen3.8 IQ3_S | 最新 Qwen3.8 架构 |

---

## Hardware Requirements

| 模型 | 最低 RAM | 推荐 RAM | 说明 |
|------|----------|----------|------|
| Gemma4 26B | 8 GB | 12 GB | 最小的 MoE |
| Huihui Qwen3.8 27B | 10 GB | 16 GB | 27B dense |
| Qwen3.6 35B | 12 GB | 16 GB | 35B MoE |
| Nail-denseIQ4X | 12 GB | 16 GB | MTP + expert-cache |
| Ornith-1.5 35B | 14 GB | 24 GB | 最强但最大 |

---

*Last updated: 2026-09-01*
*Repo: https://huggingface.co/Alexchuang/cgcengine-models*
