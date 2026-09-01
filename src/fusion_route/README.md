# FusionRoute + MoT — 安装与使用指南

> 分支：`fusionroutemot`（GitHub: `fusion-route-mot`）
> 目标：token-level 多专家路由 + KV Cache Translation，双层修正架构

---

## 1. 快速安装

```bash
# 1. Clone 仓库
git clone https://github.com/alexchuang19760730/cgcengine0823.git
cd cgcengine0823

# 2. 切换到 FusionRoute 分支
git checkout fusionroutemot
# 或
git checkout fusion-route-mot

# 3. 安装 Python 依赖
pip install torch numpy
# 如需完整功能（Mac 校准）：
pip install transformers huggingface_hub
```

---

## 2. 模块结构

```
src/
├── fusion_route/              ← FusionRoute 路由 + 训练
│   ├── __init__.py
│   ├── router.py              ← Qwen3Router（token-level 路由）
│   ├── complementary.py       ← Complementary Logit 修正
│   ├── train_cdpo.py          ← CDPO 训练脚本（CPU 可跑）
│   ├── extract_prompts.py     ← 从 Freebuff 提取 coding prompts
│   ├── sanitize_training_data.py  ← 清理 secrets
│   ├── watchdog.py            ← 后台自动采集
│   └── training_data/         ← 提取的训练数据
│       ├── freebuff_prompts.jsonl
│       └── freebuff_prompts_cdpo.jsonl
│
├── kv_translation/            ← KV Cache Translation
│   ├── __init__.py
│   ├── ridge_mapper.py        ← 岭回归映射器
│   ├── calibration.py         ← 校准数据提取
│   ├── calibration_pipeline.py ← 完整校准管线
│   └── run_calibration_mac.sh ← Mac 一键执行
│
└── llama.cpp/                 ← CGC fork（含 expert-cache）
```

---

## 3. 各平台使用方式

### Windows 本机（开发 + 推理）

```bash
# A. 提取 Freebuff 对话 prompts
python src/fusion_route/extract_prompts.py --score --cdpo

# B. 启动 watchdog 自动采集（后台）
python src/fusion_route/watchdog.py --daemon

# C. 清理 secrets 后再 commit
python src/fusion_route/sanitize_training_data.py

# D. 跑 CDPO 训练（mock 数据，验证 pipeline）
python src/fusion_route/train_cdpo.py --n_samples 100 --n_epochs 3
```

### Mac M4（模型校准 + 真实 logits）

```bash
# A. 下载模型
hf download Alexchuang/cgcengine-models Ornith-1.5-35B-A3B-IQ3_XXS.gguf
hf download Alexchuang/cgcengine-models Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf

# B. 执行 KV 校准管线（2-4 小时）
cd src/kv_translation
chmod +x run_calibration_mac.sh
./run_calibration_mac.sh

# C. 或手动执行
python calibration_pipeline.py \
  --model_a Qwen/Qwen3.6-35B-A3B \
  --model_b ornith-ai/Ornith-1.5-35B-A3B \
  --n_samples 500 --seq_len 1024

# D. 拟合映射矩阵
python ridge_mapper.py fit \
  --calibration_dir calibration_data/ \
  --output kv_map.json
```

### WSL2（Linux 环境）

```bash
# 进入 WSL2
wsl -d Ubuntu-24.04

# 激活 venv
source ~/ah/bin/activate

# 跑训练
cd /mnt/d/alex/flashkv0516/cgcengine_full
python src/fusion_route/train_cdpo.py --n_samples 500 --n_epochs 10
```

### 鸿蒙手机（HarmonyOS NEXT）

```bash
# 部署 CGC engine 到手机（需 DevEco Studio）
# 参考 deploy-harmonyos/ 目录
# FusionRoute 模块暂不支持手机端（需 PyTorch）
```

---

## 4. Watchdog 自动采集

### 启动

```bash
# 前台（调试）
python src/fusion_route/watchdog.py --verbose

# 后台守护
python src/fusion_route/watchdog.py --daemon

# 单次检查
python src/fusion_route/watchdog.py --once
```

### 工作原理

```
┌─────────────────────────────────────────────┐
│  watchdog.py                                │
│                                             │
│  每 30s 检查 desktop-v2.db                  │
│      ↓                                      │
│  检测到新消息 → 记录时间戳                   │
│      ↓                                      │
│  60s 无新消息（session 结束）                │
│      ↓                                      │
│  自动提取新 prompts → freebuff_prompts.jsonl │
│  自动生成 CDPO 对 → freebuff_prompts_cdpo.jsonl │
│      ↓                                      │
│  记录已处理的 max_seq，避免重复              │
└─────────────────────────────────────────────┘
```

### 多账号支持

每个 Freebuff 实例有独立的 `desktop-v2.db`。Watchdog 自动发现：
- `C:\Users\*\Desktop\fastprefill\.freebuff\desktop-v2.db`
- `C:\Users\*\Documents\*\.freebuff\desktop-v2.db`

手动指定：
```bash
python watchdog.py --db "C:\Users\other_user\Desktop\fastprefill\.freebuff\desktop-v2.db"
```

---

## 5. 训练数据格式

### freebuff_prompts.jsonl

```json
{
  "id": "freebuff_829a380b_491",
  "thread_id": "829a380b-c8b6-4a6f-9ab2-dada4348b306",
  "category": "debugging",
  "prompt": "鸿蒙系统在移植llama-speculative-simple...",
  "quality_score": 0.65,
  "hash": "a1b2c3d4e5f6"
}
```

### freebuff_prompts_cdpo.jsonl

```json
{
  "chosen": {
    "prompt": "完整的 coding prompt...",
    "category": "debugging",
    "expert_hint": "ornith",
    "quality": "chosen"
  },
  "rejected": {
    "prompt": "截断/模糊的 prompt...",
    "category": "debugging",
    "expert_hint": "qwen36",
    "quality": "rejected"
  },
  "source_id": "freebuff_829a380b_491"
}
```

---

## 6. KV Translation 校准

### 需要什么

| 资源 | 要求 |
|---|---|
| **硬件** | Mac M4 16GB（或同等算力） |
| **模型** | Qwen3.6-35B-A3B + Ornith-1.5-35B-A3B |
| **时间** | 2-4 小时（500 samples × 1024 token） |
| **磁盘** | ~500MB（KV 数据）+ 117MB（映射矩阵） |

### 输出

```
calibration_data/
├── model_a/               ← Qwen3.6 KV cache
│   ├── kv_000.npz
│   ├── kv_001.npz
│   └── ...
├── model_b/               ← Ornith-1.5 KV cache
│   ├── kv_000.npz
│   └── ...
├── pairs.jsonl            ← 校准对
└── kv_map.json            ← 岭回归映射矩阵（117MB）
```

---

## 7. CDPO 训练

### 快速验证（CPU，5 分钟）

```bash
python src/fusion_route/train_cdpo.py \
  --n_samples 100 \
  --n_epochs 3 \
  --hidden_size 256
```

### 完整训练（Mac GPU，数小时）

```bash
python src/fusion_route/train_cdpo.py \
  --n_samples 5000 \
  --n_epochs 20 \
  --hidden_size 2048 \
  --lr 1e-4
```

### 输出

```
fusion_route_cdpo/
├── router_weights.pt       ← 路由器权重
├── complementary_weights.pt ← logit 修正权重
└── training_log.json       ← 训练曲线
```

---

## 8. 常见问题

### Q: watchdog.py 报 "No Freebuff DB found"

```bash
# 手动指定 DB 路径
python watchdog.py --db "C:\Users\你的用户名\Desktop\fastprefill\.freebuff\desktop-v2.db"
```

### Q: CDPO 训练报 "No module named torch"

```bash
# Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Mac M4
pip install torch
```

### Q: 校准管线报 OOM

```bash
# 减少样本数和序列长度
python calibration_pipeline.py --n_samples 100 --seq_len 512
```

### Q: 如何查看已提取的 prompts

```bash
# 查看分类统计
python src/fusion_route/extract_prompts.py --score 2>&1 | grep "Category"

# 查看高质量 prompts
python -c "
import json
with open('src/fusion_route/training_data/freebuff_prompts.jsonl') as f:
    prompts = [json.loads(l) for l in f]
prompts.sort(key=lambda x: x.get('quality_score',0), reverse=True)
for p in prompts[:5]:
    print(f'{p[\"quality_score\"]:.2f} | {p[\"category\"]} | {p[\"prompt\"][:80]}...')
"
```

---

## 9. 下一步

| 阶段 | 状态 | 说明 |
|---|---|---|
| M0 KV 兼容性 | ✅ 完成 | Qwen3.6 ↔ Ornith-1.5 完全匹配 |
| M1 KV Translation 原型 | ✅ 完成 | 岭回归映射器 + cosine 0.995 |
| M2 FusionRoute Router | ✅ 完成 | Qwen3Router + Complementary Logit |
| M3 校准管线 | ✅ 完成 | Mac 一键执行脚本 |
| M4 CDPO 训练 | ✅ 完成 | CPU-only，验证通过 |
| M5 Watchdog 采集 | ✅ 完成 | 自动提取 Freebuff prompts |
| M6 真实 logits 校准 | ⏳ 待 Mac | 需要 Mac 在线运行 |
| M7 端云联调 | ⏳ 待 Mac | 双机 KV Translation 测试 |
| M8 Benchmark 验证 | ⏳ 待 Mac | MMLU ≥ 95%, speed ≥ 80% |

---

## 10. 联系

- GitHub: https://github.com/alexchuang19760730/cgcengine0823
- 分支: `fusionroutemot` / `fusion-route-mot`
- HF 模型: https://huggingface.co/Alexchuang/cgcengine-models
