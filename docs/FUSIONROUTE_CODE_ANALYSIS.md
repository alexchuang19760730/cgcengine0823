# FusionRoute 开源代码分析 + CGC 整合方案

## 仓库概况

| 项目 | 值 |
|---|---|
| **仓库** | `xiongny/FusionRoute` |
| **Stars** | 8 |
| **语言** | Python (PyTorch + HuggingFace) |
| **论文** | arXiv:2601.05106 (2026) |
| **License** | 未声明（研究原型） |
| **状态** | 研究原型，无预训练权重、无数据集、无生产封装 |

## 代码结构

```
FusionRoute/
├── code/
│   ├── router.py                    # Router 网络定义（LlamaRouter / GemmaRouter）
│   ├── transfer_ma_llama3.py        # 核心：多模型协作推理引擎
│   ├── transfer_ma_llama3_train.py  # 训练用协作引擎
│   ├── train_sft_llama3.py          # Phase 1: SFT 训练 Router
│   ├── train_dpo_llama3_mix.py      # Phase 2: CDPO 训练 Router
│   ├── run_llama3.py                # 推理入口
│   ├── run_llama3.sh                # 推理脚本
│   ├── utils.py                     # 工具函数
│   └── requirements.txt             # 依赖
├── data_process/                    # 数据处理
└── README.md
```

## 核心架构解析

### 1. Router 网络 (`router.py`)

```python
class Router(LlamaForCausalLM):
    def __init__(self, config, n=3):
        super().__init__(config)
        self.n = n  # 专家数量
        self.weight_proj = nn.Linear(config.hidden_size, self.n)  # ← 关键：hidden → n 维分数

    def forward(self, input_ids, ...):
        outputs = super().forward(input_ids, output_hidden_states=True, ...)
        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        scores = self.weight_proj(last_hidden)    # [batch, seq_len, n] ← token-level 路由分数
        return outputs, scores
```

**关键设计：**
- Router 基于 LlamaForCausalLM（或 Gemma2ForCausalLM）
- 在最后一层 hidden state 上加一个线性投影 → n 维分数
- **每个 token 独立路由**（不是 query-level）
- 支持 QwenRouter（代码里有但 README 没提）

### 2. 多模型协作推理 (`transfer_ma_llama3.py`)

```python
class ARGS:
    def __init__(self, idx, llm_list, n, rm, ...):
        # 加载 3 个 LLM
        self.LLM1 = AutoModelForCausalLM.from_pretrained(llm1).to(gpu1)
        self.LLM2 = AutoModelForCausalLM.from_pretrained(llm2).to(gpu2)
        self.LLM3 = AutoModelForCausalLM.from_pretrained(llm3).to(gpu3)
        
        # 加载 Reward Model（或 Router）
        self.RM = AutoModelForSequenceClassification.from_pretrained(rm)
```

**推理流程：**
1. Router 对每个 token 计算路由分数
2. 根据分数选择 top-k 专家
3. 每个专家独立生成 token
4. **Complementary Logit 修正**：`z_fuse = z_expert + c`
5. 选择最终 token

### 3. CDPO 训练 (`train_dpo_llama3_mix.py`)

```python
class RouterDPOTrainer(DPOTrainer):
    def __init__(self, search_obj, length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = self.search.router
        self.gamma = 0
```

**训练目标：**
- Phase 1: SFT（监督微调 Router）
- Phase 2: CDPO（对比 DPO，学习最优路由策略）
- Loss: `L = -log π(y_t|s_t) + β · D_KL(π || π_ref)`

## 与 CGC Engine 整合分析

### 现有 CGC Engine 组件

| 组件 | FusionRoute 对应 | 整合难度 |
|---|---|---|
| `llama-expert-cache.cpp` | KV cache 管理 | ⚠️ 需扩展 |
| `llama-speculative-simple.exe` | 推理引擎 | ⚠️ 需包装 |
| `edge_server.py` | API 层 | ✅ 可复用 |
| `discovery.py` | 设备发现 | ✅ 可复用 |
| `ComputeRouter` (Hermes) | 路由决策 | ⚠️ 需替换/增强 |

### 整合路径

#### 路径 A：CGC 原生整合（推荐）

```
FusionRoute Router (Python)
    ↓ token-level 分数
CGC Expert Cache (C++)
    ↓ KV cache 管理
llama-speculative-simple (C++)
    ↓ 推理
Complementary Logit (Python)
    ↓ 修正
最终输出
```

**优势：** 复用 CGC 的 expert-cache + MTP + 端云 PD 分离
**劣势：** Python ↔ C++ 边界开销

#### 路径 B：纯 Python 整合（快速原型）

```
FusionRoute Router (Python)
    ↓ token-level 分数
vLLM / Transformers (Python)
    ↓ 推理（多模型并行）
Complementary Logit (Python)
    ↓ 修正
最终输出
```

**优势：** 快速验证，无需改 C++ 代码
**劣势：** 失去 CGC 的 expert-cache + MTP 优化

### KV Translation 整合

**FusionRoute 原版没有 KV Translation。** 这是我们的增量创新。

```
现有 FusionRoute:
  Router → 选择专家 → 专家重新 prefill → decode
  
我们的增强版:
  Router → 选择专家 → KV Translation (线性映射) → decode
                      ↑ 跳过 prefill，O(n) 而非 O(n²)
```

## 需要新增的代码模块

### 1. KV Translation Engine (`src/kv_translation/`)

```python
class KVTranslator:
    def __init__(self):
        self.mappings = {}  # {("qwen36", "ornith"): W_ridge}
    
    def fit(self, source_kv, target_kv, lambda_reg=1.0):
        """拟合岭回归映射矩阵"""
        # source_kv: [n_layers, n_heads, seq_len, head_dim]
        # target_kv: [n_layers, n_heads, seq_len, head_dim]
        W = []
        for layer in range(n_layers):
            for head in range(n_heads):
                X = source_kv[layer, head]  # [seq_len, head_dim]
                Y = target_kv[layer, head]  # [seq_len, head_dim]
                W_layer_head = ridge_regression(X, Y, lambda_reg)
                W.append(W_layer_head)
        return W
    
    def translate(self, source_kv, W):
        """用拟合好的矩阵转换 KV cache"""
        target_kv = []
        for layer, W_layer in enumerate(W):
            for head, W_head in enumerate(W_layer):
                target_kv[layer, head] = source_kv[layer, head] @ W_head
        return target_kv
```

### 2. FusionRoute + MoT 推理引擎

```python
class FusionRouteMoT:
    def __init__(self, router, models, kv_translator):
        self.router = router
        self.models = models  # {"qwen36": model, "ornith": model}
        self.kv_translator = kv_translator
        self.current_model = None
        self.current_kv = None
    
    def generate(self, prompt, max_tokens):
        tokens = tokenize(prompt)
        for step in range(max_tokens):
            # 1. Router 计算路由分数
            scores = self.router(tokens)  # [n_experts]
            
            # 2. 选择最佳专家
            expert_name = self.select_expert(scores)
            
            # 3. KV Translation（如果切换了专家）
            if expert_name != self.current_model:
                if self.current_kv is not None:
                    # 核心：线性映射 KV cache
                    W = self.kv_translator.get_mapping(self.current_model, expert_name)
                    self.current_kv = self.kv_translator.translate(self.current_kv, W)
                self.current_model = expert_name
            
            # 4. 用当前专家 decode 一个 token
            token, kv = self.models[expert_name].decode(tokens, self.current_kv)
            self.current_kv = kv
            
            # 5. Complementary Logit 修正
            if scores.max() < threshold:
                token = self.complementary_correct(token, scores)
            
            tokens = append(tokens, token)
        
        return detokenize(tokens)
```

### 3. 校准管线

```python
def calibrate_kv_mappings(model_a, model_b, calibration_data):
    """
    生成 KV 映射矩阵
    calibration_data: 500 条 × 1024 token 的校准序列
    """
    mappings = []
    for batch in calibration_data:
        # 提取两个模型的 KV cache
        kv_a = model_a.get_kv_cache(batch)  # [layers, heads, seq_len, dim]
        kv_b = model_b.get_kv_cache(batch)  # [layers, heads, seq_len, dim]
        mappings.append((kv_a, kv_b))
    
    # 拟合岭回归
    W = KVTranslator()
    W.fit_all(mappings)
    
    # 保存映射矩阵
    save(W, "kv_mappings_{model_a}_{model_b}.pt")
    return W
```

## 里程碑更新（基于代码分析）

| 里程碑 | 原估时 | 更新后 | 变化原因 |
|---|---|---|---|
| **M0** KV head 分析 | 2h | 2h | 不变 |
| **M1** KV Translation | 4 周 | **3 周** | 代码有参考实现，不需要从零写 |
| **M2** FusionRoute Router | 4 周 | **2 周** | 直接复用 `router.py`，改 Qwen 适配 |
| **M3** 端云协同 | 4 周 | **3 周** | CGC edge_server 已有基础 |
| **M4** Benchmark | 4 周 | 3 周 | 不变 |
| **M5** 产品化 | 6 周 | 4 周 | 不变 |

**总工期：6 个月 → 约 4.5 个月**

## 关键发现

### ✅ 可以直接复用

| 代码 | 复用方式 |
|---|---|
| `router.py` → QwenRouter | 改 `Qwen2ForCausalLM` 基类，加 `weight_proj` |
| `transfer_ma_llama3.py` | 参考推理流程，替换为 CGC API 调用 |
| `train_dpo_llama3_mix.py` | CDPO 训练逻辑直接复用 |
| `train_sft_llama3.sh` | SFT 训练脚本改参数即可 |

### ⚠️ 需要新增

| 模块 | 说明 |
|---|---|
| `kv_translation/` | KV cache 线性映射（FusionRoute 没有） |
| `cgc_adapter.py` | CGC engine ↔ FusionRoute 桥接 |
| `calibration/` | 校准数据生成 + 映射拟合 |
| `benchmark/` | MMLU/GSM8K/HumanEval 测试框架 |

### ❌ 不可复用

| 代码 | 原因 |
|---|---|
| 硬编码的 `/fsx/` 路径 | 需要改为相对路径 |
| `torch.cuda` 依赖 | 需要加 CPU fallback |
| 3 GPU 假设 | 需要改为单 GPU / CPU 模式 |

---

*Created: 2026-09-01*
*Based on: xiongny/FusionRoute (arXiv:2601.05106)*
