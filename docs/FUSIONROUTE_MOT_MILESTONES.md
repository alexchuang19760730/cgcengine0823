# FusionRoute + MoT 工程里程碑

> 基于 FusionRoute + KV Translation 协同增益深度解析
> 结合现有 CGC engine 基础设施

---

## 现有基础盘点

| 组件 | 状态 | 位置 |
|---|---|---|
| CGC engine (expert-cache + MTP) | ✅ 生产就绪 | `src/llama.cpp/` |
| Hermes Router (4D 矩阵路由) | ✅ 基本完成 | `agent_harness/` |
| Edge Server (PD 分离) | ✅ 基本完成 | `CGC-main/cgc_engine/pd/` |
| Discovery (设备发现) | ✅ 基本完成 | `CGC-main/cgc_engine/pd/discovery.py` |
| 多模型池 | ✅ 5 个 GGUF | `Alexchuang/cgcengine-models` |
| 端云 PD 分离 | ✅ 已验证 | emit/resume API |
| Agent Harness (DSH) | ✅ E2E 跑通 | `agent_harness/` |

## 缺失组件（需开发）

| 组件 | 说明 | 优先级 |
|---|---|---|
| **KV Translation Engine** | KV cache 线性映射 (岭回归) | 🔴 P0 |
| **FusionRoute Router** | Token-level 路由 + complementary logit | 🔴 P0 |
| **KV Cache Manager** | 跨模型 KV 缓存存储/迁移 | 🔴 P0 |
| **校准管线** | 500 条 × 1024 token 校准序列 | 🟡 P1 |
| **Benchmark 框架** | MMLU/GSM8K/HumanEval/LongBench | 🟡 P1 |
| **OpenAI 兼容 API** | 推理服务接口 | 🟢 P2 |

---

## 里程碑计划

### M0: 基础验证（Week 1-2）— 本机可做 ✅

**目标：** 在现有 CGC engine 上验证 KV head 匹配性

| 任务 | 说明 | 时间 | 本机能做？ |
|---|---|---|---|
| 分析模型 KV head 数 | Qwen3.6 35B vs Ornith-1.5 35B vs Gemma4 26B 的 GQA 分组数 | 2h | ✅ |
| 验证同源专家池 | 确认 Qwen3.6/Ornith-1.5 是同家族（Qwen 架构） | 1h | ✅ |
| 提取 KV cache 样本 | 用现有 llama.cpp 提取单层 KV tensor shape | 4h | ✅ |
| 写 KV head 兼容性报告 | 哪些模型组合可以做 KV Translation | 2h | ✅ |

**产出：** `docs/KV_HEAD_COMPATIBILITY.md`

---

### M1: KV Translation 原型（Week 3-6）— 本机 + Mac

**目标：** 实现最小可运行的 KV cache 线性映射

| 任务 | 说明 | 时间 | 本机能做？ |
|---|---|---|---|
| 实现岭回归映射器 | PyTorch: `W_ridge = (X^T X + λI)^{-1} X^T Y` | 1 周 | ✅ 本机 |
| 生成校准数据 | 500 条 × 1024 token，跑两个模型提取 KV | 1 周 | ⚠️ 需 Mac（本机太慢） |
| 拟合映射矩阵 | `W_qwen36_to_ornith` 等 pairwise | 2 天 | ✅ 本机（纯 CPU） |
| 验证映射精度 | 对比映射后 KV vs 真实 KV 的 cosine similarity | 2 天 | ✅ 本机 |
| 集成到 CGC engine | 在 `llama-expert-cache.cpp` 加 KV transfer 接口 | 1 周 | ⚠️ 需编译（本机可做） |

**产出：** `src/kv_translation/` 模块 + 映射矩阵文件

**关键实验：**
```
Qwen3.6-35B → Ornith-1.5-35B KV Translation
输入: Qwen3.6 的 KV cache (32K token)
输出: 映射后的 KV cache
测量: cosine similarity > 0.95? 生成质量损失 < 2%?
```

---

### M2: FusionRoute Router 原型（Week 7-10）— 本机 + Mac

**目标：** 实现 token-level 路由 + complementary logit 修正

| 任务 | 说明 | 时间 | 本机能做？ |
|---|---|---|---|
| 实现 Router 网络 | 2 层 MLP: hidden_size → num_experts + complementary_dim | 3 天 | ✅ 本机 |
| CDPO 训练数据准备 | 收集 token-level 专家选择标注数据 | 1 周 | ⚠️ 需 Mac 生成 |
| 训练 Router | CDPO loss: `L = -log π(y_t|s_t) + β · D_KL(π || π_ref)` | 1 周 | ❌ 需 GPU（Mac M4 可做） |
| 实现 Complementary Logit | `z_fuse = z_expert + c`，c 由 router 输出 | 3 天 | ✅ 本机 |
| 端到端推理 pipeline | Router → 选择专家 → KV Translation → decode → logit 修正 | 1 周 | ⚠️ 需 Mac 跑模型 |

**产出：** `src/fusion_route/` 模块 + 训练好的 router 权重

---

### M3: 端云动态协同（Week 11-14）— 需 Mac

**目标：** Token-level 端云调度 + KV 迁移

| 任务 | 说明 | 时间 | 本机能做？ |
|---|---|---|---|
| 扩展 edge_server.py | 加 token-level 路由决策点 | 1 周 | ⚠️ 代码在本机写，Mac 测试 |
| 实现端→云 KV 迁移 | emit 时提取 KV，resume 时注入 | 1 周 | ⚠️ 需 Mac |
| 实现云→端 KV 迁移 | 反向映射：云端 KV → 端侧 KV | 1 周 | ⚠️ 需 Mac |
| 动态调度策略 | 简单 token → 端侧，复杂 token → 云端 | 1 周 | ✅ 本机写逻辑 |
| 延迟测量 | 端云切换延迟 < 300ms？ | 2 天 | ⚠️ 需双机联调 |

**产出：** 完整的端云 FusionRoute + MoT pipeline

---

### M4: Benchmark 验证（Week 15-18）— 需 Mac

**目标：** 在标准 benchmark 上验证质量 + 速度

| 任务 | 说明 | 时间 | 本机能做？ |
|---|---|---|---|
| MMLU 测试 | 对比: 单模型 vs FusionRoute+MoT | 2 天 | ❌ 需 Mac |
| GSM8K 数学推理 | 对比: 单模型 vs FusionRoute+MoT | 2 天 | ❌ 需 Mac |
| HumanEval 代码生成 | 对比: 单模型 vs FusionRoute+MoT | 2 天 | ❌ 需 Mac |
| LongBench 长上下文 | 对比: 单模型 vs FusionRoute+MoT | 2 天 | ❌ 需 Mac |
| 速度基准 | tok/s 对比: 11.9 → ? | 1 天 | ❌ 需 Mac |
| 累积误差分析 | KV Translation 误差是否被 logit 修正吸收 | 3 天 | ✅ 本机分析数据 |

**产出：** `docs/BENCHMARK_REPORT.md` + 性能数据表

---

### M5: 产品化（Week 19-24）— 需 Mac + 团队

**目标：** OpenAI 兼容 API + 开源发布

| 任务 | 说明 | 时间 | 本机能做？ |
|---|---|---|---|
| OpenAI 兼容 API | `/v1/chat/completions` + 路由决策 | 1 周 | ✅ 本机写 |
| 配置文件 | 模型池定义 + 路由策略配置 | 2 天 | ✅ 本机 |
| Docker 镜像 | 一键部署 | 3 天 | ⚠️ 需测试 |
| 文档 + README | 使用指南 + API 文档 | 3 天 | ✅ 本机 |
| 开源发布 | GitHub + HuggingFace | 1 天 | ✅ 本机 |

**产出：** 可发布的推理中间件

---

## 本机（Windows 8GB）能做的部分

### ✅ 可以做（不需要 GPU/大内存）

| 任务 | 时间 | 说明 |
|---|---|---|
| KV head 兼容性分析 | 2h | 读 config.json 分析 GQA 参数 |
| 岭回归映射器实现 | 1 周 | 纯 PyTorch，CPU 可跑 |
| 映射矩阵拟合 | 2 天 | 500×1024 数据量，CPU 够用 |
| Router 网络实现 | 3 天 | 2 层 MLP，很小 |
| Complementary logit 实现 | 3 天 | 简单残差修正 |
| OpenAI 兼容 API | 1 周 | Python HTTP server |
| 文档 + 配置 | 1 周 | Markdown + YAML |
| Benchmark 数据分析 | 3 天 | Pandas 分析已有结果 |

**总计：约 4-5 周纯本机工作**

### ⚠️ 需要 Mac（模型推理）

| 任务 | 时间 | 说明 |
|---|---|---|
| 校准数据生成 | 1 周 | 跑两个模型提取 KV cache |
| Router CDPO 训练 | 1 周 | 需要 GPU |
| 端云协同测试 | 1 周 | 需要双机 |
| Benchmark 跑分 | 1 周 | 需要模型推理 |

**总计：约 4 周需 Mac**

### ❌ 本机做不了

| 任务 | 说明 |
|---|---|
| 大规模模型推理 | 13-15GB 模型，8GB RAM 跑不动 |
| Router 训练（大规模） | 需要 GPU + 大内存 |
| 端云联调 | 需要 Mac + Windows 双机 |

---

## 时间线总览

```
Week 1-2   [本机] M0: KV head 兼容性分析
Week 3-4   [本机] M1a: 岭回归映射器 + 校准数据格式
Week 4-6   [Mac]  M1b: 校准数据生成 + 映射矩阵拟合
Week 6-7   [本机] M1c: 集成到 CGC engine
Week 7-9   [本机] M2a: Router 网络 + complementary logit
Week 9-10  [Mac]  M2b: CDPO 训练
Week 10-11 [本机+Mac] M2c: 端到端 pipeline
Week 11-14 [Mac]  M3: 端云动态协同
Week 15-18 [Mac]  M4: Benchmark 验证
Week 19-24 [本机+Mac] M5: 产品化 + 开源
```

**总工期：约 6 个月（1 人全职）**

---

## 里程碑验收标准

| 里程碑 | 验收条件 | 阻塞项 |
|---|---|---|
| **M0** | KV head 兼容性报告完成，确认 Qwen3.6 ↔ Ornith-1.5 可互转 | 无 |
| **M1** | KV Translation cosine similarity > 0.95 | 校准数据（需 Mac） |
| **M2** | FusionRoute+MoT 端到端可运行，质量损失 < 5% | Router 训练（需 GPU） |
| **M3** | 端云切换延迟 < 300ms，token-level 调度正常 | 双机联调 |
| **M4** | MMLU ≥ 95% of single model, speed ≥ 80% of single model | 完整模型推理 |
| **M5** | OpenAI API 可用，文档完整，GitHub 发布 | 团队 |

---

## 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| KV head 数不匹配 | 无法做 KV Translation | M0 阶段验证，不匹配则放弃该组合 |
| 累积误差不可控 | 质量大幅下降 | 先单独验证，再叠加 |
| 本机速度太慢 | 开发效率低 | 关键推理任务全部迁移到 Mac |
| Mac 掉线 | 无法测试 | 本机先写好所有代码，Mac 只做推理 |

---

*Created: 2026-09-01*
*Based on: FusionRoute + MoT 协同增益深度解析*
