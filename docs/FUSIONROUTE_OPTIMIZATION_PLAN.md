# FusionRoute+MoT 优化方案计划书

> 定位：从"质量增强"升级为"完整推理调度系统"，取代 Hermes Router + 4D Perception Matrix + 10-Step Pipeline
>
> 版本: v1.0 | 日期: 2026-09-01 | 分支: fusionroutemot

---

## 0. 现状差距分析

### 当前 FusionRoute+MoT 能做什么

| 能力 | 状态 | 说明 |
|---|---|---|
| Token-level 路由 | ✅ 已实现 | Qwen3Router, 2 专家 |
| Complementary Logit 修正 | ✅ 已实现 | z_fuse = z_expert + α·c |
| KV Translation 岭回归 | ✅ 已实现 | cosine 0.995 (合成数据) |
| 校准管线 | ✅ 已实现 | Mac 一键执行 |
| CDPO 训练 | ✅ 已实现 | CPU-only 验证通过 |
| Freebuff 数据采集 | ✅ 已实现 | 215 prompts, 414 CDPO, 202 SFT |

### 缺什么才能取代 Hermes + 4D + 10-Step

| 缺失能力 | 对应 Hermes/4D 组件 | 优先级 |
|---|---|---|
| **实时延迟感知** | D1 Network (RTT, bandwidth) | P0 |
| **硬件感知路由** | D2 Hardware (RAM, GPU, CPU) | P0 |
| **模型感知路由** | D3 Model (params, MoE, MTP) | P0 |
| **运行时感知路由** | D4 Runtime (load, memory pressure) | P0 |
| **动态热度跟踪** | Colibrì expert heat | P1 |
| **Speculation ROI 门控** | Colibrì acceptance tracker | P1 |
| **Grammar/JSON draft** | Colibrì grammar draft | P1 |
| **多节点负载均衡** | 10-Step Pipeline 调度 | P1 |
| **故障转移** | 10-Step Pipeline fallback | P1 |
| **P2P 设备发现** | Exo + iroh-net | P2 |
| **模型分片** | Exo model sharding | P2 |
| **分层 Residency** | Colibrì VRAM/RAM/NVMe | P2 |

---

## 1. 目标架构：FusionRoute v2

```
┌─────────────────────────────────────────────────────────────────┐
│                    Perception Layer (取代 4D Matrix)             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Network  │ │ Hardware │ │  Model   │ │ Runtime  │          │
│  │ RTT/BW   │ │ RAM/GPU  │ │ MoE/MTP  │ │ Load/Mem │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       └────────────┼───────────┼────────────┘                  │
│                    ↓                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           FeatureSchema (统一特征向量)                    │   │
│  │  [latency, bandwidth, ram, gpu, model_size, load, ...]  │   │
│  └────────────────────────┬────────────────────────────────┘   │
└───────────────────────────┼────────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│                  FusionRoute Router (取代 Hermes)             │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  Qwen3Router: hidden_state → weight_proj → [e1, e2]  │   │
│  │  + FeatureSchema conditioning                         │   │
│  │  + CDPO trained on real system metrics                │   │
│  └────────────────────────┬──────────────────────────────┘   │
│                           ↓                                   │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  RoutePolicyV2 输出:                                   │   │
│  │  - route_mode: cache_hit | local | edge | cloud       │   │
│  │  - draft_mode: off | mtp | grammar | hybrid           │   │
│  │  - residency: full | warm | streamed                  │   │
│  │  - prefetch: off | conservative | aggressive          │   │
│  │  - confidence: 0.0-1.0                                │   │
│  └────────────────────────┬──────────────────────────────┘   │
└───────────────────────────┼───────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│              KV Translation Layer (MoT 核心)                  │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  RidgeKVMapper: KV_source → W_ridge → KV_target      │   │
│  │  - 40 layers × 2 heads × 256 dim = 117MB 映射矩阵   │   │
│  │  - O(n) 线性映射，跳过 O(n²) prefill                 │   │
│  └────────────────────────┬──────────────────────────────┘   │
└───────────────────────────┼───────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│              Execution Layer (取代 10-Step Pipeline)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ CGC Edge │ │ Expert   │ │ PD       │ │ Exo +    │        │
│  │ Server   │ │ Cache    │ │ Coordinator│ │ iroh-net│        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. 六阶段实施计划

### Phase 1: 4D 感知层 (Week 1-2)

**目标：** 让 Router 看到系统状态，不只是 token hidden state

**做什么：**
1. 扩展 FeatureSchema，加入实时系统指标
2. 每个 edge_server 上报 DeviceProfile（已有）
3. Router 输入从 `[hidden_state]` 变为 `[hidden_state + feature_vector]`

**代码改动：**

```python
# 新增: src/fusion_route/perception.py
class PerceptionLayer:
    """4D 感知层，收集系统状态"""
    
    def collect_features(self) -> FeatureSchema:
        return FeatureSchema(
            # D1: Network
            network_rtt_ms=self.measure_rtt(),
            bandwidth_mbps=self.measure_bandwidth(),
            
            # D2: Hardware
            total_ram_gb=psutil.virtual_memory().total / 1e9,
            available_ram_gb=psutil.virtual_memory().available / 1e9,
            gpu_vram_gb=self.get_gpu_vram(),
            cpu_load_pct=psutil.cpu_percent(),
            
            # D3: Model
            model_params_b=self.model.params,
            is_moe=self.model.is_moe,
            has_mtp=self.model.has_mtp,
            model_size_gb=self.model.size_gb,
            
            # D4: Runtime
            memory_pressure=self.get_memory_pressure(),
            expert_hit_rate=self.expert_cache.hit_rate,
            accept_rate_ema=self.mtp.accept_rate,
        )
```

**验收标准：**
- [ ] FeatureSchema 包含 4D 所有维度
- [ ] 每次路由决策附带完整特征向量
- [ ] 特征采集延迟 < 1ms

---

### Phase 2: 智能路由 (Week 3-4)

**目标：** Router 根据系统状态做路由决策，不只是 token 级专家选择

**做什么：**
1. 路由决策从"选专家"扩展为"选模式+专家+策略"
2. 实现 RoutePolicyV2 输出
3. 实现硬门控（ROI 低于阈值关闭 speculation）

**代码改动：**

```python
# 新增: src/fusion_route/smart_router.py
class SmartRouter:
    """系统感知路由器，取代 Hermes"""
    
    def route(self, request, features: FeatureSchema) -> RoutePolicyV2:
        # 1. Token-level 专家分数
        expert_scores = self.qwen3_router(request.hidden_state)
        
        # 2. 系统状态评估
        system_score = self.evaluate_system(features)
        
        # 3. 综合决策
        if features.available_ram_gb < self.model.size_gb:
            # 内存不足，必须走端云
            mode = "edge_pivot_draft"
            target = self.select_cloud_node(features)
        elif features.expert_hit_rate > 0.9:
            # cache 命中率高，本地 decode
            mode = "cache_hit"
            target = "local"
        elif features.network_rtt_ms < 5:
            # 网络延迟低，可以跨机
            mode = "edge_draft_cloud_verify"
            target = self.select_best_node(features)
        else:
            mode = "local_only"
            target = "local"
        
        # 4. Draft 策略
        draft_mode = "off"
        if features.accept_rate_ema > 0.7:
            draft_mode = "mtp"
        if features.grammar_accept_rate > 0.8:
            draft_mode = "hybrid"
        
        return RoutePolicyV2(
            route_mode=mode,
            draft_mode=draft_mode,
            target_node=target,
            confidence=self.compute_confidence(expert_scores, system_score),
        )
```

**验收标准：**
- [ ] 路由决策包含 mode + target + draft + confidence
- [ ] 内存不足时自动切换端云模式
- [ ] 网络延迟高时自动切换纯端模式
- [ ] Accept rate 低时自动关闭 speculation

---

### Phase 3: KV Translation 生产化 (Week 5-6)

**目标：** 从合成数据验证升级为真实模型校准

**做什么：**
1. Mac 上跑真实校准（Qwen3.6 + Ornith-1.5）
2. 拟合真实映射矩阵
3. 集成到 CGC engine 的 expert-cache

**代码改动：**

```python
# 新增: src/kv_translation/production_translator.py
class ProductionKVTranslator:
    """生产级 KV Translation，集成到 CGC engine"""
    
    def __init__(self, mapping_path: str):
        self.mappings = self.load_mappings(mapping_path)
        # 映射矩阵: {("qwen36", "ornith"): W_ridge [40, 2, 256, 256]}
    
    def translate_kv(self, source_kv, source_model, target_model):
        """O(n) 线性映射，跳过 prefill"""
        W = self.mappings[(source_model, target_model)]
        target_kv = torch.einsum('lhsh, lhst -> lhst', W, source_kv)
        return target_kv
    
    def integrate_with_cgc(self, cgc_engine):
        """集成到 CGC expert-cache"""
        # 在 expert 切换时调用
        cgc_engine.set_kv_translator(self)
```

**验收标准：**
- [ ] 真实模型校准 cosine > 0.95
- [ ] KV Translation 延迟 < 300ms (32K context)
- [ ] 集成到 CGC engine 无 crash

---

### Phase 4: Speculation ROI 门控 (Week 7-8)

**目标：** 让 speculation " earn its keep"，自动开关

**做什么：**
1. 跟踪 accept rate、draft cost、verify cost
2. 计算 ROI = (verify_cost - draft_cost) / draft_cost
3. ROI < 0 时自动关闭 speculation

**代码改动：**

```python
# 新增: src/fusion_route/speculation_guard.py
class SpeculationGuard:
    """Speculation ROI 门控，取代 Colibrì 的 acceptance tracker"""
    
    def __init__(self):
        self.accept_rate_ema = 0.0
        self.draft_cost_ema = 0.0
        self.verify_cost_ema = 0.0
        self.roi_ema = 0.0
    
    def update(self, accepted: bool, draft_ms: float, verify_ms: float):
        """每次 speculate 后更新"""
        self.accept_rate_ema = 0.9 * self.accept_rate_ema + 0.1 * accepted
        self.draft_cost_ema = 0.9 * self.draft_cost_ema + 0.1 * draft_ms
        self.verify_cost_ema = 0.9 * self.verify_cost_ema + 0.1 * verify_ms
        
        if self.draft_cost_ema > 0:
            self.roi_ema = (self.verify_cost_ema - self.draft_cost_ema) / self.draft_cost_ema
    
    def should_speculate(self) -> bool:
        """决定是否启用 speculation"""
        if self.roi_ema < 0:
            return False  # speculation 亏本，关闭
        if self.accept_rate_ema < 0.5:
            return False  # accept rate 太低
        return True
    
    def get_draft_depth(self) -> int:
        """根据 ROI 决定 draft 深度"""
        if self.roi_ema > 2.0:
            return 4  # 高 ROI，激进 draft
        elif self.roi_ema > 0.5:
            return 2  # 中等 ROI，保守 draft
        else:
            return 0  # 低 ROI，关闭
```

**验收标准：**
- [ ] Accept rate < 50% 时自动关闭 MTP
- [ ] Draft cost > verify cost 时自动关闭
- [ ] 所有决策写入 JSON 报告

---

### Phase 5: Exo + iroh-net P2P 整合 (Week 9-10)

**目标：** 设备发现和路由从 HTTP 轮询升级为 P2P mesh

**做什么：**
1. 用 iroh-net 替换 HTTP heartbeat
2. 用 Exo 做模型分片（大模型跨设备）
3. 整合到 CGC discovery

**代码改动：**

```python
# 新增: src/fusion_route/p2p_discovery.py
class IrohDiscovery:
    """P2P 设备发现，取代 HTTP heartbeat"""
    
    def __init__(self):
        self.node = iroh.new_node()  # iroh-net P2P node
    
    def register(self, device_profile: DeviceProfile):
        """注册到 P2P 网络"""
        self.node.publish("cgc/device", device_profile.to_json())
    
    def discover(self) -> List[DeviceProfile]:
        """发现所有在线设备"""
        return self.node.subscribe("cgc/device")
    
    def get_latency(self, peer_id: str) -> float:
        """测量到 peer 的延迟"""
        return self.node.ping(peer_id)


class ExoModelSharder:
    """Exo 模型分片，大模型跨设备加载"""
    
    def shard_model(self, model_path: str, devices: List[DeviceProfile]):
        """把大模型分片到多个设备"""
        total_params = self.get_model_params(model_path)
        shards = []
        for device in devices:
            shard_size = total_params * (device.ram_gb / total_ram)
            shards.append(Shard(device=device, size=shard_size))
        return shards
```

**验收标准：**
- [ ] P2P 发现延迟 < 100ms
- [ ] 设备上下线自动感知
- [ ] 模型分片跨设备推理无 crash

---

### Phase 6: 端到端联调 (Week 11-12)

**目标：** 全组件联调，验证完整流程

**做什么：**
1. Mac + Windows + 鸿蒙 PC + 鸿蒙手机四机联调
2. 跑 Terminal-Bench 验证 Agent 能力
3. 性能基准测试

**验收标准：**
- [ ] 四机 PD 分离推理成功
- [ ] KV Translation 跨机工作
- [ ] Speculation ROI 门控生效
- [ ] Agent Harness 端到端任务完成率 > 60%
- [ ] 稳态速度 Mac 20+ t/s, 手机 2+ t/s

---

## 3. 与 Hermes/4D/10-Step 的对比

| 维度 | Hermes Router | FusionRoute v2 (目标) |
|---|---|---|
| **路由粒度** | Query-level | **Token-level** |
| **输入** | 静态 FeatureSchema | **动态 FeatureSchema + hidden state** |
| **学习方式** | SFT on rules | **CDPO on real metrics** |
| **KV 管理** | 无 | **KV Translation (O(n) 映射)** |
| **Speculation** | 固定策略 | **动态 ROI 门控** |
| **多节点** | HTTP heartbeat | **P2P mesh (iroh-net)** |
| **模型分片** | 不支持 | **Exo model sharding** |

| 维度 | 4D Perception Matrix | FusionRoute v2 (目标) |
|---|---|---|
| **决策者** | LLM 建议 + 契约投影 | **Router 直接决策 + 硬门控** |
| **延迟** | LLM 推理延迟 | **< 1ms 线性投影** |
| **确定性** | LLM 输出不确定 | **Router 输出可验证** |
| **成本** | 需要额外 LLM 推理 | **零额外推理成本** |

| 维度 | 10-Step Pipeline | FusionRoute v2 (目标) |
|---|---|---|
| **步骤数** | 10 步串行 | **4 步并行** |
| **调度** | 静态规则 | **动态 Router** |
| **容错** | 手动 fallback | **自动 ROI 门控** |
| **扩展性** | 加节点需改配置 | **P2P 自动发现** |

---

## 4. 资源需求

| 资源 | 用途 | 时间 |
|---|---|---|
| **Mac M4 16GB** | KV 校准 + 模型推理 | 2-4 小时/次 |
| **Windows 8GB** | 代码开发 + 测试 | 持续 |
| **WSL2** | PyTorch 训练 | 持续 |
| **鸿蒙手机** | 端侧测试 | 联调时 |
| **GPU (可选)** | 加速 CDPO 训练 | 有则更好 |

### 本机可做的 vs 需要 Mac 的

| 任务 | 本机 (Windows) | Mac M4 |
|---|---|---|
| 代码开发 | ✅ 全部 | - |
| 单元测试 | ✅ WSL2 | - |
| CDPO 训练 (mock) | ✅ CPU | - |
| KV 校准 | - | ✅ 必须 |
| 端云联调 | ✅ | ✅ |
| Agent Harness | ✅ WSL2 | ✅ |
| Benchmark | - | ✅ 必须 |

---

## 5. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| KV Translation cosine < 0.95 | 质量退化 | 增加校准数据量，调 lambda |
| P2P 发现延迟高 | 路由决策慢 | 保留 HTTP fallback |
| Speculation ROI 为负 | 速度下降 | 自动关闭，回退 plain decode |
| 四机联调网络不稳定 | 测试失败 | 单机 mock + 真实环境分离 |
| Ornith-1.5 GGUF 不兼容 | MTP 失败 | 保留 Nail-denseIQ4X 作为 fallback |

---

## 6. 成功指标

| 指标 | 当前 | Phase 3 后 | Phase 6 后 |
|---|---|---|---|
| 路由延迟 | N/A | < 1ms | < 1ms |
| KV Translation cosine | 0.995 (synthetic) | > 0.95 (real) | > 0.95 |
| Mac decode 速度 | 27 t/s (单模型) | 20-24 t/s (2 专家) | 20-24 t/s |
| Agent 任务完成率 | ~50% | ~60% | **> 70%** |
| Speculation 自动开关 | 手动 | ✅ 自动 | ✅ 自动 |
| P2P 设备发现 | HTTP 轮询 | HTTP + P2P | ✅ P2P mesh |
| 故障转移 | 手动 | 半自动 | ✅ 全自动 |

---

## 7. 下一步行动

| 行动 | 负责 | 时间 |
|---|---|---|
| 完成 PerceptionLayer 代码 | 本机 | 本周 |
| Mac 上跑真实 KV 校准 | Mac | 下周 |
| 实现 SpeculationGuard | 本机 | 本周 |
| 写 RoutePolicyV2 schema | 本机 | 本周 |
| iroh-net P2P 原型 | 本机 | Phase 5 |

---

> 这份计划的核心思路：**FusionRoute+MoT 不只是"更好的 Router"，而是"完整的推理调度系统"**。
> 它取代 Hermes 的是 token-level 路由 + 系统感知，取代 4D Matrix 的是零延迟特征收集，
> 取代 10-Step Pipeline 的是 4 步并行 + 自动容错。
