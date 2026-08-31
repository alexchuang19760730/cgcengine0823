# HarmonyOS Vulkan 崩溃分析：Expert Cache + L4 Skip Load

> 崩溃现场：`llama-speculative-simple` + `-ngl 30` + expert-cache + L4 skip load + Vulkan
> 平台：HarmonyOS NEXT (Kirin 9030), 24GB RAM
> 信号：SIGSEGV (si_code=1, SEGV_MAPERR), thread OS_FFRT_2_19

---

## 崩溃现场摘要

| 指标 | 值 | 含义 |
|---|---|---|
| si_code | 1 (SEGV_MAPERR) | 访问未映射地址，非 OOM |
| RSS 峰值 | ~976 MB | 远低于 24GB，排除 OOM |
| Major faults | +1031 / 8s (129/s) | L4 pread 正在工作 |
| 崩溃位置 | stderr 停在 `il=11` | 第 12 层首次专家写入/GPU 读取 |
| 线程 | OS_FFRT_2_19 | FFRT 异步调度线程 |

---

## Bug A（已确认，代码实锤）：LAYER_CAPS 领养容量不一致

### 问题

三个位置对 "每层 slot 数" 的计算不一致：

| 位置 | 代码 | 值 |
|---|---|---|
| **Loader** (model-loader.cpp:1403) | `t_meta.ne[2] = cgc_layer_cap(l4_il, expert_cache_pool_capacity)` | per-layer (e.g., 256) |
| **Cache** (expert-cache.cpp:1506) | `cache->n_slots_l[l] = cgc_layer_cap(l, cache->n_slots)` | per-layer (e.g., 256) |
| **Adopt** (llama.cpp:407) | `adopt_pool_region(... model->expert_cache_pool_capacity ...)` | **统一 53** |

### 后果

当 `LLAMA_EXPERT_CACHE_LAYER_CAPS=40-40:256` 时：
- 第 40 层 buffer 大小 = 256 slots
- Cache slot 向量 = 256 slots
- **但 adopt 只记录 53 slots** → slot 53–255 永不被使用

这不是崩溃的直接原因（崩溃在第 12 层，不在第 40 层），但是一个真实的 correctness bug。

### 修复

```cpp
// llama.cpp:407 — 改为 per-layer capacity
llama_expert_cache_adopt_pool_region(model->expert_cache,
        ref.layer, ref.kind, (const uint8_t *) ref.tensor->data,
        (int64_t) cgc_layer_cap(ref.layer, (uint32_t) model->expert_cache_pool_capacity),
        ref.expert_bytes);
```

---

## Bug B（崩溃主嫌疑）：Vulkan host-visible 泛化缺口

### 根因链路

```
1. tensor->data 指向 Vulkan buffer（通过 select_weight_buft 选择）
2. adopt_pool_region 记录 tensor->data 作为 pool_ext base 指针
3. fill_pool_direct → pread() 直接写入 pool_ext[layer][kind] + slot_idx * stride
4. pread() 写入 Vulkan buffer → 如果 buffer 非 host-visible → SEGV_MAPERR
```

### 为什么 Metal 不崩但 Vulkan 崩

| 平台 | GPU buffer 内存模型 | tensor->data 可写？ |
|---|---|---|
| **macOS (Metal)** | 统一内存 (UMA)，所有 GPU buffer 天然 host-visible | ✅ |
| **HarmonyOS (Vulkan)** | 独立显存，buffer 可能是 device-local only | ❌ 如果非 host-visible |

代码里的日志（model-loader.cpp:1275）已经打了 `host=%d`，但 **没有检查/断言**。

### 为什么崩在第 12 层（il=11）

- 第 0–11 层可能碰巧被分配到 host-visible buffer（Vulkan 实现的 fallback）
- 第 12 层首次被分配到 device-local-only buffer → pread 写入未映射地址 → SEGV

### 验证方法

在 Mac 上用 `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` 强制 Vulkan 不用 host-visible buffer，应该能复现同样的崩溃。

### 修复方案

**方案 A（推荐）：L4 pool 强制 host-visible**

在 `select_weight_buft` 选择 buft 后，检查 `ggml_backend_buft_is_host(buft)`：
- 如果为 false → fallback 到 CPU buffer type（expert-cache 不走 GPU zero-copy，走 staging copy）
- 或者设置 `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=0`（确保 Vulkan allocator 优先选 host-visible）

**方案 B：pread → staging copy**

```cpp
// fill_pool_direct 改为：pread 到临时 CPU buffer，然后 memcpy 到 Vulkan buffer
// 需要 ggml_backend_buffer_get_host() 或 ggml_backend_tensor_map()
```

**方案 C（最快验证）：禁用 L4 pool，走 L3-B gather**

```bash
# 不设 LLAMA_EXPERT_CACHE_POOL=1，也不设 expert_cache_bytes
# expert-cache 用 L3-B 路径（staging buffer gather），不直接写 Vulkan buffer
```

---

## 建议下一步

1. **修 Bug A**：`llama.cpp:407` 改用 `cgc_layer_cap`（5 分钟）
2. **验证 Bug B**：在 Mac 上 `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` 复现崩溃（10 分钟）
3. **修 Bug B**：在 adopt 前检查 `ggml_backend_buft_is_host`，非 host-visible 时 fallback（30 分钟）
4. **鸿蒙全量测试**：`-ngl 30` + expert-cache + L4 skip load 跑通 5 次 n=400 benchmark

---

## 代码引用

| 文件 | 行号 | 内容 |
|---|---|---|
| `src/llama.cpp` | 407 | `adopt_pool_region(... expert_cache_pool_capacity ...)` ← Bug A |
| `src/llama-model-loader.cpp` | 1403 | `cgc_layer_cap(l4_il, expert_cache_pool_capacity)` ← 正确 |
| `src/llama-expert-cache.cpp` | 1506 | `cgc_layer_cap(l, cache->n_slots)` ← 正确 |
| `src/llama-expert-cache.cpp` | 30 | `pread(fileno(f), dst, ...)` ← Bug B 写入点 |
| `src/llama-model-loader.cpp` | 1275 | `host=%d` 日志 ← 可验证 host-visible |
| `src/llama-expert-cache.h` | 207 | `cgc_layer_cap()` 定义 |
