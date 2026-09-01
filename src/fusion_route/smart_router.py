#!/usr/bin/env python3
"""SmartRouter v3 — Rule-based 4D matrix routing.

Decision tree derived from 5000-sample Hermes SFT v4 data analysis.

No ML training needed — pure rule-based, <1ms inference.

Usage:
    router = SmartRouter()
    decision = router.route(features)
    # decision = {"mode": "edge_cloud", "confidence": 0.92, ...}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass
class RouteDecision:
    """SmartRouter output."""
    mode: str = "cloud_only"
    confidence: float = 0.0
    edge_backend: str = "none"
    use_mtp: bool = False
    expected_ttft_ms: float = 0.0
    expected_decode_tps: float = 0.0
    cloud_compute_savings_pct: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


# ── Feature extraction from Hermes 4D matrix ──

def extract_features(user_data: dict) -> dict:
    """Extract raw feature dict from Hermes SFT user data."""
    fdm = user_data.get("four_d_matrix", {})
    d1 = fdm.get("D1_network", {})
    d2 = fdm.get("D2_hardware", {})
    d3 = fdm.get("D3_model", {})
    ctx = fdm.get("context", {})
    rc = user_data.get("request_context", {})

    return {
        "rtt_ms": d1.get("rtt_ms", 50),
        "bandwidth_mbps": d1.get("bandwidth_mbps", 100),
        "total_mem_gb": d2.get("total_mem_gb", 16),
        "avail_mem_gb": d2.get("avail_mem_gb", 8),
        "gpu_vram_gb": d2.get("gpu_vram_gb", 0),
        "tflops_fp16": d2.get("tflops_fp16", 10),
        "unified_memory": d2.get("unified_memory", False),
        "draft_model_size_gb": d3.get("draft_model_size_gb", 0),
        "draft_params_m": d3.get("draft_params_m", 0),
        "has_native_mtp": d3.get("has_native_mtp", False) or d3.get("draft_params_m", 0) > 0,
        "prompt_has_code": ctx.get("prompt_has_code") or rc.get("prompt_has_code", False),
        "history_accept_rate": ctx.get("history_accept_rate", 0.7),
        "cache_hit_rate": ctx.get("cache_hit_rate", 0.5),
        "online": rc.get("online", True),
    }


def to_flat(features: dict) -> list:
    """Convert feature dict to 22-dim flat vector (for PerceptionLayer compat)."""
    return [
        min(math.log1p(features["rtt_ms"]) / math.log1p(10000), 1.0),
        min(features["bandwidth_mbps"] / 1000, 1.0),
        0.0,  # jitter (always 0 in data)
        1.0 if features["online"] else 0.0,
        min(features["total_mem_gb"] / 64, 1.0),
        min(features["avail_mem_gb"] / 48, 1.0),
        min(features["gpu_vram_gb"] / 24, 1.0),
        min(features["tflops_fp16"] / 100, 1.0),
        1.0 if features["unified_memory"] else 0.0,
        min(features["draft_model_size_gb"] / 5, 1.0),
        min(features["draft_params_m"] / 500, 1.0),
        1.0 if features["has_native_mtp"] else 0.0,
        1.0 if features["prompt_has_code"] else 0.0,
        min(max(features["history_accept_rate"], 0), 1.0),
        min(max(features["cache_hit_rate"], 0), 1.0),
        1.0 if features["online"] else 0.0,
        min(features["avail_mem_gb"] / max(features["total_mem_gb"], 1), 1.0),
        1.0 if features["avail_mem_gb"] > features["draft_model_size_gb"] * 10 else 0.0,
        min(features["gpu_vram_gb"] / 24, 1.0),
        max(0, 1.0 - min(math.log1p(features["rtt_ms"]) / math.log1p(10000), 1.0))
            * min(features["bandwidth_mbps"] / 1000, 1.0),
        min(features["tflops_fp16"] / 100, 1.0) * 0.6 + min(features["gpu_vram_gb"] / 24, 1.0) * 0.4,
        min(math.log1p(features["rtt_ms"]) / math.log1p(10000), 1.0),
    ]


# ── Rule-based routing ──

class SmartRouter:
    """Rule-based router using 4D matrix decision tree.

    Thresholds derived from 5000-sample Hermes SFT v4 data analysis:
    - offline (online=False): 85% of local_only samples
    - RTT > 200ms: separates cache_hit from edge/cloud
    - RTT > 2000ms: cloud_only (too slow for cache/edge)
    - GPU > 8GB + low RTT: cloud_mtp
    - Low RTT + online: edge_draft
    """

    def __init__(self, model_path: str = ""):
        # Accept model_path for API compat, but we don't use ML
        pass

    def route(self, features) -> RouteDecision:
        """Route based on 4D features."""
        if isinstance(features, dict):
            return self._route(features)
        elif hasattr(features, "to_flat"):
            vec = features.to_flat()
            return self._route_from_flat(vec)
        elif isinstance(features, list):
            return self._route_from_flat(features)
        else:
            return RouteDecision(mode="cloud_only", reason="invalid features")

    def _route_from_flat(self, vec: list) -> RouteDecision:
        """Route from flat 22-dim vector (backward compat with PerceptionLayer)."""
        rtt_log = vec[0]
        online = vec[3] > 0.5
        gpu_norm = vec[6]
        avail_norm = vec[5]
        has_mtp = vec[11] > 0.5
        cache_rate = vec[14]

        rtt_ms = math.expm1(rtt_log * math.log1p(10000))
        gpu_gb = gpu_norm * 24
        avail_gb = avail_norm * 48

        return self._decide(rtt_ms, online, gpu_gb, avail_gb, has_mtp, cache_rate)

    def _route(self, feat: dict) -> RouteDecision:
        """Route from raw feature dict."""
        return self._decide(
            rtt_ms=feat["rtt_ms"],
            online=feat["online"],
            gpu_gb=feat["gpu_vram_gb"],
            avail_gb=feat["avail_mem_gb"],
            has_mtp=feat["has_native_mtp"],
            cache_rate=feat["cache_hit_rate"],
        )

    def _decide(self, rtt_ms, online, gpu_gb, avail_gb, has_mtp, cache_rate):
        """Core decision tree.

        Derived from Hermes SFT v4 data distributions:
        ┌─────────────────────────────────────────────────────────┐
        │  offline (online=False)                                 │
        │  → local_only (85% of local_only samples are offline)  │
        ├─────────────────────────────────────────────────────────┤
        │  RTT > 2000ms + online                                  │
        │  → cloud_only (too slow for cache/edge)                 │
        ├─────────────────────────────────────────────────────────┤
        │  RTT > 200ms + online                                   │
        │  → cache_hit (RTT=1107ms mean for cache_hit samples)   │
        ├─────────────────────────────────────────────────────────┤
        │  RTT < 50ms + GPU > 8GB                                 │
        │  → cloud_mtp (RTT=22ms, GPU=13GB for cloud_mtp)        │
        ├─────────────────────────────────────────────────────────┤
        │  RTT < 200ms + online                                   │
        │  → edge_draft (RTT=61ms mean for edge_draft)            │
        ├─────────────────────────────────────────────────────────┤
        │  fallback                                                │
        │  → cloud_only                                           │
        └─────────────────────────────────────────────────────────┘
        """
        # Rule 1: Offline → local only
        if not online:
            return RouteDecision(
                mode="local_only",
                confidence=0.85,
                edge_backend="llamacpp" if gpu_gb >= 2 else "cpu",
                use_mtp=False,
                expected_ttft_ms=200,
                expected_decode_tps=25.0 if gpu_gb > 8 else 3.0,
                cloud_compute_savings_pct=1.0,
                reason="offline, local inference",
            )

        # Rule 2a: Very high RTT → cloud only (too slow for cache/edge)
        if rtt_ms > 2000:
            return RouteDecision(
                mode="cloud_only",
                confidence=0.80,
                use_mtp=has_mtp,
                expected_ttft_ms=800 + rtt_ms,
                expected_decode_tps=30.0,
                cloud_compute_savings_pct=0.0,
                reason=f"very high RTT {rtt_ms:.0f}ms, cloud only",
            )

        # Rule 2b: High RTT → cache hit (slow but usable network)
        if rtt_ms > 200:
            return RouteDecision(
                mode="cache_hit",
                confidence=0.90,
                expected_ttft_ms=1.0,
                expected_decode_tps=999.0,
                cloud_compute_savings_pct=1.0,
                reason=f"high RTT {rtt_ms:.0f}ms, cache hit",
            )

        # Rule 3: Low RTT + high GPU → cloud MTP
        if rtt_ms < 50 and gpu_gb >= 8:
            return RouteDecision(
                mode="cloud_mtp",
                confidence=0.92,
                edge_backend="cuda" if gpu_gb >= 8 else "llamacpp",
                use_mtp=has_mtp,
                expected_ttft_ms=500 + rtt_ms,
                expected_decode_tps=40.0,
                cloud_compute_savings_pct=0.0,
                reason=f"low RTT {rtt_ms:.0f}ms, GPU {gpu_gb:.0f}GB, cloud MTP",
            )

        # Rule 4: Low RTT + online → edge draft
        if rtt_ms < 200:
            return RouteDecision(
                mode="edge_draft",
                confidence=0.88,
                edge_backend="llamacpp" if gpu_gb >= 2 else "mlx",
                use_mtp=has_mtp and gpu_gb >= 2,
                expected_ttft_ms=300 + rtt_ms,
                expected_decode_tps=27.0,
                cloud_compute_savings_pct=0.29,
                reason=f"RTT {rtt_ms:.0f}ms, edge draft + cloud verify",
            )

        # Fallback: cloud only
        return RouteDecision(
            mode="cloud_only",
            confidence=0.70,
            use_mtp=has_mtp,
            expected_ttft_ms=800 + rtt_ms,
            expected_decode_tps=30.0,
            cloud_compute_savings_pct=0.0,
            reason=f"fallback, RTT {rtt_ms:.0f}ms",
        )

    def train(self, *args, **kwargs):
        """No-op for API compat. Rule-based router doesn't need training."""
        print("[SmartRouter] Rule-based router, no training needed.")
        return 1.0


# ── Self-test ──
if __name__ == "__main__":
    router = SmartRouter()

    test_cases = [
        {"label": "offline device", "rtt_ms": 5000, "online": False, "gpu_vram_gb": 0},
        {"label": "cache hit (slow net)", "rtt_ms": 1200, "online": True, "gpu_vram_gb": 16},
        {"label": "cloud MTP (fast net + GPU)", "rtt_ms": 15, "online": True, "gpu_vram_gb": 24},
        {"label": "edge draft (medium net)", "rtt_ms": 80, "online": True, "gpu_vram_gb": 4},
        {"label": "cloud only (very slow)", "rtt_ms": 3000, "online": True, "gpu_vram_gb": 0},
    ]

    for tc in test_cases:
        feat = {
            "rtt_ms": tc["rtt_ms"], "bandwidth_mbps": 500,
            "total_mem_gb": 32, "avail_mem_gb": 16,
            "gpu_vram_gb": tc["gpu_vram_gb"], "tflops_fp16": 50,
            "unified_memory": False, "draft_model_size_gb": 1.0,
            "draft_params_m": 200, "has_native_mtp": True,
            "prompt_has_code": True, "history_accept_rate": 0.75,
            "cache_hit_rate": 0.6, "online": tc["online"],
        }
        d = router._route(feat)
        print(f"[{tc['label']}] RTT={tc['rtt_ms']}ms, online={tc['online']}, GPU={tc['gpu_vram_gb']}GB")
        print(f"  → mode={d.mode}, conf={d.confidence:.2f}, reason={d.reason}")
