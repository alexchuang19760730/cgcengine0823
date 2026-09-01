#!/usr/bin/env python3
"""4D PerceptionLayer — collects system features for SmartRouter.

Replaces the static 4D Perception Matrix with deterministic, zero-cost
feature collection. No LLM involved — pure system probing.

D1: Network quality (RTT, bandwidth, jitter, stability)
D2: Hardware capability (RAM, GPU VRAM, CPU, compute tier)
D3: Model parameters (params, layers, MoE, quantization, per_layer_gb)
D4: Runtime state (memory pressure, expert cache hit rate, speculation ROI)
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class D1Network:
    """D1: Network quality between this node and peers."""
    rtt_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    jitter_ms: float = 0.0
    stability: str = "unknown"  # stable / unstable / unknown

    def to_vector(self) -> list:
        return [self.rtt_ms, self.bandwidth_mbps, self.jitter_ms,
                1.0 if self.stability == "stable" else 0.0]


@dataclass
class D2Hardware:
    """D2: Hardware capability of this node."""
    chip: str = ""
    total_mem_gb: float = 0.0
    avail_mem_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    tflops_fp16: float = 0.0
    tflops_int8: float = 0.0
    engine: str = ""  # mlx / cuda / rocm / cpu
    unified_memory: bool = False
    compute_tier: str = "unknown"  # low / medium / high / ultra
    cpu_cores: int = 0
    cpu_arch: str = ""

    def to_vector(self) -> list:
        # Normalize to 0-1 range for ML
        return [
            min(self.total_mem_gb / 64.0, 1.0),
            min(self.avail_mem_gb / 48.0, 1.0),
            min(self.gpu_vram_gb / 24.0, 1.0),
            min(self.tflops_fp16 / 100.0, 1.0),
            min(self.cpu_cores / 16.0, 1.0),
            1.0 if self.unified_memory else 0.0,
        ]

    @classmethod
    def detect_local(cls) -> "D2Hardware":
        """Auto-detect local hardware."""
        h = cls()
        h.cpu_cores = os.cpu_count() or 0
        h.cpu_arch = platform.machine()
        h.chip = platform.processor() or platform.machine()

        # RAM detection
        try:
            if platform.system() == "Windows":
                import ctypes
                k32 = ctypes.windll.kernel32
                class MS(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                    ]
                m = MS()
                m.dwLength = ctypes.sizeof(MS)
                k32.GlobalMemoryStatusEx(ctypes.byref(m))
                h.total_mem_gb = round(m.ullTotalPhys / (1024**3), 1)
                h.avail_mem_gb = round(m.ullAvailPhys / (1024**3), 1)
            elif platform.system() == "Darwin":
                r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True, timeout=5)
                h.total_mem_gb = round(int(r.stdout.strip()) / (1024**3), 1)
                h.avail_mem_gb = h.total_mem_gb  # unified memory approximation
            else:  # Linux
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            h.total_mem_gb = round(int(line.split()[1]) / (1024**2), 1)
                        elif line.startswith("MemAvailable:"):
                            h.avail_mem_gb = round(int(line.split()[1]) / (1024**2), 1)
        except Exception:
            pass

        # GPU detection
        try:
            if platform.system() == "Windows":
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and r.stdout.strip():
                    parts = r.stdout.strip().split(",")
                    h.gpu_name = parts[0].strip()
                    if len(parts) > 1:
                        h.gpu_vram_gb = float(
                            parts[1].strip().replace("MiB", "").replace("GiB", "")
                        ) / 1024
                    h.engine = "cuda"
            elif h.cpu_arch == "arm64":
                h.gpu_name = "Apple Silicon"
                h.gpu_vram_gb = h.total_mem_gb * 0.7  # unified memory
                h.unified_memory = True
                h.engine = "mlx"
        except Exception:
            pass

        # Compute tier
        score = (min(h.total_mem_gb / 64, 1) * 30 +
                 min(h.gpu_vram_gb / 24, 1) * 40 +
                 min(h.cpu_cores / 16, 1) * 30)
        if score >= 80:
            h.compute_tier = "ultra"
        elif score >= 50:
            h.compute_tier = "high"
        elif score >= 25:
            h.compute_tier = "medium"
        else:
            h.compute_tier = "low"

        return h


@dataclass
class D3Model:
    """D3: Model parameters and characteristics."""
    name: str = ""
    params_b: float = 0.0
    num_layers: int = 0
    is_moe: bool = False
    num_experts: int = 0
    experts_per_tok: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    quantization: str = ""
    model_size_gb: float = 0.0
    per_layer_gb: float = 0.0
    has_native_mtp: bool = False
    draft_model_path: str = ""

    def to_vector(self) -> list:
        return [
            min(self.params_b / 70.0, 1.0),
            min(self.num_layers / 80.0, 1.0),
            1.0 if self.is_moe else 0.0,
            min(self.num_experts / 256.0, 1.0),
            min(self.model_size_gb / 30.0, 1.0),
            1.0 if self.has_native_mtp else 0.0,
        ]


@dataclass
class D4Runtime:
    """D4: Runtime state — memory pressure, cache hits, speculation ROI."""
    memory_pressure: float = 0.0  # 0=free, 1=OOM imminent
    expert_cache_hit_rate: float = 0.0
    speculation_accept_rate: float = 0.0
    current_load: float = 0.0  # 0-100
    active_requests: int = 0
    uptime_s: float = 0.0

    def to_vector(self) -> list:
        return [
            self.memory_pressure,
            self.expert_cache_hit_rate,
            self.speculation_accept_rate,
            self.current_load / 100.0,
            min(self.active_requests / 10.0, 1.0),
        ]


@dataclass
class FeatureVector:
    """Complete 4D feature vector for routing decisions."""
    d1: D1Network = field(default_factory=D1Network)
    d2: D2Hardware = field(default_factory=D2Hardware)
    d3: D3Model = field(default_factory=D3Model)
    d4: D4Runtime = field(default_factory=D4Runtime)
    timestamp: float = field(default_factory=time.time)

    def to_flat(self) -> list:
        """Flatten to a single vector for ML model input."""
        return (self.d1.to_vector() + self.d2.to_vector() +
                self.d3.to_vector() + self.d4.to_vector())

    @property
    def dim(self) -> int:
        return len(self.to_flat())

    def to_dict(self) -> dict:
        return {
            "D1_network": self.d1.__dict__,
            "D2_hardware": self.d2.__dict__,
            "D3_model": self.d3.__dict__,
            "D4_runtime": self.d4.__dict__,
            "timestamp": self.timestamp,
            "flat_dim": self.dim,
        }


class PerceptionLayer:
    """Collects 4D features for routing decisions.

    Usage:
        layer = PerceptionLayer()
        features = layer.collect(
            model_name="qwen36_35b",
            model_info={"params_b": 35, "is_moe": True, ...},
            network_peers={"mac-m4": {"rtt_ms": 2, ...}},
        )
        # features.to_flat() → [22 floats] for SmartRouter input
    """

    def __init__(self):
        self._d2_cache: Optional[D2Hardware] = None
        self._d2_cache_time: float = 0
        self._d4: D4Runtime = D4Runtime()

    def collect(
        self,
        model_name: str = "",
        model_info: Optional[dict] = None,
        network_peers: Optional[dict] = None,
        expert_cache_stats: Optional[dict] = None,
        speculation_stats: Optional[dict] = None,
    ) -> FeatureVector:
        """Collect all 4D features."""
        fv = FeatureVector()

        # D1: Network (from peers or default)
        if network_peers:
            # Use average RTT to closest peer
            rtts = [p.get("rtt_ms", 50) for p in network_peers.values()]
            fv.d1.rtt_ms = min(rtts) if rtts else 50
            fv.d1.bandwidth_mbps = max(
                (p.get("bandwidth_mbps", 100) for p in network_peers.values()),
                default=100)
            fv.d1.stability = "stable" if fv.d1.rtt_ms < 100 else "unstable"
        else:
            fv.d1.rtt_ms = 0  # local
            fv.d1.bandwidth_mbps = 1000
            fv.d1.stability = "stable"

        # D2: Hardware (cached, refresh every 30s)
        now = time.time()
        if self._d2_cache is None or now - self._d2_cache_time > 30:
            self._d2_cache = D2Hardware.detect_local()
            self._d2_cache_time = now
        fv.d2 = self._d2_cache

        # D3: Model
        if model_info:
            fv.d3.name = model_name
            fv.d3.params_b = model_info.get("params_b", 0)
            fv.d3.num_layers = model_info.get("num_layers", 0)
            fv.d3.is_moe = model_info.get("is_moe", False)
            fv.d3.num_experts = model_info.get("num_experts", 0)
            fv.d3.experts_per_tok = model_info.get("experts_per_tok", 0)
            fv.d3.hidden_size = model_info.get("hidden_size", 0)
            fv.d3.model_size_gb = model_info.get("model_size_gb", 0)
            fv.d3.has_native_mtp = model_info.get("has_native_mtp", False)

        # D4: Runtime
        if expert_cache_stats:
            self._d4.expert_cache_hit_rate = expert_cache_stats.get("hit_rate", 0)
        if speculation_stats:
            self._d4.speculation_accept_rate = speculation_stats.get("accept_rate", 0)

        # Memory pressure: 1 - (available / total)
        if fv.d2.total_mem_gb > 0:
            self._d4.memory_pressure = max(0, 1 - fv.d2.avail_mem_gb / fv.d2.total_mem_gb)

        fv.d4 = self._d4
        return fv


# ── Self-test ──
if __name__ == "__main__":
    layer = PerceptionLayer()
    fv = layer.collect(
        model_name="qwen36_35b",
        model_info={
            "params_b": 35, "is_moe": True, "num_experts": 256,
            "num_layers": 40, "hidden_size": 2048, "model_size_gb": 13.0,
            "has_native_mtp": True,
        },
    )
    import json
    print(json.dumps(fv.to_dict(), indent=2))
    print(f"\nFlat vector: {fv.dim} dimensions")
    print(f"Values: {[round(v, 3) for v in fv.to_flat()]}")
