#!/usr/bin/env python3
"""SmartRouter — ML-based routing decisions, replaces hardcoded rules.

Trained on Hermes SFT v4 data (5000 samples, 4D matrix → 5 routing modes).
Uses a small MLP (22 inputs → 128 → 5 outputs) for <1ms inference.

Five routing modes (from Hermes v4):
  - cache_hit:    L1-L5 cache hit, return immediately
  - local_only:   offline/privacy, local inference
  - edge_draft:   local draft + cloud verify (saves cloud compute)
  - cloud_mtp:    cloud NEXTN MTP (draft+verify both on cloud)
  - cloud_only:   direct cloud (MTP degraded/unavailable)

Usage:
    router = SmartRouter()
    router.train("data/hermes_sft_train_v4.jsonl", "data/hermes_sft_eval_v4.jsonl")
    decision = router.route(feature_vector)
    # decision = {"mode": "edge_draft", "confidence": 0.92, ...}
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Routing mode definitions ──

ROUTING_MODES = ["cache_hit", "local_only", "edge_draft", "cloud_mtp", "cloud_only"]
MODE_TO_ID = {m: i for i, m in enumerate(ROUTING_MODES)}
ID_TO_MODE = {i: m for m, i in MODE_TO_ID.items()}

# Feature dimensions (must match PerceptionLayer.to_flat())
# D1: rtt, bandwidth, jitter, stability = 4
# D2: total_mem, avail_mem, gpu_vram, tflops, cpu_cores, unified_mem = 6
# D3: params_b, num_layers, is_moe, num_experts, model_size, has_mtp = 6
# D4: mem_pressure, cache_hit, spec_accept, load, active_req = 5
FEATURE_DIM = 21


@dataclass
class RouteDecision:
    """SmartRouter output."""
    mode: str = "cloud_only"
    confidence: float = 0.0
    edge_backend: str = "none"  # mlx / llamacpp / none
    use_mtp: bool = False
    expected_ttft_ms: float = 0.0
    expected_decode_tps: float = 0.0
    cloud_compute_savings_pct: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "confidence": round(self.confidence, 3),
            "edge_backend": self.edge_backend,
            "use_mtp": self.use_mtp,
            "expected_ttft_ms": round(self.expected_ttft_ms, 1),
            "expected_decode_tps": round(self.expected_decode_tps, 1),
            "cloud_compute_savings_pct": round(self.cloud_compute_savings_pct, 3),
            "reason": self.reason,
        }


# ── Neural network ──

if HAS_TORCH:
    class RouterMLP(nn.Module):
        """Small MLP for routing decisions: 22 → 128 → 64 → 5."""

        def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden // 2, len(ROUTING_MODES)),
            )

        def forward(self, x):
            return self.net(x)  # logits [batch, 5]


# ── Dataset ──

class HermesSFTDataset:
    """Load Hermes SFT v4 data for training/eval."""

    def __init__(self, jsonl_path: str, max_samples: int = 0):
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                d = json.loads(line)
                self._parse(d)

    def _parse(self, d):
        msgs = d.get("messages", [])
        if len(msgs) < 3:
            return

        # Extract user content (4D matrix)
        user_msg = msgs[1]["content"]
        try:
            user_data = json.loads(user_msg)
        except json.JSONDecodeError:
            return

        # Extract assistant content (routing decision)
        assistant_msg = msgs[2]["content"]
        try:
            ans = json.loads(assistant_msg)
        except json.JSONDecodeError:
            return

        mode = ans.get("mode", "cloud_only")
        if mode not in MODE_TO_ID:
            return

        # Extract 4D features from user_data
        fdm = user_data.get("four_d_matrix", {})
        d1 = fdm.get("D1_network", {})
        d2 = fdm.get("D2_hardware", {})
        d3 = fdm.get("D3_model", {})
        d4 = fdm.get("D4_context", fdm.get("D4_route", {}))

        features = [
            # D1: network (4)
            d1.get("rtt_ms", 50) / 200.0,
            d1.get("bandwidth_mbps", 100) / 1000.0,
            d1.get("jitter_ms", 0) / 50.0,
            1.0 if d1.get("stability") == "stable" else 0.0,
            # D2: hardware (6)
            min(d2.get("total_mem_gb", 8) / 64.0, 1.0),
            min(d2.get("avail_mem_gb", 4) / 48.0, 1.0),
            min(d2.get("gpu_vram_gb", 0) / 24.0, 1.0),
            min(d2.get("tflops_fp16", 10) / 100.0, 1.0),
            min(d2.get("cpu_cores", 4) / 16.0, 1.0),
            1.0 if d2.get("unified_memory") else 0.0,
            # D3: model (6)
            min(d3.get("params_b", 7) / 70.0, 1.0),
            min(d3.get("num_layers", 28) / 80.0, 1.0),
            1.0 if d3.get("is_moe") else 0.0,
            min(d3.get("num_experts", 0) / 256.0, 1.0),
            min(d3.get("model_size_gb", 4) / 30.0, 1.0),
            1.0 if d3.get("has_native_mtp") else 0.0,
            # D4: runtime (5)
            d4.get("memory_pressure", 0) if "memory_pressure" in d4 else 0.0,
            d4.get("expert_cache_hit_rate", 0) if "expert_cache_hit_rate" in d4 else 0.0,
            d4.get("expected_accept_rate", 0) if "expected_accept_rate" in d4 else 0.0,
            d4.get("confidence", 0.5),
            min(d4.get("draft_n_tokens", 0) / 8.0, 1.0),
        ]

        self.samples.append({
            "features": features,
            "mode_id": MODE_TO_ID[mode],
            "mode": mode,
            "raw": ans,
        })

    def __len__(self):
        return len(self.samples)

    def to_tensors(self):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        X = torch.tensor([s["features"] for s in self.samples], dtype=torch.float32)
        y = torch.tensor([s["mode_id"] for s in self.samples], dtype=torch.long)
        return X, y


# ── SmartRouter ──

class SmartRouter:
    """ML-based router trained on Hermes SFT data.

    Fallback: if model not trained, uses heuristic rules (same as ComputeRouter).
    """

    def __init__(self, model_path: str = ""):
        self.model = None
        self.model_path = model_path
        self._loaded = False

        if model_path and os.path.exists(model_path) and HAS_TORCH:
            self._load(model_path)

    def _load(self, path):
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.model = RouterMLP(input_dim=state.get("input_dim", FEATURE_DIM))
            self.model.load_state_dict(state["model"])
            self.model.eval()
            self._loaded = True
        except Exception as e:
            print(f"[SmartRouter] Failed to load {path}: {e}")

    def route(self, features) -> RouteDecision:
        """Route a request based on 4D features.

        Args:
            features: FeatureVector from PerceptionLayer, or flat list of 21 floats.

        Returns:
            RouteDecision with mode, confidence, etc.
        """
        if hasattr(features, "to_flat"):
            vec = features.to_flat()
        elif isinstance(features, list):
            vec = features
        else:
            return RouteDecision(mode="cloud_only", reason="invalid features")

        # Try ML model
        if self._loaded and HAS_TORCH:
            try:
                x = torch.tensor([vec], dtype=torch.float32)
                with torch.no_grad():
                    logits = self.model(x)
                    probs = F.softmax(logits, dim=-1)[0]
                    mode_id = int(probs.argmax())
                    conf = float(probs[mode_id])
                return self._build_decision(ID_TO_MODE[mode_id], conf, vec)
            except Exception:
                pass

        # Fallback: heuristic rules (same logic as ComputeRouter)
        return self._heuristic(vec)

    def _heuristic(self, vec: list) -> RouteDecision:
        """Fallback heuristic routing when ML model not available."""
        # vec layout: [rtt, bw, jitter, stab, mem, avail, gpu, tflops, cpu, unified,
        #              params, layers, moe, experts, size, mtp, mem_press, cache_hit,
        #              spec_accept, confidence, draft_n]
        rtt = vec[0] * 200
        avail_gb = vec[5] * 48
        gpu_gb = vec[6] * 24
        model_gb = vec[13] * 30
        has_mtp = vec[14] > 0.5
        mem_pressure = vec[15]

        # Rule 0: Memory critical → cloud
        if avail_gb < model_gb * 0.8:
            mode = "cloud_mtp" if has_mtp else "cloud_only"
            return RouteDecision(mode=mode, confidence=0.9,
                                reason=f"OOM risk: {avail_gb:.1f}GB < {model_gb:.1f}GB")

        # Rule 1: Local fits well → local
        if avail_gb >= model_gb * 1.5 and mem_pressure < 0.5:
            return RouteDecision(mode="local_only", confidence=0.85,
                                reason="local fits, low pressure")

        # Rule 2: Has GPU + MTP → edge_draft
        if gpu_gb >= 2 and has_mtp:
            return RouteDecision(mode="edge_draft", confidence=0.88,
                                edge_backend="llamacpp", use_mtp=True,
                                reason="GPU available + MTP supported")

        # Rule 3: Good network → cloud_mtp
        if rtt < 50:
            return RouteDecision(mode="cloud_mtp", confidence=0.8,
                                reason=f"low RTT {rtt:.0f}ms")

        # Default
        return RouteDecision(mode="cloud_only", confidence=0.7,
                            reason="fallback")

    def _build_decision(self, mode: str, conf: float, vec: list) -> RouteDecision:
        """Build a full RouteDecision from mode + confidence."""
        has_mtp = vec[14] > 0.5
        gpu_gb = vec[6] * 24

        edge_backend = "none"
        if gpu_gb >= 2:
            edge_backend = "llamacpp"
        elif vec[9] > 0.5:  # unified memory
            edge_backend = "mlx"

        # Estimate TTFT and decode
        ttft = 100.0  # default ms
        decode_tps = 5.0
        savings = 0.0

        if mode == "local_only":
            decode_tps = 25.0 if gpu_gb > 8 else 5.0
            ttft = 200.0
        elif mode == "edge_draft":
            decode_tps = 27.0
            ttft = 300.0
            savings = 0.29  # 29% cloud compute saved
        elif mode == "cloud_mtp":
            decode_tps = 40.0
            ttft = 500.0
        elif mode == "cloud_only":
            decode_tps = 30.0
            ttft = 800.0
        elif mode == "cache_hit":
            decode_tps = 999.0
            ttft = 1.0
            savings = 1.0

        return RouteDecision(
            mode=mode,
            confidence=conf,
            edge_backend=edge_backend,
            use_mtp=has_mtp and mode in ("edge_draft", "cloud_mtp"),
            expected_ttft_ms=ttft,
            expected_decode_tps=decode_tps,
            cloud_compute_savings_pct=savings,
            reason=f"ML prediction (conf={conf:.2f})",
        )

    def train(
        self,
        train_path: str,
        eval_path: str,
        output_path: str = "models/smart_router.pt",
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 64,
    ):
        """Train the router on Hermes SFT data."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for training")

        print(f"[SmartRouter] Loading training data from {train_path}")
        train_ds = HermesSFTDataset(train_path)
        print(f"[SmartRouter] Loading eval data from {eval_path}")
        eval_ds = HermesSFTDataset(eval_path)

        X_train, y_train = train_ds.to_tensors()
        X_eval, y_eval = eval_ds.to_tensors()
        print(f"[SmartRouter] Train: {len(X_train)} samples, Eval: {len(X_eval)} samples")

        # Check class distribution
        for i, mode in enumerate(ROUTING_MODES):
            n = int((y_train == i).sum())
            print(f"  {mode}: {n} ({n*100//len(X_train)}%)")

        # Create model
        model = RouterMLP(input_dim=FEATURE_DIM)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        best_eval_acc = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            n_batches = 0
            for X_batch, y_batch in loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()

            # Eval
            model.eval()
            with torch.no_grad():
                eval_logits = model(X_eval)
                eval_preds = eval_logits.argmax(dim=-1)
                eval_acc = float((eval_preds == y_eval).float().mean())
                eval_loss = float(criterion(eval_logits, y_eval))

            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
                  f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.3f}")

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                torch.save({
                    "model": model.state_dict(),
                    "input_dim": FEATURE_DIM,
                    "modes": ROUTING_MODES,
                    "eval_acc": eval_acc,
                }, output_path)

        print(f"[SmartRouter] Training complete. Best eval accuracy: {best_eval_acc:.3f}")
        print(f"[SmartRouter] Model saved to {output_path}")
        return best_eval_acc


# ── Self-test ──
if __name__ == "__main__":
    router = SmartRouter()

    # Test heuristic fallback
    from perception import PerceptionLayer, FeatureVector
    layer = PerceptionLayer()
    fv = layer.collect(
        model_name="qwen36_35b",
        model_info={"params_b": 35, "is_moe": True, "num_experts": 256,
                     "num_layers": 40, "model_size_gb": 13.0, "has_native_mtp": True},
    )
    decision = router.route(fv)
    print(json.dumps(decision.to_dict(), indent=2))

    # Train if data available
    train_path = "data/hermes_sft_train_v4.jsonl"
    eval_path = "data/hermes_sft_eval_v4.jsonl"
    if os.path.exists(train_path):
        print("\n=== Training SmartRouter ===")
        acc = router.train(train_path, eval_path, epochs=10)
        print(f"Final accuracy: {acc:.3f}")
    else:
        print(f"\n[skip] Training data not found at {train_path}")
