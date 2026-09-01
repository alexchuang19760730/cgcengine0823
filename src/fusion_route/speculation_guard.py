#!/usr/bin/env python3
"""SpeculationGuard — ROI-based gating for MTP speculation.

Automatically enables/disables speculation based on whether it "earns its keep":
  ROI = (verify_cost - draft_cost) / draft_cost

If ROI < 0, speculation is wasting compute → disable it.
If accept rate drops below threshold → disable it.

Tracks per-request metrics and maintains a rolling window for decisions.

Usage:
    guard = SpeculationGuard()
    guard.record_request(
        draft_cost_ms=5, verify_cost_ms=2,
        accept_rate=0.95, n_drafted=3, n_accepted=3
    )
    decision = guard.should_use_speculation()
    # decision = {"use": True, "roi": 0.95, "accept_rate": 0.95, ...}
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpeculationMetrics:
    """Metrics for a single speculation attempt."""
    timestamp: float = 0.0
    draft_cost_ms: float = 0.0  # time to draft N tokens
    verify_cost_ms: float = 0.0  # time to verify N tokens
    accept_rate: float = 0.0  # accepted / drafted
    n_drafted: int = 0
    n_accepted: int = 0
    decode_speedup: float = 1.0  # actual speedup vs no-spec baseline


@dataclass
class SpeculationDecision:
    """Output of SpeculationGuard.should_use_speculation()."""
    use_speculation: bool = True
    roi: float = 0.0
    accept_rate_avg: float = 0.0
    reason: str = ""
    confidence: float = 0.0
    draft_n_recommend: int = 2  # recommended draft_n_tokens

    def to_dict(self) -> dict:
        return {
            "use_speculation": self.use_speculation,
            "roi": round(self.roi, 3),
            "accept_rate_avg": round(self.accept_rate_avg, 3),
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "draft_n_recommend": self.draft_n_recommend,
        }


class SpeculationGuard:
    """ROI-based gating for MTP speculation.

    Config:
        window_size:    rolling window for metrics (default 50 requests)
        min_accept_rate: minimum accept rate to keep speculation on (default 0.3)
        min_roi:         minimum ROI to keep speculation on (default 0.0)
        warmup_requests:  requests before making decisions (default 5)
    """

    def __init__(
        self,
        window_size: int = 50,
        min_accept_rate: float = 0.3,
        min_roi: float = 0.0,
        warmup_requests: int = 5,
    ):
        self.window_size = window_size
        self.min_accept_rate = min_accept_rate
        self.min_roi = min_roi
        self.warmup_requests = warmup_requests

        self._history: deque = deque(maxlen=window_size)
        self._total_requests = 0
        self._spec_enabled = True
        self._disable_count = 0
        self._enable_count = 0

    def record_request(
        self,
        draft_cost_ms: float,
        verify_cost_ms: float,
        accept_rate: float,
        n_drafted: int,
        n_accepted: int,
    ):
        """Record metrics from a speculation attempt."""
        metrics = SpeculationMetrics(
            timestamp=time.time(),
            draft_cost_ms=draft_cost_ms,
            verify_cost_ms=verify_cost_ms,
            accept_rate=accept_rate,
            n_drafted=n_drafted,
            n_accepted=n_accepted,
        )

        # ROI calculation
        if draft_cost_ms > 0:
            # ROI = time saved / time spent drafting
            # Each accepted token saves ~verify_cost_ms/n_drafted
            time_saved = n_accepted * (verify_cost_ms / max(n_drafted, 1))
            metrics.decode_speedup = time_saved / max(draft_cost_ms, 0.001)

        self._history.append(metrics)
        self._total_requests += 1

    def should_use_speculation(self) -> SpeculationDecision:
        """Decide whether to use speculation for the next request."""
        # Warmup: don't have enough data yet
        if self._total_requests < self.warmup_requests:
            return SpeculationDecision(
                use_speculation=True,
                reason=f"warmup ({self._total_requests}/{self.warmup_requests})",
                confidence=0.5,
            )

        if not self._history:
            return SpeculationDecision(use_speculation=True, reason="no data")

        # Compute rolling averages
        n = len(self._history)
        avg_accept = sum(m.accept_rate for m in self._history) / n
        avg_draft_cost = sum(m.draft_cost_ms for m in self._history) / n
        avg_verify_cost = sum(m.verify_cost_ms for m in self._history) / n
        avg_speedup = sum(m.decode_speedup for m in self._history) / n

        # ROI = (verify_cost - draft_cost) / draft_cost
        # If draft is expensive relative to verify, speculation is worth it
        if avg_draft_cost > 0:
            roi = (avg_verify_cost * avg_accept - avg_draft_cost) / avg_draft_cost
        else:
            roi = 0.0

        # Decision logic
        use = True
        reason = ""
        confidence = 0.0
        draft_n = 2

        if avg_accept < self.min_accept_rate:
            use = False
            reason = f"accept_rate {avg_accept:.2f} < min {self.min_accept_rate}"
            confidence = min(1.0, (self.min_accept_rate - avg_accept) / self.min_accept_rate)
        elif roi < self.min_roi:
            use = False
            reason = f"ROI {roi:.3f} < min {self.min_roi}"
            confidence = min(1.0, abs(roi - self.min_roi) / max(abs(self.min_roi), 0.01))
        else:
            # Speculation is worth it
            reason = f"ROI {roi:.3f}, accept {avg_accept:.2f}, speedup {avg_speedup:.1f}x"
            confidence = min(1.0, avg_accept * (1 + roi))

            # Recommend draft_n based on accept rate
            if avg_accept >= 0.9:
                draft_n = 3
            elif avg_accept >= 0.7:
                draft_n = 2
            else:
                draft_n = 1  # barely worth it

        # Track state changes
        if use and not self._spec_enabled:
            self._enable_count += 1
        elif not use and self._spec_enabled:
            self._disable_count += 1
        self._spec_enabled = use

        return SpeculationDecision(
            use_speculation=use,
            roi=roi,
            accept_rate_avg=avg_accept,
            reason=reason,
            confidence=confidence,
            draft_n_recommend=draft_n,
        )

    def get_stats(self) -> dict:
        """Get current statistics."""
        if not self._history:
            return {"total_requests": self._total_requests, "window_size": 0}

        n = len(self._history)
        return {
            "total_requests": self._total_requests,
            "window_size": n,
            "avg_accept_rate": sum(m.accept_rate for m in self._history) / n,
            "avg_draft_cost_ms": sum(m.draft_cost_ms for m in self._history) / n,
            "avg_verify_cost_ms": sum(m.verify_cost_ms for m in self._history) / n,
            "disable_count": self._disable_count,
            "enable_count": self._enable_count,
            "currently_enabled": self._spec_enabled,
        }

    def reset(self):
        """Reset all metrics."""
        self._history.clear()
        self._total_requests = 0
        self._spec_enabled = True
        self._disable_count = 0
        self._enable_count = 0


# ── Self-test ──
if __name__ == "__main__":
    import json
    import random

    guard = SpeculationGuard(warmup_requests=3)

    # Simulate 20 requests with varying accept rates
    print("=== Simulating speculation decisions ===\n")
    for i in range(20):
        # Gradually decreasing accept rate
        accept = max(0.1, 0.95 - i * 0.04 + random.uniform(-0.05, 0.05))
        drafted = random.choice([2, 3])
        accepted = int(drafted * accept)
        draft_ms = random.uniform(3, 8)
        verify_ms = random.uniform(1, 3)

        guard.record_request(
            draft_cost_ms=draft_ms,
            verify_cost_ms=verify_ms,
            accept_rate=accept,
            n_drafted=drafted,
            n_accepted=accepted,
        )

        decision = guard.should_use_speculation()
        print(f"  Request {i+1:2d}: accept={accept:.2f} → "
              f"use={decision.use_speculation}, roi={decision.roi:.3f}, "
              f"reason={decision.reason}")

    print(f"\n=== Stats ===")
    print(json.dumps(guard.get_stats(), indent=2))
