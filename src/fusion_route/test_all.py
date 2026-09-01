#!/usr/bin/env python3
"""Unit tests for fusion_route modules.

Run: python -m pytest src/fusion_route/test_all.py -v
Or:  python src/fusion_route/test_all.py
"""
import json
import os
import sys
import unittest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestSmartRouter(unittest.TestCase):
    """Test SmartRouter rule-based routing."""

    def setUp(self):
        from smart_router import SmartRouter, extract_features, to_flat
        self.router = SmartRouter()
        self.extract = extract_features
        self.to_flat = to_flat

    def test_offline_routes_local_only(self):
        feat = self._make_feat(rtt_ms=5000, online=False, gpu_gb=0)
        d = self.router._route(feat)
        self.assertEqual(d.mode, "local_only")
        self.assertGreater(d.confidence, 0.8)

    def test_high_rtt_routes_cache_hit(self):
        feat = self._make_feat(rtt_ms=1200, online=True, gpu_gb=16)
        d = self.router._route(feat)
        self.assertEqual(d.mode, "cache_hit")

    def test_very_high_rtt_routes_cloud_only(self):
        feat = self._make_feat(rtt_ms=3000, online=True, gpu_gb=0)
        d = self.router._route(feat)
        self.assertEqual(d.mode, "cloud_only")

    def test_low_rtt_high_gpu_routes_cloud_mtp(self):
        feat = self._make_feat(rtt_ms=15, online=True, gpu_gb=24)
        d = self.router._route(feat)
        self.assertEqual(d.mode, "cloud_mtp")
        self.assertTrue(d.use_mtp)

    def test_low_rtt_routes_edge_draft(self):
        feat = self._make_feat(rtt_ms=80, online=True, gpu_gb=4)
        d = self.router._route(feat)
        self.assertEqual(d.mode, "edge_draft")

    def test_extract_features_returns_dict(self):
        user_data = self._make_user_data(rtt_ms=50, online=True)
        feat = self.extract(user_data)
        self.assertIsInstance(feat, dict)
        self.assertIn("rtt_ms", feat)
        self.assertIn("online", feat)

    def test_to_flat_returns_22_dims(self):
        feat = self._make_feat(rtt_ms=50, online=True, gpu_gb=8)
        flat = self.to_flat(feat)
        self.assertEqual(len(flat), 22)
        self.assertTrue(all(isinstance(v, (int, float)) for v in flat))

    def test_route_from_flat(self):
        feat = self._make_feat(rtt_ms=80, online=True, gpu_gb=4)
        flat = self.to_flat(feat)
        d = self.router._route_from_flat(flat)
        self.assertEqual(d.mode, "edge_draft")

    def test_train_noop(self):
        result = self.router.train()
        self.assertEqual(result, 1.0)

    def _make_feat(self, rtt_ms=50, online=True, gpu_gb=8):
        return {
            "rtt_ms": rtt_ms, "bandwidth_mbps": 500,
            "total_mem_gb": 32, "avail_mem_gb": 16,
            "gpu_vram_gb": gpu_gb, "tflops_fp16": 50,
            "unified_memory": False, "draft_model_size_gb": 1.0,
            "draft_params_m": 200, "has_native_mtp": True,
            "prompt_has_code": True, "history_accept_rate": 0.75,
            "cache_hit_rate": 0.6, "online": online,
        }

    def _make_user_data(self, rtt_ms=50, online=True):
        return {
            "four_d_matrix": {
                "D1_network": {"rtt_ms": rtt_ms, "bandwidth_mbps": 500, "jitter_ms": 0, "stability": "stable"},
                "D2_hardware": {"total_mem_gb": 32, "avail_mem_gb": 16, "gpu_vram_gb": 8, "tflops_fp16": 50, "unified_memory": False},
                "D3_model": {"draft_model_size_gb": 1.0, "draft_params_m": 200, "has_native_mtp": True},
                "context": {"prompt_has_code": True, "history_accept_rate": 0.75, "cache_hit_rate": 0.6},
            },
            "request_context": {"online": online, "prompt_has_code": True},
        }


class TestSpeculationGuard(unittest.TestCase):
    """Test SpeculationGuard ROI gating."""

    def setUp(self):
        from speculation_guard import SpeculationGuard
        self.guard = SpeculationGuard(warmup_requests=3)

    def test_warmup_returns_true(self):
        for _ in range(2):
            self.guard.record_request(5, 2, 0.95, 3, 3)
        d = self.guard.should_use_speculation()
        self.assertTrue(d.use_speculation)
        self.assertIn("warmup", d.reason)

    def test_high_accept_keeps_speculation(self):
        for _ in range(10):
            self.guard.record_request(2, 5, 0.95, 3, 3)
        d = self.guard.should_use_speculation()
        self.assertTrue(d.use_speculation)
        self.assertGreater(d.roi, 0)

    def test_low_accept_disables_speculation(self):
        for _ in range(10):
            self.guard.record_request(5, 2, 0.1, 3, 0)
        d = self.guard.should_use_speculation()
        self.assertFalse(d.use_speculation)
        self.assertIn("accept_rate", d.reason)

    def test_stats(self):
        for _ in range(5):
            self.guard.record_request(5, 2, 0.8, 3, 2)
        stats = self.guard.get_stats()
        self.assertEqual(stats["total_requests"], 5)
        self.assertEqual(stats["window_size"], 5)

    def test_reset(self):
        for _ in range(5):
            self.guard.record_request(5, 2, 0.8, 3, 2)
        self.guard.reset()
        stats = self.guard.get_stats()
        self.assertEqual(stats["total_requests"], 0)


class TestPerceptionLayer(unittest.TestCase):
    """Test PerceptionLayer feature extraction."""

    def test_feature_vector_21_dims(self):
        from perception import PerceptionLayer
        layer = PerceptionLayer()
        fv = layer.collect(
            model_name="qwen36_35b",
            model_info={"params_b": 35, "is_moe": True, "num_experts": 256,
                        "num_layers": 40, "model_size_gb": 13.0, "has_native_mtp": True},
        )
        flat = fv.to_flat()
        self.assertEqual(len(flat), 21)
        self.assertTrue(all(isinstance(v, (int, float)) for v in flat))


class TestComplementaryLogit(unittest.TestCase):
    """Test ComplementaryLogit module."""

    def test_output_shape(self):
        try:
            import torch
            from complementary import ComplementaryLogit
            model = ComplementaryLogit(hidden_size=64, vocab_size=100)
            z = torch.randn(1, 10, 100)
            h = torch.randn(1, 10, 64)
            out = model(z, h, alpha=0.1)
            self.assertEqual(out.shape, (1, 10, 100))
        except ImportError:
            self.skipTest("PyTorch not available")


class TestRidgeMapper(unittest.TestCase):
    """Test RidgeKVMapper."""

    def test_translate_shape(self):
        try:
            import torch
            from ridge_mapper import RidgeKVMapper
            mapper = RidgeKVMapper(n_layers=2, n_heads=2, head_dim=64)
            kv = torch.randn(10, 64)  # [seq, head_dim]
            out = mapper.translate(kv, layer=0, head=0)
            self.assertEqual(out.shape, (10, 64))
        except ImportError:
            self.skipTest("PyTorch not available")


class TestExtractPrompts(unittest.TestCase):
    """Test extract_prompts data extraction."""

    def test_extract_from_db(self):
        db_path = os.path.expanduser("~/.freebuff/data/desktop-v2.db")
        if not os.path.exists(db_path):
            self.skipTest("Freebuff DB not found")
        # Just verify the module imports and has the right functions
        from extract_prompts import extract_prompts
        self.assertTrue(callable(extract_prompts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
