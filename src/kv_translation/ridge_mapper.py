"""
KV Translation via Ridge Regression

Based on: "KV Cache Transfer" (NVIDIA, 2026)
Paper shows: same-family models have linearly-structured KV caches
that can be mapped via closed-form per-head ridge regression.

Core formula: W_ridge = (X^T X + λI)^{-1} X^T Y
  X = source model KV cache [seq_len, head_dim]
  Y = target model KV cache [seq_len, head_dim]
  W = mapping matrix [head_dim, head_dim]

Usage:
  mapper = RidgeKVMapper(num_layers=40, num_heads=2, head_dim=256)
  mapper.fit(source_kvs, target_kvs, lambda_reg=1.0)
  mapped_kv = mapper.translate(source_kv)
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import json


class RidgeKVMapper:
    """
    Per-layer, per-head ridge regression KV cache mapper.
    
    KV cache shape: [num_layers, num_heads, seq_len, head_dim]
    Mapping matrix: [num_layers, num_heads, head_dim, head_dim]
    """
    
    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W = None  # [num_layers, num_heads, head_dim, head_dim]
        self.fitted = False
        self.calibration_stats = {}
    
    def fit(
        self,
        source_kvs: List[np.ndarray],
        target_kvs: List[np.ndarray],
        lambda_reg: float = 1.0,
    ) -> dict:
        """
        Fit ridge regression mappings from calibration data.
        
        Args:
            source_kvs: list of source KV caches, each [num_layers, num_heads, seq_len, head_dim]
            target_kvs: list of target KV caches, same shape
            lambda_reg: regularization strength
        
        Returns:
            stats dict with per-layer cosine similarities
        """
        assert len(source_kvs) == len(target_kvs), "source and target must have same number of samples"
        assert len(source_kvs) > 0, "need at least one calibration sample"
        
        n_samples = len(source_kvs)
        W = np.zeros((self.num_layers, self.num_heads, self.head_dim, self.head_dim))
        cos_sims = np.zeros((self.num_layers, self.num_heads))
        
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                # Collect all samples for this layer+head
                # X: [n_samples * seq_len, head_dim]
                # Y: [n_samples * seq_len, head_dim]
                X_list = []
                Y_list = []
                for src, tgt in zip(source_kvs, target_kvs):
                    X_list.append(src[layer, head])  # [seq_len, head_dim]
                    Y_list.append(tgt[layer, head])
                
                X = np.concatenate(X_list, axis=0)  # [total_tokens, head_dim]
                Y = np.concatenate(Y_list, axis=0)
                
                # Ridge regression: W = (X^T X + λI)^{-1} X^T Y
                XtX = X.T @ X  # [head_dim, head_dim]
                XtY = X.T @ Y  # [head_dim, head_dim]
                I = np.eye(self.head_dim) * lambda_reg
                W[layer, head] = np.linalg.solve(XtX + I, XtY)
                
                # Compute cosine similarity for quality check
                Y_pred = X @ W[layer, head]
                cos_sim = self._cosine_similarity(Y, Y_pred)
                cos_sims[layer, head] = cos_sim
        
        self.W = W
        self.fitted = True
        
        # Stats
        self.calibration_stats = {
            "n_samples": n_samples,
            "lambda_reg": lambda_reg,
            "mean_cosine_similarity": float(np.mean(cos_sims)),
            "min_cosine_similarity": float(np.min(cos_sims)),
            "max_cosine_similarity": float(np.max(cos_sims)),
            "per_layer_mean": [float(np.mean(cos_sims[l])) for l in range(self.num_layers)],
        }
        
        return self.calibration_stats
    
    def translate(self, source_kv: np.ndarray) -> np.ndarray:
        """
        Translate a source KV cache to target model's KV space.
        
        Args:
            source_kv: [num_layers, num_heads, seq_len, head_dim]
        
        Returns:
            target_kv: [num_layers, num_heads, seq_len, head_dim]
        """
        assert self.fitted, "Must call fit() first"
        assert source_kv.shape[0] == self.num_layers and source_kv.shape[1] == self.num_heads and source_kv.shape[3] == self.head_dim, \
            f"Expected shape [{self.num_layers}, {self.num_heads}, *, {self.head_dim}], got {source_kv.shape}"
        
        seq_len = source_kv.shape[2]
        target_kv = np.zeros_like(source_kv)
        
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                # X: [seq_len, head_dim]
                X = source_kv[layer, head]
                # Y_pred: [seq_len, head_dim] = X @ W[layer, head]
                target_kv[layer, head] = X @ self.W[layer, head]
        
        return target_kv
    
    def translate_single_token(
        self, source_kv: np.ndarray, new_token_kv: np.ndarray
    ) -> np.ndarray:
        """
        Translate a single new token's KV using the fitted mapping.
        More efficient than translating the full cache for incremental decode.
        
        Args:
            source_kv: [num_layers, num_heads, 1, head_dim] (single token)
        
        Returns:
            target_kv: [num_layers, num_heads, 1, head_dim]
        """
        assert self.fitted, "Must call fit() first"
        target_kv = np.zeros_like(source_kv)
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                target_kv[layer, head] = source_kv[layer, head] @ self.W[layer, head]
        return target_kv
    
    def save(self, path: str):
        """Save mapping matrix to file."""
        assert self.fitted, "Must call fit() first"
        data = {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "W": self.W.tolist(),
            "stats": self.calibration_stats,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Saved KV mapper to {path} ({Path(path).stat().st_size / 1024 / 1024:.1f} MB)")
    
    @classmethod
    def load(cls, path: str) -> "RidgeKVMapper":
        """Load mapping matrix from file."""
        with open(path) as f:
            data = json.load(f)
        mapper = cls(data["num_layers"], data["num_heads"], data["head_dim"])
        mapper.W = np.array(data["W"])
        mapper.fitted = True
        mapper.calibration_stats = data.get("stats", {})
        print(f"Loaded KV mapper: {mapper.num_layers} layers, {mapper.num_heads} heads, "
              f"cos_sim={mapper.calibration_stats.get('mean_cosine_similarity', 'N/A')}")
        return mapper
    
    @staticmethod
    def _cosine_similarity(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """Compute mean cosine similarity across all tokens and dims."""
        # Flatten to [tokens, dim]
        Y_t = Y_true.reshape(-1, Y_true.shape[-1])
        Y_p = Y_pred.reshape(-1, Y_pred.shape[-1])
        
        # Cosine similarity per token, then average
        dot = np.sum(Y_t * Y_p, axis=-1)
        norm_t = np.linalg.norm(Y_t, axis=-1) + 1e-8
        norm_p = np.linalg.norm(Y_p, axis=-1) + 1e-8
        cos = dot / (norm_t * norm_p)
        return float(np.mean(cos))


def demo_fit_and_translate():
    """Demo: fit on random data, verify translation works."""
    num_layers, num_heads, head_dim = 40, 2, 256
    seq_len = 512
    n_samples = 10
    
    # Create mapper
    mapper = RidgeKVMapper(num_layers, num_heads, head_dim)
    
    # Generate fake calibration data (simulating two related models)
    np.random.seed(42)
    source_kvs = []
    target_kvs = []
    for _ in range(n_samples):
        src = np.random.randn(num_layers, num_heads, seq_len, head_dim).astype(np.float32)
        # Target is source + small perturbation (simulating same-family models)
        tgt = src + 0.1 * np.random.randn(num_layers, num_heads, seq_len, head_dim).astype(np.float32)
        source_kvs.append(src)
        target_kvs.append(tgt)
    
    # Fit
    stats = mapper.fit(source_kvs, target_kvs, lambda_reg=1.0)
    print(f"Calibration stats:")
    print(f"  Mean cosine similarity: {stats['mean_cosine_similarity']:.4f}")
    print(f"  Min cosine similarity: {stats['min_cosine_similarity']:.4f}")
    print(f"  Max cosine similarity: {stats['max_cosine_similarity']:.4f}")
    
    # Test translation
    test_src = np.random.randn(num_layers, num_heads, seq_len, head_dim).astype(np.float32)
    test_tgt = mapper.translate(test_src)
    print(f"\nTranslation:")
    print(f"  Input shape: {test_src.shape}")
    print(f"  Output shape: {test_tgt.shape}")
    print(f"  Max abs diff: {np.max(np.abs(test_tgt - test_src)):.4f}")
    
    # Save/load test
    mapper.save("/tmp/kv_mapper_test.json")
    loaded = RidgeKVMapper.load("/tmp/kv_mapper_test.json")
    test_tgt2 = loaded.translate(test_src)
    assert np.allclose(test_tgt, test_tgt2), "Save/load mismatch!"
    print(f"  Save/load verified: OK")


if __name__ == "__main__":
    demo_fit_and_translate()
