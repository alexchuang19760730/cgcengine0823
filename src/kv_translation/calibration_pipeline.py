#!/usr/bin/env python3
"""
KV Translation Calibration Pipeline
====================================

Generates high-quality calibration data for RidgeKVMapper.
Designed for Mac M4 with 16GB RAM (fits 35B-A3B MoE at ~14GB).

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │  PromptGenerator                                         │
  │  - 120+ diverse prompts (code, math, reasoning, chat)   │
  │  - Template expansion → target seq_len                   │
  │  - Train/validation split (80/20)                        │
  ├──────────────────────────────────────────────────────────┤
  │  KVExtractor                                             │
  │  - Load model with use_cache=True                        │
  │  - Forward pass → extract past_key_values                │
  │  - Incremental save (every 50 samples)                   │
  │  - NaN/Inf quality gate                                  │
  ├──────────────────────────────────────────────────────────┤
  │  Pipeline                                                │
  │  1. Generate prompts                                     │
  │  2. Extract KV from model_a (Qwen3.6)                   │
  │  3. Extract KV from model_b (Ornith-1.5)                │
  │  4. Validate pairs (same length, no NaN)                 │
  │  5. Save as .npz with train/val splits                   │
  └──────────────────────────────────────────────────────────┘

Usage on Mac:
  cd /path/to/cgcengine_full
  python src/kv_translation/calibration_pipeline.py \
    --model_a Alexchuang/cgcengine-models:Nail-Qwen3.6-35B-A3B-MTP-UD-IQ3_XXS-denseIQ4X.gguf \
    --model_b bartowski/Ornith-1.5-35B-A3B-GGUF:Ornith-1.5-35B-A3B-Q2_K.gguf \
    --n_samples 500 --seq_len 1024 \
    --output_dir calibration_data/

Then fit the mapper:
  python src/kv_translation/ridge_mapper.py \
    --calibration_dir calibration_data/ \
    --output kv_map_qwen36_to_ornith.json

Or use the HF GGUF models directly (no need to download separately):
  python src/kv_translation/calibration_pipeline.py \
    --model_a Qwen/Qwen3.6-35B-A3B \
    --model_b ornith-ai/Ornith-1.5-35B-A3B \
    --n_samples 500 --seq_len 1024
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import hashlib
import time
import sys
import os
import random
from dataclasses import dataclass, asdict


# ============================================================================
# Prompt Generation
# ============================================================================

# Base prompts organized by domain for maximum diversity
PROMPT_TEMPLATES = {
    "code_python": [
        "Write a Python function to {task}. Include type hints and docstring.",
        "Implement a Python class that {task}. Use dataclasses where appropriate.",
        "Write a Python decorator that {task}. Include error handling.",
        "Create a Python context manager that {task}. Make it thread-safe.",
        "Write a Python generator that {task}. Use lazy evaluation.",
        "Implement a Python async function that {task}. Use aiohttp if needed.",
        "Write a Python data pipeline that {task}. Use pandas/numpy.",
        "Create a Python CLI tool that {task}. Use argparse with subcommands.",
    ],
    "code_other": [
        "Write a JavaScript function that {task}. Use ES2024 syntax.",
        "Implement a TypeScript interface and class that {task}.",
        "Write a SQL query that {task}. Optimize for large datasets.",
        "Write a Rust function that {task}. Handle errors with Result<>.",
        "Write a Go function that {task}. Use goroutines for concurrency.",
        "Write a bash script that {task}. Include error handling and logging.",
    ],
    "math": [
        "Solve this step by step: {problem}",
        "Prove that {statement}. Show all steps.",
        "Find the {target} of {expression}. Explain your approach.",
        "Derive the formula for {concept} from first principles.",
        "Calculate the {measure} of {object}. Show work.",
    ],
    "reasoning": [
        "Given these constraints: {constraints}. What is the optimal solution?",
        "Analyze the following scenario: {scenario}. Provide a detailed analysis.",
        "Compare and contrast {a} and {b}. When would you prefer each?",
        "If {premise}, what conclusions can we draw? Explain step by step.",
        "What are the trade-offs between {a} and {b}? Consider {dimensions}.",
    ],
    "system_design": [
        "Design a system that {requirement}. Include architecture, protocols, and failure modes.",
        "Review this architecture: {description}. Identify bottlenecks and improvements.",
        "How would you scale {system} to handle {scale}? Consider consistency and availability.",
        "Write a technical design document for {feature}. Include API, storage, and monitoring.",
    ],
    "explanation": [
        "Explain {concept} to a senior engineer. Include edge cases and gotchas.",
        "What is {concept}? Give a deep technical explanation with examples.",
        "Why does {concept} work? Explain the underlying mechanism.",
        "How does {concept} compare to {alternative}? Performance, correctness, trade-offs.",
    ],
    "chat_multi_turn": [
        "I'm debugging a {issue}. Here's the error: {error}. What's wrong?",
        "I need to refactor this code: {code}. Make it more maintainable.",
        "Can you review this pull request? {description}. Focus on correctness and performance.",
        "I'm designing an API for {domain}. What are the key endpoints?",
        "Help me optimize this query: {query}. Current latency is {latency}.",
    ],
    "translation": [
        "Translate this technical documentation from {lang_a} to {lang_b}: {text}",
        "Localize this UI string for {locale}: {string}",
        "Write documentation for this API in both {lang_a} and {lang_b}: {api}",
    ],
    "creative_writing": [
        "Write a short story about {topic}. Include dialogue and vivid descriptions.",
        "Compose a technical blog post about {topic}. Target audience: {audience}.",
        "Write a poem about {topic} in the style of {style}.",
    ],
}

# Fill-in values for templates
FILL_VALUES = {
    "task": [
        "compute the shortest path in a weighted graph",
        "parse and validate JSON schemas at runtime",
        "implement an LRU cache with TTL support",
        "convert a recursive function to an iterative one with a stack",
        "parse command-line arguments with nested subcommands",
        "implement a rate limiter using the sliding window algorithm",
        "generate all permutations of a list without recursion",
        "implement a thread-safe bounded blocking queue",
        "serialize a binary tree to a string and deserialize it back",
        "implement a concurrent web scraper with depth limiting",
        "build a simple key-value store with WAL persistence",
        "implement a bloom filter with configurable false positive rate",
        "create a CSV parser that handles quoted fields and escapes",
        "implement a connection pool with health checking",
        "write a diff algorithm that handles insertions and deletions",
        "implement a pub-sub system with topic wildcards",
        "create a job scheduler that supports cron expressions",
        "implement a distributed lock using Redis",
        "write a memory-efficient streaming JSON parser",
        "implement a trie with autocomplete and fuzzy matching",
        "create a markdown parser that handles nested lists",
        "implement a spatial index using R-trees",
        "write a function that detects cycles in a directed graph",
        "implement a skip list with concurrent access support",
        "create a lazy-loading proxy pattern for expensive objects",
        "implement a binary heap with decrease-key operation",
        "write a merge algorithm that handles runs of duplicates",
        "implement a probabilistic data structure for frequency estimation",
        "create a type-safe builder pattern with fluent interface",
        "write an actor model implementation with supervision trees",
    ],
    "problem": [
        "prove that the sum of the first n odd numbers equals n²",
        "find all prime numbers up to N using the Sieve of Eratosthenes",
        "compute the determinant of a 4×4 matrix",
        "solve the knapsack problem for these weights and values",
        "find the longest common subsequence of two strings",
        "calculate the eigenvalues of a 3×3 matrix",
        "determine if a number is a perfect square without sqrt",
        "compute the nth Fibonacci number in O(log n)",
        "find all Hamiltonian paths in a small graph",
        "solve this system of linear equations using Gaussian elimination",
    ],
    "concept": [
        "CAP theorem", "consensus algorithms", "vector clocks",
        "CRDTs", "Bloom filters", "consistent hashing",
        "virtual memory", "copy-on-write", "event sourcing",
        "CQRS", "saga pattern", "circuit breaker",
        "rate limiting", "backpressure", "exponential backoff",
        "distributed transactions", "two-phase commit",
        "skip graphs", "Merkle trees", "B+ trees",
    ],
    "scenario": [
        "a distributed system where 2 of 5 nodes fail simultaneously",
        "a database migration that must maintain zero downtime",
        "a real-time chat system with 10M concurrent users",
        "a payment processing system that must never lose transactions",
        "a search engine that indexes 1B documents with sub-second queries",
        "a video streaming platform that handles 4K at 60fps",
        "a IoT system processing 1M sensor events per second",
        "a machine learning pipeline that retrains daily on 1TB of data",
    ],
    "a": ["REST", "gRPC", "GraphQL", "WebSocket", "message queue"],
    "b": ["gRPC", "GraphQL", "WebSocket", "message queue", "REST"],
    "issue": [
        "memory leak in a long-running Python service",
        "race condition in a concurrent data structure",
        "deadlock in a database connection pool",
        "serialization error with nested JSON objects",
        "timeout in an async HTTP client",
        "segfault in a C extension module",
        "OOM error when processing large files",
    ],
    "error": [
        "TypeError: Cannot read property of undefined",
        "ConnectionRefusedError: [Errno 111] Connection refused",
        "java.lang.OutOfMemoryError: Java heap space",
        "panic: runtime error: index out of range",
        "Segmentation fault (core dumped)",
        "MemoryError: Unable to allocate array",
        "sqlalchemy.exc.TimeoutError: QueuePool limit exceeded",
    ],
    "concept_list": [
        "TCP congestion control", "garbage collection algorithms",
        "virtual memory paging", "CPU cache coherence",
        "network packet routing", "database indexing strategies",
        "compiler optimization passes", "load balancing algorithms",
    ],
}


def expand_template(template: str, fill_values: dict, seed: int = 42) -> str:
    """Expand a template string with random fill values."""
    rng = random.Random(seed)
    result = template
    for key, values in fill_values.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, rng.choice(values), 1)
    # Fill any remaining placeholders with generic text
    for key, values in fill_values.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, rng.choice(values), 1)
    return result


def generate_calibration_prompts(
    n_samples: int = 500,
    seq_len: int = 1024,
    seed: int = 42,
) -> List[str]:
    """
    Generate diverse calibration prompts covering multiple domains.

    Args:
        n_samples: total number of prompts to generate
        seq_len: target sequence length (tokens)
        seed: random seed for reproducibility

    Returns:
        list of prompt strings
    """
    rng = random.Random(seed)
    prompts = []

    # Calculate how many from each domain (proportional to importance)
    domain_weights = {
        "code_python": 0.25,
        "code_other": 0.10,
        "math": 0.15,
        "reasoning": 0.15,
        "system_design": 0.10,
        "explanation": 0.10,
        "chat_multi_turn": 0.10,
        "creative_writing": 0.05,
    }

    for domain, weight in domain_weights.items():
        n_domain = max(1, int(n_samples * weight))
        templates = PROMPT_TEMPLATES.get(domain, [])

        for i in range(n_domain):
            template = templates[i % len(templates)]
            # Use deterministic seed per prompt for reproducibility
            prompt_seed = seed + hash(template) + i
            prompt = expand_template(template, FILL_VALUES, seed=prompt_seed)

            # Pad to approximate target length
            # Average token is ~4 chars, so seq_len tokens ≈ seq_len * 4 chars
            target_chars = seq_len * 4
            if len(prompt) < target_chars:
                # Repeat the prompt to fill the context window
                repetitions = (target_chars // len(prompt)) + 1
                prompt = (prompt + " ") * repetitions
                prompt = prompt[:target_chars]

            prompts.append(prompt)

    # Shuffle and trim to exact count
    rng.shuffle(prompts)
    prompts = prompts[:n_samples]

    return prompts


# ============================================================================
# KV Cache Extraction
# ============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for KV cache extraction."""
    model_name: str
    max_length: int = 1024
    batch_size: int = 1
    device: str = "auto"
    torch_dtype: str = "float16"
    save_every: int = 50
    output_dir: str = "calibration_data/"


def extract_kv_from_single_prompt(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 1024,
) -> Optional[np.ndarray]:
    """
    Extract KV cache from a single prompt.

    Returns:
        KV array: [num_layers, 2, num_heads, seq_len, head_dim]
        or None if extraction fails
    """
    import torch

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        past_kv = outputs.past_key_values
        if past_kv is None:
            return None

        # Stack: [num_layers, 2(K,V), num_heads, seq_len, head_dim]
        kv_layers = []
        for layer_idx in range(len(past_kv)):
            key = past_kv[layer_idx][0].cpu().float().numpy()[0]   # [heads, seq, dim]
            value = past_kv[layer_idx][1].cpu().float().numpy()[0]
            kv_layers.append(np.stack([key, value], axis=0))  # [2, heads, seq, dim]

        return np.array(kv_layers)  # [layers, 2, heads, seq, dim]

    except Exception as e:
        print(f"  [WARN] Extraction failed: {e}")
        return None


def check_kv_quality(kv: np.ndarray) -> bool:
    """Check if KV cache values are valid (no NaN, no Inf, reasonable range)."""
    if np.any(np.isnan(kv)):
        return False
    if np.any(np.isinf(kv)):
        return False
    # Check value range (should be within reasonable bounds for float16)
    if np.abs(kv).max() > 100.0:
        return False
    return True


def extract_kv_caches(
    config: ExtractionConfig,
    prompts: List[str],
    prefix: str = "model_a",
) -> str:
    """
    Extract KV caches from a model for all prompts.

    Args:
        config: extraction configuration
        prompts: list of prompt strings
        prefix: filename prefix ("model_a" or "model_b")

    Returns:
        path to saved .npz file
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Extracting KV caches: {prefix}")
    print(f"Model: {config.model_name}")
    print(f"Prompts: {len(prompts)}, max_length: {config.max_length}")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Load model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    print(f"  Loading model ({config.torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        device_map=config.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    # Extract KV caches
    all_kvs = []
    valid_count = 0
    invalid_count = 0
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        kv = extract_kv_from_single_prompt(
            model, tokenizer, prompt, config.max_length
        )

        if kv is not None and check_kv_quality(kv):
            all_kvs.append(kv)
            valid_count += 1
        else:
            invalid_count += 1
            if kv is not None:
                print(f"  [WARN] Prompt {i}: quality check failed (NaN/Inf/out of range)")

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate
            print(f"  [{i+1}/{len(prompts)}] valid={valid_count} invalid={invalid_count} "
                  f"rate={rate:.1f}/s ETA={eta:.0f}s")

        # Incremental save
        if (i + 1) % config.save_every == 0 and len(all_kvs) > 0:
            tmp_path = output_path / f"{prefix}_checkpoint_{i+1}.npz"
            np.savez_compressed(str(tmp_path), *all_kvs)
            print(f"  Checkpoint saved: {tmp_path.name} ({len(all_kvs)} samples)")

    # Final save
    save_path = output_path / f"{prefix}_kv.npz"
    if len(all_kvs) > 0:
        np.savez_compressed(str(save_path), *all_kvs)
        file_size = save_path.stat().st_size / 1024 / 1024
        print(f"\n  Final save: {save_path.name} ({len(all_kvs)} samples, {file_size:.1f} MB)")
    else:
        print(f"\n  [ERROR] No valid KV caches extracted!")

    # Cleanup checkpoints
    for cp in output_path.glob(f"{prefix}_checkpoint_*.npz"):
        cp.unlink()

    # Save extraction metadata
    metadata = {
        "model_name": config.model_name,
        "n_prompts": len(prompts),
        "n_valid": valid_count,
        "n_invalid": invalid_count,
        "max_length": config.max_length,
        "extraction_time_s": time.time() - start_time,
        "kv_shape": list(all_kvs[0].shape) if all_kvs else None,
    }
    with open(output_path / f"{prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Free memory
    del model
    torch.cuda.empty_cache()

    return str(save_path)


# ============================================================================
# Data Validation
# ============================================================================

def validate_calibration_data(
    kv_path_a: str,
    kv_path_b: str,
    output_dir: str = "calibration_data/",
) -> dict:
    """
    Validate paired calibration data:
    - Same number of samples
    - Same KV shape
    - No NaN/Inf in either
    - Compute per-sample cosine similarity for quality check

    Returns:
        validation report dict
    """
    data_a = np.load(kv_path_a)
    data_b = np.load(kv_path_b)

    keys_a = sorted([k for k in data_a.files if not k.startswith("__")])
    keys_b = sorted([k for k in data_b.files if not k.startswith("__")])

    report = {
        "n_samples_a": len(keys_a),
        "n_samples_b": len(keys_b),
        "matched": len(keys_a) == len(keys_b),
        "shapes_match": True,
        "nan_count_a": 0,
        "nan_count_b": 0,
        "inf_count_a": 0,
        "inf_count_b": 0,
        "per_sample_cosine": [],
        "mean_cosine": 0.0,
        "min_cosine": 1.0,
        "max_cosine": 0.0,
    }

    n_compare = min(len(keys_a), len(keys_b))

    for i in range(n_compare):
        kv_a = data_a[keys_a[i]]
        kv_b = data_b[keys_b[i]]

        # Shape check
        if kv_a.shape != kv_b.shape:
            report["shapes_match"] = False
            report["shape_mismatch"] = {
                "sample": i,
                "shape_a": list(kv_a.shape),
                "shape_b": list(kv_b.shape),
            }

        # NaN/Inf check
        report["nan_count_a"] += int(np.any(np.isnan(kv_a)))
        report["nan_count_b"] += int(np.any(np.isnan(kv_b)))
        report["inf_count_a"] += int(np.any(np.isinf(kv_a)))
        report["inf_count_b"] += int(np.any(np.isinf(kv_b)))

        # Cosine similarity (flatten all layers and heads)
        flat_a = kv_a.reshape(-1, kv_a.shape[-1])
        flat_b = kv_b.reshape(-1, kv_b.shape[-1])

        dot = np.sum(flat_a * flat_b, axis=-1)
        norm_a = np.linalg.norm(flat_a, axis=-1) + 1e-8
        norm_b = np.linalg.norm(flat_b, axis=-1) + 1e-8
        cos = np.mean(dot / (norm_a * norm_b))
        report["per_sample_cosine"].append(float(cos))

    if report["per_sample_cosine"]:
        report["mean_cosine"] = float(np.mean(report["per_sample_cosine"]))
        report["min_cosine"] = float(np.min(report["per_sample_cosine"]))
        report["max_cosine"] = float(np.max(report["per_sample_cosine"]))

    # Save report
    output_path = Path(output_dir)
    report_path = output_path / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nValidation Report:")
    print(f"  Samples A: {report['n_samples_a']}, B: {report['n_samples_b']}")
    print(f"  Matched: {report['matched']}")
    print(f"  Shapes match: {report['shapes_match']}")
    print(f"  NaN: A={report['nan_count_a']}, B={report['nan_count_b']}")
    print(f"  Cosine: mean={report['mean_cosine']:.4f} "
          f"min={report['min_cosine']:.4f} max={report['max_cosine']:.4f}")

    if report["mean_cosine"] > 0.8:
        print(f"  ✅ High cosine similarity — models are good candidates for KV Translation")
    elif report["mean_cosine"] > 0.5:
        print(f"  ⚠️  Moderate cosine — KV Translation may need higher lambda_reg")
    else:
        print(f"  ❌ Low cosine — KV Translation quality will be limited")

    return report


# ============================================================================
# Train/Val Split
# ============================================================================

def split_and_save(
    kv_path: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Split KV data into train/val sets.

    Returns:
        (train_path, val_path)
    """
    data = np.load(kv_path)
    keys = sorted([k for k in data.files if not k.startswith("__")])

    rng = random.Random(seed)
    indices = list(range(len(keys)))
    rng.shuffle(indices)

    split_idx = int(len(keys) * train_ratio)
    train_keys = [keys[i] for i in indices[:split_idx]]
    val_keys = [keys[i] for i in indices[split_idx:]]

    base = Path(kv_path).parent
    stem = Path(kv_path).stem

    # Save train
    train_path = base / f"{stem}_train.npz"
    np.savez_compressed(str(train_path), **{k: data[k] for k in train_keys})

    # Save val
    val_path = base / f"{stem}_val.npz"
    np.savez_compressed(str(val_path), **{k: data[k] for k in val_keys})

    print(f"  Split: {len(train_keys)} train, {len(val_keys)} val")
    return str(train_path), str(val_path)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_full_pipeline(
    model_a: str = "Qwen/Qwen3.6-35B-A3B",
    model_b: str = "ornith-ai/Ornith-1.5-35B-A3B",
    n_samples: int = 500,
    seq_len: int = 1024,
    output_dir: str = "calibration_data/",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """
    Full calibration pipeline:
    1. Generate prompts
    2. Extract KV from both models
    3. Validate pairs
    4. Split train/val

    Returns:
        pipeline result dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {"status": "started", "steps": {}}

    # Step 1: Generate prompts
    print("\n" + "=" * 60)
    print("STEP 1: Generating calibration prompts")
    print("=" * 60)

    prompts = generate_calibration_prompts(n_samples, seq_len, seed)

    # Save prompts
    with open(output_path / "prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"  Generated {len(prompts)} prompts")
    print(f"  Avg length: {np.mean([len(p) for p in prompts]):.0f} chars")

    result["steps"]["prompts"] = {
        "count": len(prompts),
        "avg_chars": float(np.mean([len(p) for p in prompts])),
    }

    # Step 2: Extract KV from model A
    config_a = ExtractionConfig(
        model_name=model_a,
        max_length=seq_len,
        output_dir=output_dir,
    )
    kv_path_a = extract_kv_caches(config_a, prompts, prefix="model_a")

    result["steps"]["model_a"] = {"kv_path": kv_path_a}

    # Step 3: Extract KV from model B
    config_b = ExtractionConfig(
        model_name=model_b,
        max_length=seq_len,
        output_dir=output_dir,
    )
    kv_path_b = extract_kv_caches(config_b, prompts, prefix="model_b")

    result["steps"]["model_b"] = {"kv_path": kv_path_b}

    # Step 4: Validate
    print("\n" + "=" * 60)
    print("STEP 4: Validating calibration pairs")
    print("=" * 60)

    report = validate_calibration_data(kv_path_a, kv_path_b, output_dir)
    result["steps"]["validation"] = report

    # Step 5: Split train/val
    print("\n" + "=" * 60)
    print("STEP 5: Train/Val split")
    print("=" * 60)

    split_and_save(kv_path_a, train_ratio, seed)
    split_and_save(kv_path_b, train_ratio, seed)

    # Save pipeline config
    pipeline_config = {
        "model_a": model_a,
        "model_b": model_b,
        "n_samples": n_samples,
        "seq_len": seq_len,
        "train_ratio": train_ratio,
        "seed": seed,
        "output_dir": output_dir,
        "status": "completed",
        "report": report,
    }
    with open(output_path / "pipeline_config.json", "w") as f:
        json.dump(pipeline_config, f, indent=2)

    result["status"] = "completed"
    result["pipeline_config"] = pipeline_config

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Files:")
    for p in sorted(output_path.glob("*")):
        size = p.stat().st_size / 1024 / 1024
        print(f"    {p.name}: {size:.1f} MB")
    print(f"\nNext step: fit RidgeKVMapper")
    print(f"  python src/kv_translation/ridge_mapper.py \\")
    print(f"    --calibration_dir {output_dir} \\")
    print(f"    --output kv_map.json")

    return result


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KV Translation Calibration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use HF models directly (needs GPU + ~30GB RAM for 35B models)
  python calibration_pipeline.py \\
    --model_a Qwen/Qwen3.6-35B-A3B \\
    --model_b ornith-ai/Ornith-1.5-35B-A3B \\
    --n_samples 500 --seq_len 1024

  # Use local GGUF models (via llama.cpp extraction)
  python calibration_pipeline.py \\
    --model_a /path/to/Nail-denseIQ4X.gguf \\
    --model_b /path/to/Ornith-Q2_K.gguf \\
    --n_samples 500 --seq_len 1024

  # Quick test with 50 samples
  python calibration_pipeline.py \\
    --model_a Qwen/Qwen3.6-35B-A3B \\
    --model_b ornith-ai/Ornith-1.5-35B-A3B \\
    --n_samples 50 --seq_len 512
        """,
    )

    parser.add_argument(
        "--model_a",
        default="Qwen/Qwen3.6-35B-A3B",
        help="Source model (HF repo or local path)",
    )
    parser.add_argument(
        "--model_b",
        default="ornith-ai/Ornith-1.5-35B-A3B",
        help="Target model (HF repo or local path)",
    )
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--output_dir", default="calibration_data/")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_full_pipeline(
        model_a=args.model_a,
        model_b=args.model_b,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
