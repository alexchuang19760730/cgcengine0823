"""
Calibration Data Pipeline for KV Translation

Generates calibration sequences and extracts KV caches from two models.
These are used to fit the RidgeKVMapper.

Usage:
  python calibration.py \
    --model_a Qwen/Qwen3.6-35B-A3B \
    --model_b ornith-ai/Ornith-1.5-35B-A3B \
    --n_samples 500 \
    --seq_len 1024 \
    --output_dir calibration_data/
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json


# Calibration prompts - diverse topics for good coverage
CALIBRATION_PROMPTS = [
    "Explain the difference between TCP and UDP protocols.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the key principles of object-oriented programming?",
    "Describe the process of photosynthesis in plants.",
    "How does a neural network learn from data?",
    "What is the time complexity of quicksort?",
    "Explain the concept of garbage collection in Java.",
    "Write a SQL query to find the top 10 customers by revenue.",
    "What are the SOLID principles in software design?",
    "Describe how HTTPS works step by step.",
    "Explain the difference between process and thread.",
    "Write a binary search algorithm in Python.",
    "What is the CAP theorem in distributed systems?",
    "How does DNS resolution work?",
    "Explain the concept of virtual memory.",
    "Write a function to check if a string is a palindrome.",
    "What are the advantages of microservices over monoliths?",
    "Describe the lifecycle of a TCP connection.",
    "How does load balancing work?",
    "Explain the difference between SQL and NoSQL databases.",
    "Write a Python decorator that logs function execution time.",
    "What is the observer design pattern?",
    "How does caching improve system performance?",
    "Explain the concept of consistent hashing.",
    "What are the trade-offs between REST and GraphQL?",
    "Describe how a compiler works.",
    "Write a function to merge two sorted arrays.",
    "What is the difference between optimistic and pessimistic locking?",
    "How does connection pooling work?",
    "Explain the concept of eventual consistency.",
]


def generate_calibration_sequences(
    n_samples: int = 500,
    seq_len: int = 1024,
    output_dir: str = "calibration_data/",
) -> List[str]:
    """
    Generate calibration prompt sequences.
    Returns list of prompt strings (truncated to target length).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    prompts = []
    for i in range(n_samples):
        # Cycle through base prompts and add variations
        base = CALIBRATION_PROMPTS[i % len(CALIBRATION_PROMPTS)]
        variation = f" (variation {i // len(CALIBRATION_PROMPTS) + 1})"
        prompt = base + variation
        prompts.append(prompt)
    
    # Save prompts
    with open(output_path / "prompts.json", "w") as f:
        json.dump(prompts, f)
    
    print(f"Generated {len(prompts)} calibration prompts → {output_path / 'prompts.json'}")
    return prompts


def extract_kv_cache_from_model(
    model_name: str,
    prompts: List[str],
    max_length: int = 1024,
    output_dir: str = "calibration_data/",
) -> str:
    """
    Extract KV caches from a HuggingFace model.
    
    Args:
        model_name: HF model repo name
        prompts: list of prompt strings
        max_length: max sequence length
        output_dir: where to save KV caches
    
    Returns:
        path to saved KV cache file
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    all_kvs = []
    
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Forward pass with output_hidden_states and past_key_values
            outputs = model(**inputs, output_hidden_states=False, use_cache=True)
            
            # Extract KV cache: tuple of (key, value) per layer
            past_kv = outputs.past_key_values
            if past_kv is not None:
                # Stack into numpy: [num_layers, 2, seq_len, num_heads, head_dim]
                # We only need the keys (or values, both work for ridge regression)
                kv_array = []
                for layer_idx in range(len(past_kv)):
                    key = past_kv[layer_idx][0].cpu().float().numpy()   # [batch, heads, seq, dim]
                    value = past_kv[layer_idx][1].cpu().float().numpy()
                    kv_array.append(np.stack([key[0], value[0]], axis=0))  # [2, heads, seq, dim]
                
                all_kvs.append(np.array(kv_array))  # [layers, 2, heads, seq, dim]
            
            if (i + 1) % 50 == 0:
                print(f"  Extracted KV from {i + 1}/{len(prompts)} prompts")
    
    # Save as numpy
    save_path = output_path / f"kv_cache_{model_name.split('/')[-1]}.npz"
    np.savez_compressed(str(save_path), *all_kvs)
    print(f"Saved KV caches → {save_path} ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    del model
    torch.cuda.empty_cache()
    
    return str(save_path)


def run_calibration_pipeline(
    model_a: str,
    model_b: str,
    n_samples: int = 500,
    seq_len: int = 1024,
    output_dir: str = "calibration_data/",
) -> Tuple[str, str]:
    """
    Full calibration pipeline: generate prompts → extract KV from both models.
    
    Returns:
        (kv_path_a, kv_path_b)
    """
    # Step 1: Generate prompts
    prompts = generate_calibration_sequences(n_samples, seq_len, output_dir)
    
    # Step 2: Extract KV caches from both models
    kv_path_a = extract_kv_cache_from_model(model_a, prompts, seq_len, output_dir)
    kv_path_b = extract_kv_cache_from_model(model_b, prompts, seq_len, output_dir)
    
    return kv_path_a, kv_path_b


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KV Translation Calibration Pipeline")
    parser.add_argument("--model_a", default="Qwen/Qwen3.6-35B-A3B", help="Source model")
    parser.add_argument("--model_b", default="ornith-ai/Ornith-1.5-35B-A3B", help="Target model")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--output_dir", default="calibration_data/")
    args = parser.parse_args()
    
    run_calibration_pipeline(args.model_a, args.model_b, args.n_samples, args.seq_len, args.output_dir)
