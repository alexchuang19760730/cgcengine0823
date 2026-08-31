#!/usr/bin/env bash
# ============================================================
# Benchmark CGC llama.cpp on Windows
# Tests: basic vs expert-cache vs MTP
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$SCRIPT_DIR:$PATH"

MODEL="${1:?Usage: $0 <model.gguf> [ngl]}"
NGL="${2:-4}"
N_PREDICT="${N_PREDICT:-128}"
CTX="${CTX:-2048}"

echo "=========================================="
echo "Windows Benchmark"
echo "=========================================="
echo "  Model:    $MODEL"
echo "  NGL:      $NGL"
echo "  Predict:  $N_PREDICT tokens"
echo "  Context:  $CTX"
echo "=========================================="

# Test 1: Basic (no expert-cache)
echo ""
echo "--- Test 1: Basic (no cache) ---"
CGC_EXPERT_CACHE_BYTES=0 "$SCRIPT_DIR/llama-bench.exe" \
    -m "$MODEL" -ngl "$NGL" -c "$CTX" -n "$N_PREDICT" -p 128 2>&1 | tail -5

# Test 2: Expert-cache 2GB
echo ""
echo "--- Test 2: Expert-cache 2GB ---"
CGC_EXPERT_CACHE_BYTES=2147483648 \
LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1 \
"$SCRIPT_DIR/llama-bench.exe" \
    -m "$MODEL" -ngl "$NGL" -c "$CTX" -n "$N_PREDICT" -p 128 2>&1 | tail -5

# Test 3: Expert-cache 4GB
echo ""
echo "--- Test 3: Expert-cache 4GB ---"
CGC_EXPERT_CACHE_BYTES=4294967296 \
LLAMA_EXPERT_CACHE_ALLOW_NGL=1 \
LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1 \
LLAMA_EXPERT_CACHE_WORKERS=4 \
"$SCRIPT_DIR/llama-bench.exe" \
    -m "$MODEL" -ngl "$NGL" -c "$CTX" -n "$N_PREDICT" -p 128 2>&1 | tail -5

echo ""
echo "=========================================="
echo "Benchmark complete"
echo "=========================================="
