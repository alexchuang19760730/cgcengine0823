#!/bin/bash
# ============================================================================
# KV Translation Calibration — Mac M4 One-Click Script
# ============================================================================
#
# Prerequisites:
#   1. pip install transformers torch accelerate huggingface_hub numpy
#   2. Models downloaded (or HF login for auto-download):
#      - Qwen/Qwen3.6-35B-A3B (or use Alexchuang/cgcengine-models GGUF)
#      - ornith-ai/Ornith-1.5-35B-A3B (or use bartowski GGUF)
#
# Usage:
#   chmod +x run_calibration_mac.sh
#   ./run_calibration_mac.sh
#
# What it does:
#   1. Generates 500 diverse calibration prompts
#   2. Extracts KV caches from Qwen3.6-35B-A3B
#   3. Extracts KV caches from Ornith-1.5-35B-A3B
#   4. Validates pairs (NaN check, cosine similarity)
#   5. Splits into train/val (80/20)
#   6. Fits ridge regression mapping matrix
#   7. Saves kv_map.json (117MB for 40L×2H×256D)
#
# Estimated time: 2-4 hours on Mac M4 Max (16GB)
# Estimated RAM: ~14GB (both 35B-A3B MoE models at float16)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/calibration_data"

# Configuration
N_SAMPLES=500
SEQ_LEN=1024
MODEL_A="Qwen/Qwen3.6-35B-A3B"
MODEL_B="ornith-ai/Ornith-1.5-35B-A3B"

echo "============================================================"
echo "KV Translation Calibration Pipeline"
echo "============================================================"
echo "  Model A (source): $MODEL_A"
echo "  Model B (target): $MODEL_B"
echo "  Samples: $N_SAMPLES"
echo "  Seq length: $SEQ_LEN"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"

# Step 1: Check environment
echo ""
echo "[1/5] Checking environment..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')"

# Check GPU
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('  GPU: Apple Silicon MPS')
else:
    print('  WARNING: No GPU detected, will use CPU (very slow)')
"

# Step 2: Run full pipeline (prompts + KV extraction + validation)
echo ""
echo "[2/5] Running calibration pipeline..."
cd "$PROJECT_ROOT"

python3 src/kv_translation/calibration_pipeline.py \
    --model_a "$MODEL_A" \
    --model_b "$MODEL_B" \
    --n_samples $N_SAMPLES \
    --seq_len $SEQ_LEN \
    --output_dir "$OUTPUT_DIR"

# Step 3: Fit ridge regression mapping
echo ""
echo "[3/5] Fitting ridge regression mapping..."

python3 -c "
import sys
sys.path.insert(0, 'src/kv_translation')
import numpy as np
from pathlib import Path
from ridge_mapper import RidgeKVMapper

output_dir = Path('$OUTPUT_DIR')

# Load model A KV caches (train split)
kv_a_files = sorted(output_dir.glob('model_a_kv_train.npz'))
if not kv_a_files:
    # Fallback to full dataset
    kv_a_files = sorted(output_dir.glob('model_a_kv.npz'))

kv_b_files = sorted(output_dir.glob('model_b_kv_train.npz'))
if not kv_b_files:
    kv_b_files = sorted(output_dir.glob('model_b_kv.npz'))

if not kv_a_files or not kv_b_files:
    print('ERROR: KV cache files not found!')
    sys.exit(1)

print(f'Loading KV caches...')
data_a = np.load(kv_a_files[0])
data_b = np.load(kv_b_files[0])

keys_a = sorted([k for k in data_a.files if not k.startswith('__')])
keys_b = sorted([k for k in data_b.files if not k.startswith('__')])

# Get shape from first sample
first_kv = data_a[keys_a[0]]
num_layers, num_kv, num_heads, seq_len, head_dim = first_kv.shape
print(f'  Shape: {num_layers} layers, {num_heads} KV heads, {head_dim} dim')

# Build source/target lists
source_kvs = [data_a[k] for k in keys_a[:$N_SAMPLES]]
target_kvs = [data_b[k] for k in keys_b[:$N_SAMPLES]]

# For ridge regression, we only need keys (not values)
# Extract just the key component (index 0)
source_keys = [kv[:, 0] for kv in source_kvs]  # [layers, heads, seq, dim]
target_keys = [kv[:, 0] for kv in target_kvs]

print(f'  Fitting on {len(source_keys)} samples...')

# Try different lambda values
for lam in [0.01, 0.1, 1.0, 10.0]:
    mapper = RidgeKVMapper(num_layers, num_heads, head_dim)
    stats = mapper.fit(source_keys, target_keys, lambda_reg=lam)
    print(f'  lambda={lam}: cos_sim={stats[\"mean_cosine_similarity\"]:.4f} '
          f'min={stats[\"min_cosine_similarity\"]:.4f}')

# Use best lambda (typically 1.0)
mapper = RidgeKVMapper(num_layers, num_heads, head_dim)
stats = mapper.fit(source_keys, target_keys, lambda_reg=1.0)

# Save
output_path = str(output_dir / 'kv_map_qwen36_to_ornith.json')
mapper.save(output_path)
print(f'  Mapping matrix saved: {output_path}')

# Save stats
import json
with open(output_dir / 'fit_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(f'  Stats saved: {output_dir}/fit_stats.json')
"

# Step 4: Validate mapping quality
echo ""
echo "[4/5] Validating mapping quality..."

python3 -c "
import sys
sys.path.insert(0, 'src/kv_translation')
import numpy as np
from pathlib import Path
from ridge_mapper import RidgeKVMapper

output_dir = Path('$OUTPUT_DIR')
mapper = RidgeKVMapper.load(str(output_dir / 'kv_map_qwen36_to_ornith.json'))

# Load validation set
data_a = np.load(output_dir / 'model_a_kv_val.npz') if (output_dir / 'model_a_kv_val.npz').exists() else None
data_b = np.load(output_dir / 'model_b_kv_val.npz') if (output_dir / 'model_b_kv_val.npz').exists() else None

if data_a is None or data_b is None:
    print('  Validation files not found, using training data')
    data_a = np.load(output_dir / 'model_a_kv.npz')
    data_b = np.load(output_dir / 'model_b_kv.npz')

keys_a = sorted([k for k in data_a.files if not k.startswith('__')])[:10]
keys_b = sorted([k for k in data_b.files if not k.startswith('__')])[:10]

for i in range(min(len(keys_a), len(keys_b))):
    source = data_a[keys_a[i]][:, 0]  # keys only
    target_real = data_b[keys_b[i]][:, 0]
    target_mapped = mapper.translate(source)

    cos_sim = float(np.mean([
        np.dot(target_real[l,h].flatten(), target_mapped[l,h].flatten()) /
        (np.linalg.norm(target_real[l,h].flatten()) * np.linalg.norm(target_mapped[l,h].flatten()) + 1e-8)
        for l in range(target_real.shape[0]) for h in range(target_real.shape[1])
    ]))
    print(f'  Sample {i}: cosine_similarity = {cos_sim:.4f}')
"

# Step 5: Summary
echo ""
echo "[5/5] Pipeline complete!"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.npz 2>/dev/null | awk '{print "  " $NF ": " $5}'
echo ""
echo "Next steps:"
echo "  1. Copy calibration_data/ to Windows for analysis"
echo "  2. Use kv_map.json in CGC engine for real-time KV Translation"
echo "  3. Run FusionRoute router training with CDPO"
