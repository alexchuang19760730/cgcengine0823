#!/usr/bin/env bash
# ============================================================
# Run CGC llama.cpp on Windows
# Auto-detects DLLs in script directory, falls back to MSYS2
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Add script dir to PATH (for bundled DLLs)
export PATH="$SCRIPT_DIR:$PATH"

# Fallback to MSYS2 if DLLs not found locally
if ! ldd "$SCRIPT_DIR/llama-simple.exe" 2>/dev/null | grep -q "not found"; then
    :
else
    export PATH="/c/msys64/mingw64/bin:$PATH"
fi

# Defaults
MODEL="${MODEL:-}"
BINARY="${BINARY:-llama-simple.exe}"
NGL="${NGL:-4}"
CTX="${CTX:-2048}"
THREADS="${THREADS:-4}"
EXPERT_CACHE="${EXPERT_CACHE:-0}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m) MODEL="$2"; shift 2 ;;
        -ngl) NGL="$2"; shift 2 ;;
        -c) CTX="$2"; shift 2 ;;
        -t) THREADS="$2"; shift 2 ;;
        --mtp) BINARY="llama-speculative-simple.exe"; MTP_N="${2:-2}"; shift 2 ;;
        --expert-cache) EXPERT_CACHE="$2"; shift 2 ;;
        --server) BINARY="llama-server.exe"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 -m <model.gguf> [--server|--mtp N] [--expert-cache BYTES]"
    echo ""
    echo "Examples:"
    echo "  $0 -m model.gguf                          # Basic decode"
    echo "  $0 -m model.gguf --server                  # OpenAI API server"
    echo "  $0 -m model.gguf --mtp 2                   # MTP speculative decode"
    echo "  $0 -m model.gguf --expert-cache 4294967296 # 4GB expert cache"
    exit 1
fi

# Expert cache env
if [[ "$EXPERT_CACHE" != "0" ]]; then
    export CGC_EXPERT_CACHE_BYTES="$EXPERT_CACHE"
    export LLAMA_EXPERT_CACHE_ALLOW_NGL=1
    export LLAMA_EXPERT_CACHE_L4_SKIP_LAYER0=1
    export LLAMA_EXPERT_CACHE_WORKERS=4
fi

echo "=== CGC llama.cpp (Windows) ==="
echo "  Binary: $BINARY"
echo "  Model:  $MODEL"
echo "  NGL:    $NGL"
echo "  CTX:    $CTX"
echo "  Cache:  $EXPERT_CACHE bytes"
echo ""

CMD=("$SCRIPT_DIR/$BINARY" -m "$MODEL" -ngl "$NGL" -c "$CTX" -t "$THREADS")

if [[ "$BINARY" == "llama-speculative-simple.exe" ]]; then
    CMD+=(--spec-type draft-mtp --spec-draft-n-max "${MTP_N:-2}" --temp 0)
fi

if [[ "$BINARY" == "llama-server.exe" ]]; then
    CMD+=(--host 0.0.0.0 --port 1234)
fi

exec "${CMD[@]}"
