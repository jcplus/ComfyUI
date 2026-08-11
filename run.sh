#!/usr/bin/env bash
# ComfyUI launcher tuned for Apple Silicon with 16 GB unified memory.
#
# Usage:
#   ./run.sh                 # low profile (default) - large models, lowest memory use
#   ./run.sh balanced        # SD1.5 / SDXL
#   ./run.sh fast            # small models, keeps caches hot
#   ./run.sh --port 8288     # any extra flags are passed to main.py
set -euo pipefail
cd "$(dirname "$0")"

PROFILE="low"
case "${1:-}" in
    balanced|low|fast) PROFILE="$1"; shift ;;
esac

PY=".venv/bin/python"
[ -x "$PY" ] || { echo "No .venv found. Create it with: uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt"; exit 1; }

# --- Apple Silicon environment ---------------------------------------------
# Unimplemented MPS operators run on the CPU instead of raising an error.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Keep CPU work on the performance cores. Efficiency cores slow down the
# tokenizer / VAE / data loading paths when they join the thread pool.
export OMP_NUM_THREADS="$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 4)"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# HuggingFace tokenizers fork warning.
export TOKENIZERS_PARALLELISM=false
# Metal reports a recommended working set of ~11.8 GB on this 16 GB machine.
# PyTorch defaults to 1.7x that (~20 GB), which is more than the machine has:
# the allocator keeps saying yes until macOS swaps and the whole desktop stalls.
# Ratio 1.0 keeps every allocation inside physical memory, so a too-large model
# fails with a clean OOM instead of freezing the Mac.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
# Never set the ratio to 0.0 - that disables the ceiling and brings the stall back.

# --- Common flags -----------------------------------------------------------
# --use-pytorch-cross-attention: SDPA on MPS. Without it ComfyUI falls back to
#   the sub-quadratic path, which is slower on Apple Silicon.
# --preview-method latent2rgb: near-free previews (TAESD costs extra memory).
# --disable-auto-launch: no browser window on every restart.
ARGS=(
    --use-pytorch-cross-attention
    --preview-method latent2rgb
    --disable-auto-launch
)

case "$PROFILE" in
    balanced)
        # ComfyUI now sizes the device at the Metal limit (~11.8 GB), which
        # already excludes what macOS holds, so the reserve can be small.
        ARGS+=(--reserve-vram 1 --cache-ram 2)
        ;;
    low)
        # Flux.2 / video. The 9B GGUF and the 8B text encoder cannot be resident
        # at the same time, so nothing is cached and every model is offloaded
        # back to RAM as soon as its node finishes.
        # --cpu-vae: the VAE is only ~160 MB of weights but its attention block
        #   needs several GB of Metal scratch at full resolution. On a 16 GB Mac
        #   that scratch is what tips the machine into swap. Running it on the CPU
        #   costs a few seconds and takes the whole VAE stage off the Metal budget.
        ARGS+=(--reserve-vram 1.5 --cache-none --disable-smart-memory --cpu-vae)
        ;;
    fast)
        # Small models only. Keeps results cached, so re-runs skip node execution.
        ARGS+=(--reserve-vram 1 --cache-lru 4)
        ;;
esac

echo "ComfyUI profile: $PROFILE | threads: $OMP_NUM_THREADS"
exec "$PY" main.py "${ARGS[@]}" "$@"
