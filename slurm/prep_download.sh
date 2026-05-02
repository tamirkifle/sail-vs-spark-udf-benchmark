#!/usr/bin/env bash
# Step 1: Run this on the LOGIN NODE (has internet)

set -euo pipefail

# Environment & Modules
export PYTHONNOUSERSITE=1
export CARGO_HOME="/scratch/yirga.t/.cargo"
export RUSTUP_HOME="/scratch/yirga.t/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"

REPO_DIR="$(pwd)"
OFFLINE_DIR="$REPO_DIR/offline_pkgs"
# This MUST match the HF_HOME used in your submission scripts
export HF_HOME="$REPO_DIR/.cache/huggingface"

mkdir -p "$OFFLINE_DIR" "$HF_HOME" models

# Load modules so that 'nvcc' is visible.
module load anaconda3/2024.06 || true
module load cuda/12.8.0 || true
export CUDA_HOME="/shared/EL9/explorer/cuda/12.8.0"
export PATH="$CUDA_HOME/bin:$PATH"

echo "--- Downloading Python Wheels and Source ---"
python -m pip download -d "$OFFLINE_DIR" \
    "numpy>=2.0" "transformers>=4.51.0" "tokenizers>=0.21" \
    "sentence-transformers" accelerate bitsandbytes "einops" \
    "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" \
    "triton==3.2.0" "sympy==1.13.1" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple

# vLLM replaces the Transformers generator stack. It handles FP8 quantization
# natively — causal-conv1d, finegrained-fp8, and deep-gemm are no longer needed.
# --only-binary=:all: ensures pip downloads a pre-built wheel and never falls
# back to a source tarball (which would trigger a build-dep install here).
echo "  - Downloading vLLM pre-built wheel..."
python -m pip download -d "$OFFLINE_DIR" "vllm>=0.4.0" \
    --only-binary=:all: \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Download build tools
python -m pip download -d "$OFFLINE_DIR" ninja "setuptools<82" wheel maturin

# Download requirements.txt dependencies
python -m pip download -d "$OFFLINE_DIR" -r requirements.txt

echo "--- Pre-downloading HuggingFace Models ---"
python -m pip install huggingface-hub pyyaml

# Download the benchmark models (scorer + embedder; generator is served by vLLM)
python scripts/download_models.py

echo "--- Pre-fetching Rust dependencies ---"
SAIL_REPO_DIR="$REPO_DIR/third_party/sail"
if [ -d "$SAIL_REPO_DIR" ]; then
    module load rust >/dev/null 2>&1 || module load rustc >/dev/null 2>&1 || true
    if command -v cargo >/dev/null 2>&1; then
        echo "✓ cargo found: $(cargo --version)"
        cd "$SAIL_REPO_DIR"
        cargo fetch
        cd "$REPO_DIR"
    fi
fi

echo "=========================================================="
echo "✅ DOWNLOAD COMPLETE"
echo "Now run: srun --partition=gpu ... slurm/prep_install.sh"
echo "=========================================================="
