#!/usr/bin/env bash
# Step 2: Run this on a GPU COMPUTE NODE (e.g. via srun)

set -euo pipefail

# 1. Environment & Modules
export PYTHONNOUSERSITE=1
export CARGO_HOME="/scratch/yirga.t/.cargo"
export RUSTUP_HOME="/scratch/yirga.t/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"

module purge
module load anaconda3/2024.06 || true
module load cuda/12.8.0 || true
module load OpenJDK/22.0.2 || true

# Explicitly set CUDA paths for compilation
export CUDA_HOME="/shared/EL9/explorer/cuda/12.8.0"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

REPO_DIR="$(pwd)"
OFFLINE_DIR="$REPO_DIR/offline_pkgs"

if [ ! -d "$OFFLINE_DIR" ]; then
    echo "Error: $OFFLINE_DIR not found. Run slurm/prep_download.sh on login node first."
    exit 1
fi

# 2. Setup Virtual Environments
setup_venv() {
    local venv_path="$1"
    local is_gpu="$2"
    
    echo "--- Preparing isolated venv: $venv_path ---"
    if [ ! -d "$venv_path" ]; then
        python -m venv "$venv_path"
    fi

    echo "Updating base tools (OFFLINE)..."
    "$venv_path/bin/python" -m pip install --no-index --find-links="$OFFLINE_DIR" --upgrade pip "setuptools<82" wheel maturin ninja

    if [ "$is_gpu" = true ]; then
        # vLLM brings its own torch + triton — install it first so its torch
        # version wins. transformers/sentence-transformers (for scorer/embedder)
        # are installed after and will use vLLM's torch.
        echo "--- Installing vLLM (OFFLINE) ---"
        "$venv_path/bin/python" -m pip install --no-index --find-links="$OFFLINE_DIR" vllm

        echo "--- Installing scorer/embedder stack (OFFLINE) ---"
        "$venv_path/bin/python" -m pip install --no-index --find-links="$OFFLINE_DIR" \
                        "numpy>=2.0" "transformers>=4.51.0" "tokenizers>=0.21" \
                        "sentence-transformers" accelerate
    else
        echo "--- Installing CPU AI stack (OFFLINE) ---"
        "$venv_path/bin/python" -m pip install --no-index --find-links="$OFFLINE_DIR" \
                        "numpy>=2.0" "transformers>=4.51.0" "tokenizers>=0.21" "sentence-transformers" "torch"
    fi

    echo "--- Installing benchmark package (OFFLINE) ---"
    "$venv_path/bin/python" -m pip install --no-index --find-links="$OFFLINE_DIR" -r requirements.txt
    "$venv_path/bin/python" -m pip install --no-index --find-links="$OFFLINE_DIR" -e . --no-build-isolation
}

setup_venv ".venv" false
setup_venv ".venv_gpu" true

# 3. Build Sail (Offline)
SAIL_REPO_DIR="$REPO_DIR/third_party/sail"
if [ -d "$SAIL_REPO_DIR" ]; then
    echo "--- Building Sail (OFFLINE) ---"
    cd "$SAIL_REPO_DIR"
    "$REPO_DIR/.venv_gpu/bin/python" -m maturin develop --release --offline
    cd "$REPO_DIR"
fi

echo "=========================================================="
echo "✅ OFFLINE INSTALL COMPLETE"
echo "=========================================================="
