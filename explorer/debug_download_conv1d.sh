#!/usr/bin/env bash
# Debugging script for causal-conv1d download

set -euo pipefail

echo "--- 1. Loading Modules ---"
module load anaconda3/2024.06 || true
module load cuda/12.8.0 || true

# Explicitly set paths
export CUDA_HOME="/shared/EL9/explorer/cuda/12.8.0"
export PATH="$CUDA_HOME/bin:$PATH"

echo "--- 2. Environment Check ---"
if command -v nvcc >/dev/null 2>&1; then
    echo "✓ nvcc found at: $(which nvcc)"
    nvcc --version | grep release
else
    echo "❌ ERROR: nvcc not found! Check your 'module load' paths."
    exit 1
fi

echo "--- 3. Attempting Download (Clean) ---"
mkdir -p debug_pkgs
# We use --no-deps and --no-build-isolation to stop pip from trying to 
# 'inspect' or 'build' the package on the login node.
python -m pip download -d debug_pkgs \
    --no-deps \
    --no-build-isolation \
    causal-conv1d

echo "--- 4. Success ---"
ls -l debug_pkgs/causal_conv1d*
