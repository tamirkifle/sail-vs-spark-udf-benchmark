#!/usr/bin/env bash
#SBATCH --job-name=sail_gpu_smoke
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/bench_gpu_smoke_%j.out
#SBATCH --error=logs/bench_gpu_smoke_%j.err

# GPU Smoke Test for Sail vs Spark on NEU Explorer.
# Matching production architecture (FP8 + MoE) on an H100.

set -euo pipefail

export PYTHONNOUSERSITE=1
export CARGO_HOME="/scratch/yirga.t/.cargo"
export RUSTUP_HOME="/scratch/yirga.t/.rustup"
# Pin cargo/rustc to the controlled rustup installation before module loads.
export PATH="$CARGO_HOME/bin:$PATH"

# Load the same modules used during prep.
module purge
module load anaconda3/2024.06 || true
module load cuda/12.8.0 || true
module load OpenJDK/22.0.2 || true

# Explicitly export CUDA paths for Spark workers
export CUDA_HOME="/shared/EL9/explorer/cuda/12.8.0"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PATH="$CUDA_HOME/bin:$PATH"

REPO_DIR="$(pwd)"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export VENV="$REPO_DIR/.venv_gpu"

# Enable strict offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Build Sail on the compute node
SAIL_REPO_DIR="$REPO_DIR/third_party/sail"
echo "Building Sail v0.6.0 on compute node (smoke test, OFFLINE)..."
cd "$SAIL_REPO_DIR"
# export PYO3_PYTHON="$VENV/bin/python3"
# export PYO3_ENVIRONMENT_SIGNATURE="pysail-bench-gpu"
"$VENV/bin/python" -m maturin develop --release --offline
cd "$REPO_DIR"

# Generate a unique timestamp for this run
TS=$(date +%Y%m%d_%H%M%S)
export RESULTS_DIR="results/gpu_smoke/$TS"
mkdir -p "$RESULTS_DIR"

echo "[explorer] Starting GPU Smoke Test (H100 / FP8) -> $RESULTS_DIR"
export CONFIG="config/gpu_v100_smoke.yaml"
bash scripts/run_all_gpu.sh

echo "[explorer] Done. Aggregating results in $RESULTS_DIR..."
"$VENV/bin/python" analysis/aggregate_results.py --results_dir "$RESULTS_DIR"
