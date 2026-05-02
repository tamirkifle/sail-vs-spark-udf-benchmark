#!/usr/bin/env bash
#SBATCH --job-name=sail_bench_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/bench_gpu_%j.out
#SBATCH --error=logs/bench_gpu_%j.err

# GPU benchmark for Sail vs Spark on NEU Explorer (H200).
# NOTE: Run slurm/prep_download.sh and slurm/prep_install.sh before submitting.

set -euo pipefail

export PYTHONNOUSERSITE=1
export CARGO_HOME="/scratch/yirga.t/.cargo"
export RUSTUP_HOME="/scratch/yirga.t/.rustup"

# Load the same modules used during prep.
module purge
module load anaconda3/2024.06 || true
module load cuda/12.8.0 || true
module load OpenJDK/22.0.2 || true

REPO_DIR="$(pwd)"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export VENV="$REPO_DIR/.venv_gpu"

# Enable strict offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Ensure venv exists
if [ ! -d "$VENV" ]; then
    echo "Error: Venv $VENV not found. Run slurm/prep_download.sh and slurm/prep_install.sh first."
    exit 1
fi

# Build Sail on the compute node
SAIL_REPO_DIR="$REPO_DIR/third_party/sail"
echo "Building Sail v0.6.0 on GPU compute node (OFFLINE)..."
cd "$SAIL_REPO_DIR"
"$VENV/bin/python" -m maturin develop --release --offline
cd "$REPO_DIR"

# Generate a unique timestamp for this run
TS=$(date +%Y%m%d_%H%M%S)
export RESULTS_DIR="results/gpu/$TS"
mkdir -p "$RESULTS_DIR"

echo "[explorer] Starting GPU benchmarks (H200) -> $RESULTS_DIR"
bash scripts/run_all_gpu.sh

echo "[explorer] Done. Aggregating results in $RESULTS_DIR..."
"$VENV/bin/python" analysis/aggregate_results.py --results_dir "$RESULTS_DIR"
