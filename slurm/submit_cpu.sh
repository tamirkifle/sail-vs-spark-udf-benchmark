#!/usr/bin/env bash
#SBATCH --job-name=sail_bench_cpu
#SBATCH --partition=short
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/bench_cpu_%j.out
#SBATCH --error=logs/bench_cpu_%j.err

# Laptop-scale (CPU) benchmark for Sail vs Spark on NEU Explorer.
# NOTE: Run slurm/prep_download.sh and slurm/prep_install.sh before submitting.

set -euo pipefail

export PYTHONNOUSERSITE=1
export CARGO_HOME="/scratch/yirga.t/.cargo"
export RUSTUP_HOME="/scratch/yirga.t/.rustup"
# Pin cargo/rustc to the controlled rustup installation in scratch.
# Must come before module loads so the module-provided rust never shadows it.
export PATH="$CARGO_HOME/bin:$PATH"

# Load the same modules used during prep.
module purge
module load anaconda3/2024.06 || true
module load OpenJDK/22.0.2 || true
# NOTE: do NOT load a rust module — CARGO_HOME/RUSTUP_HOME above provide a
# stable, version-pinned toolchain. Loading the cluster rust module puts a
# different rustc in PATH which invalidates all cargo fingerprints and forces
# a full rebuild on every job.

REPO_DIR="$(pwd)"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export VENV="$REPO_DIR/.venv"

# Ensure venv exists
if [ ! -d "$VENV" ]; then
    echo "Error: Venv $VENV not found. Run slurm/prep_download.sh and slurm/prep_install.sh first."
    exit 1
fi

# Build Sail on the compute node
SAIL_REPO_DIR="$REPO_DIR/third_party/sail"
echo "Building Sail v0.6.0 on compute node (OFFLINE)..."
cd "$SAIL_REPO_DIR"
"$VENV/bin/python" -m maturin develop --release --offline
cd "$REPO_DIR"

# Generate a unique timestamp for this run
TS=$(date +%Y%m%d_%H%M%S)
export RESULTS_DIR="results/laptop_live/$TS"
mkdir -p "$RESULTS_DIR"

echo "[explorer] Starting CPU benchmarks (laptop-scale) -> $RESULTS_DIR"
bash scripts/run_all_laptop.sh

echo "[explorer] Done. Aggregating results in $RESULTS_DIR..."
"$VENV/bin/python" analysis/aggregate_results.py --results_dir "$RESULTS_DIR"
