#!/usr/bin/env bash
#SBATCH --job-name=sail_cpu_real
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/bench_cpu_real_%j.out
#SBATCH --error=logs/bench_cpu_real_%j.err

# CPU real-model benchmark for Sail vs Spark on NEU Explorer.
# Requires a CPU-capable vLLM install in .venv before submitting.

set -euo pipefail

export PYTHONNOUSERSITE=1
export CARGO_HOME="/scratch/yirga.t/.cargo"
export RUSTUP_HOME="/scratch/yirga.t/.rustup"
# Pin cargo/rustc to the controlled rustup installation in scratch.
# Must come before module loads so the module-provided rust never shadows it.
export PATH="$CARGO_HOME/bin:$PATH"

module purge
module load anaconda3/2024.06 || true
module load OpenJDK/22.0.2 || true

REPO_DIR="$(pwd)"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$REPO_DIR/models"
export VENV="$REPO_DIR/.venv"
export MODELS_DIR="$REPO_DIR/models"

if [ ! -d "$VENV" ]; then
    echo "Error: Venv $VENV not found. Run scripts/setup_env.sh --mode cpu_real --venv .venv before submitting."
    exit 1
fi

if ! "$VENV/bin/python" -c "import vllm" >/dev/null 2>&1; then
    echo "Error: vLLM is not importable from $VENV."
    echo "Run scripts/setup_env.sh --mode cpu_real --venv .venv on a compatible Linux x86_64 node first."
    exit 1
fi

SAIL_REPO_DIR="$REPO_DIR/third_party/sail"
if [ -d "$SAIL_REPO_DIR" ]; then
    echo "Building Sail v0.6.0 on compute node..."
    cd "$SAIL_REPO_DIR"
    "$VENV/bin/python" -m maturin develop --release --offline
    cd "$REPO_DIR"
fi

TS=$(date +%Y%m%d_%H%M%S)
export RESULTS_DIR="results/cpu_real/$TS"
mkdir -p "$RESULTS_DIR"

echo "[slurm] Starting CPU real-model benchmarks -> $RESULTS_DIR"
bash scripts/run_benchmark.sh --mode cpu_real --venv "$VENV"

echo "[slurm] Done. Report: $RESULTS_DIR/report/aggregate.html"
