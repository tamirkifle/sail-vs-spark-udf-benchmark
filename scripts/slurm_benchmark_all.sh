#!/usr/bin/env bash
#SBATCH --job-name=sail_vs_spark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/bench_all_%j.out
#SBATCH --error=logs/bench_all_%j.err

# Full Sail-vs-Spark benchmark submitted to Northeastern Discovery cluster.
# All 4 configs × 4 workloads run sequentially inside a single job so they
# share identical hardware/driver/CUDA state (prior learnings §9).

set -euo pipefail

module purge
module load anaconda3
conda activate sail

cd /scratch/yirga.t/sail_vs_spark_benchmark
mkdir -p logs

# Single environment variable cascaded through all runs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"
export HF_HOME="${HF_HOME:-$MODELS_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$MODELS_DIR/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$MODELS_DIR}"
export VENV="$HOME/.conda/envs/sail"

echo "[slurm] node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Iseconds)"
nvidia-smi || true

RESULTS_DIR="${RESULTS_DIR:-results/gpu/$(date +%Y%m%d_%H%M%S)}"
export RESULTS_DIR
bash scripts/run_all_gpu.sh
echo "[slurm] done=$(date -Iseconds)"
echo "[slurm] aggregated. Copy results back with:"
echo "       scp -r yirga.t@discovery.northeastern.edu:$(pwd)/$RESULTS_DIR ~/Documents/MyCode/LakeSail/benchmark_results/"
