#!/usr/bin/env bash
#SBATCH --job-name=sail_cpu_real
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/bench_cpu_real_%j.out
#SBATCH --error=logs/bench_cpu_real_%j.err

# CPU real-model benchmark for Sail vs Spark on NEU Explorer.
# Requires the CPU Transformers model stack in .venv before submitting.

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

if [ -n "${JAVA_HOME:-}" ]; then
    export PATH="$JAVA_HOME/bin:$PATH"
fi

java_major_version() {
    local version
    version="$(java -version 2>&1 | awk -F '"' '/version/ {print $2; exit}')"
    if [[ "$version" == 1.* ]]; then
        echo "$version" | cut -d. -f2
    else
        echo "$version" | cut -d. -f1
    fi
}

if ! command -v java >/dev/null 2>&1; then
    echo "Error: java is not on PATH. Load OpenJDK 17+ before submitting." >&2
    exit 1
fi

JAVA_MAJOR="$(java_major_version)"
echo "[slurm] java=$(command -v java)"
java -version
if [ "${JAVA_MAJOR:-0}" -lt 17 ]; then
    echo "Error: Spark requires Java 17+. Current java major version is ${JAVA_MAJOR:-unknown}." >&2
    echo "Load a newer module, for example: module load OpenJDK/22.0.2" >&2
    exit 1
fi

REPO_DIR="$(pwd)"
export HF_HOME="$REPO_DIR/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$REPO_DIR/models"
export VENV="$REPO_DIR/.venv"
export MODELS_DIR="$REPO_DIR/models"
# Compute nodes may not have internet, so avoid Hugging Face dataset fetches
# unless the submitter explicitly opts into them with FORCE_SYNTHETIC=0.
export FORCE_SYNTHETIC="${FORCE_SYNTHETIC:-1}"

if [ ! -d "$VENV" ]; then
    echo "Error: Venv $VENV not found. Run scripts/setup_env.sh --mode cpu_real --venv .venv before submitting."
    exit 1
fi

"$VENV/bin/python" - <<'PY'
import importlib
import sys

import transformers

version = getattr(transformers, "__version__", "unknown")
path = getattr(transformers, "__file__", "unknown")
print(f"[slurm] transformers={version} path={path}")

def require_transformers_attr(name: str) -> None:
    if getattr(transformers, name, None) is not None:
        return
    try:
        auto_module = importlib.import_module("transformers.models.auto")
    except Exception as exc:
        raise SystemExit(
            f"Error: cannot import transformers.models.auto while checking {name}: {exc}"
        )
    if getattr(auto_module, name, None) is None:
        raise SystemExit(
            f"Error: transformers {version} does not provide {name}. "
            "Re-run scripts/setup_env.sh --mode cpu_real --venv .venv with transformers>=4.51.0."
        )

for attr in ("AutoTokenizer", "AutoModelForCausalLM", "AutoModelForSequenceClassification"):
    require_transformers_attr(attr)
PY

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
echo "[slurm] FORCE_SYNTHETIC=$FORCE_SYNTHETIC"
bash scripts/run_benchmark.sh --mode cpu_real --venv "$VENV"

echo "[slurm] Done. Report: $RESULTS_DIR/report/aggregate.html"
