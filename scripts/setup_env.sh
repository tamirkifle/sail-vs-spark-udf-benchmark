#!/usr/bin/env bash
# Create/update a local Python environment for benchmark runs.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/setup_env.sh [--mode mock|cpu|cpu_real|gpu|dev] [--venv .venv]

Modes:
  mock  Base benchmark dependencies only. Fastest path; models are mocked.
  cpu   Base deps plus torch/transformers/sentence-transformers.
  cpu_real
        CPU deps plus Transformers/Accelerate real-model stack.
  gpu   CPU deps plus vLLM/accelerate/bitsandbytes for CUDA hosts.
  dev   Base deps for tests and development.
USAGE
}

MODE="${MODE:-mock}"
VENV="${VENV:-.venv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --venv)
      VENV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  mock|cpu|cpu_real|gpu|dev) ;;
  *)
    echo "invalid mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"
if [[ "$VENV" != /* ]]; then
  VENV="$REPO_DIR/$VENV"
fi

MODELS_DIR="${MODELS_DIR:-$REPO_DIR/models}"
mkdir -p "$MODELS_DIR"
export HF_HOME="${HF_HOME:-$MODELS_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$MODELS_DIR/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$MODELS_DIR}"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install -e .
"$PY" -m pip install -r requirements.txt

if [[ "$MODE" == "cpu_real" ]]; then
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  "$PY" -m pip install "transformers>=4.51.0" "sentence-transformers>=3.0.0" "accelerate>=0.26.0"
elif [[ "$MODE" == "cpu" || "$MODE" == "gpu" ]]; then
  "$PY" -m pip install torch "transformers>=4.51.0" "sentence-transformers>=3.0.0" "accelerate>=0.26.0"
fi

if [[ "$MODE" == "gpu" ]]; then
  "$PY" -m pip install vllm bitsandbytes
fi

echo "[setup] ready: mode=$MODE venv=$VENV python=$PY"
