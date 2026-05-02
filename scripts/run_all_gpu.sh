#!/usr/bin/env bash
# Compatibility wrapper. Prefer scripts/run_benchmark.sh --mode gpu.

set -euo pipefail

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"

MODE="${MODE:-gpu}"
CONFIG="${CONFIG:-config/gpu_h200.yaml}"
VENV="${VENV:-.venv_gpu}"
exec scripts/run_benchmark.sh --mode "$MODE" --config "$CONFIG" --venv "$VENV" "$@"
