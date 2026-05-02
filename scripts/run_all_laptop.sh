#!/usr/bin/env bash
# Compatibility wrapper. Prefer scripts/run_benchmark.sh --mode cpu.

set -euo pipefail

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"

MODE="${MODE:-cpu}"
CONFIG="${CONFIG:-config/cpu.yaml}"
exec scripts/run_benchmark.sh --mode "$MODE" --config "$CONFIG" "$@"
