#!/usr/bin/env bash
# Run the full 4×4 benchmark matrix on GPU (H200 141GB, 10K rows).
#
# Expected use: submitted via SLURM with a single gres request so all 4 configs
# run on the same hardware/driver/CUDA state (per prior learnings §9).

set -euo pipefail

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"

VENV="${VENV:-$HOME/.conda/envs/sail}"
PY="$VENV/bin/python"
SAIL="$VENV/bin/sail"
CONFIG="${CONFIG:-config/gpu_h200.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results/gpu}"

mkdir -p "$RESULTS_DIR" data/gpu
echo "[run_all_gpu] prepping dataset (10K rows)"
"$PY" scripts/prep_dataset.py --config "$CONFIG" || \
    "$PY" scripts/prep_dataset.py --config "$CONFIG" --force-synthetic

run_spark_cli() {
  local workload="$1" execution="$2" extra_args="${3:-}"
  local rid="${workload}_${execution}$( [ -n "$extra_args" ] && echo "_$(echo "$extra_args" | tr -d ' -')" )"
  "$PY" -m sail_vs_spark.runner.cli \
      --config "$CONFIG" --workload "$workload" --execution "$execution" \
      --results-dir "$RESULTS_DIR" --run-id "$rid" $extra_args
}


run_sail_cli() {
  local workload="$1" execution="$2" extra_args="${3:-}"
  local rid="${workload}_${execution}$( [ -n "$extra_args" ] && echo "_$(echo "$extra_args" | tr -d ' -')" )"
  local driver="/tmp/sail_gpu_driver_$$.py"
  cat >"$driver" <<PYEOF
import sys
sys.path.insert(0, "$REPO_DIR/src")
from sail_vs_spark.runner.cli import main
raise SystemExit(main([
    "--config", "$CONFIG",
    "--workload", "$workload", "--execution", "$execution",
    "--results-dir", "$RESULTS_DIR", "--run-id", "$rid",
    $(echo "$extra_args" | awk '{for(i=1;i<=NF;i++) printf "\"%s\", ", $i}')
]))
PYEOF
  "$SAIL" spark run -f "$driver"
  rm -f "$driver"
}

# Run order per prior learnings §9: A → B → C → D (slowest first so it fails
# soonest; C/D most important for the writeup).
for depth in 1 2 3; do
  run_spark_cli "w0" "A" "--depth $depth"
  run_spark_cli "w0" "B" "--depth $depth"
  run_sail_cli  "w0" "C" "--depth $depth"
  run_sail_cli  "w0" "D" "--depth $depth"
done
for wl in w1 w2 w3; do
  run_spark_cli "$wl" "A"
  run_spark_cli "$wl" "B"
  run_sail_cli  "$wl" "C"
  run_sail_cli  "$wl" "D"
done
echo "[run_all_gpu] complete. Artefacts → $RESULTS_DIR"
