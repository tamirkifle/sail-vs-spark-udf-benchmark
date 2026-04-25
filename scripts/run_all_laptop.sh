#!/usr/bin/env bash
# Run the full 4×4 benchmark matrix on a laptop (CPU/MPS, 100 rows).
#
# Behaviour
# ─────────
# - Prepares the dataset once (synthetic if HF download fails).
# - For Configs A and B invokes the CLI with the sail venv's python directly.
# - For Configs C and D invokes via ``sail spark run -f <driver>`` — the
#   driver script imports the CLI and invokes ``run_one`` with the right
#   args. This is the correct invocation for Sail 0.5.3+ (see prior
#   learnings §2).
# - Writes every run's artefacts to ``results/laptop/<run_id>/...``.

set -euo pipefail

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"

VENV="${VENV:-/Users/tamir/Documents/MyCode/LakeSail/sail/.venvs/default}"
PY="$VENV/bin/python"
SAIL="$VENV/bin/sail"
CONFIG="${CONFIG:-config/laptop.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results/laptop}"

mkdir -p "$RESULTS_DIR" data/laptop

echo "[run_all_laptop] prepping dataset"
"$PY" scripts/prep_dataset.py --config "$CONFIG" --force-synthetic


run_spark_cli() {
  local workload="$1" execution="$2" extra_args="${3:-}"
  local rid="${workload}_${execution}$( [ -n "$extra_args" ] && echo "_$(echo "$extra_args" | tr -d ' -')" )"
  echo "[run_all_laptop] ▶ $rid  (python direct)"
  "$PY" -m sail_vs_spark.runner.cli \
      --config "$CONFIG" \
      --workload "$workload" \
      --execution "$execution" \
      --results-dir "$RESULTS_DIR" \
      --run-id "$rid" \
      $extra_args
}

run_sail_cli() {
  local workload="$1" execution="$2" extra_args="${3:-}"
  local rid="${workload}_${execution}$( [ -n "$extra_args" ] && echo "_$(echo "$extra_args" | tr -d ' -')" )"
  local driver="/tmp/sail_vs_spark_driver_$$.py"
  cat >"$driver" <<PYEOF
import sys
sys.path.insert(0, "$REPO_DIR/src")
from sail_vs_spark.runner.cli import main
raise SystemExit(main([
    "--config", "$CONFIG",
    "--workload", "$workload",
    "--execution", "$execution",
    "--results-dir", "$RESULTS_DIR",
    "--run-id", "$rid",
    $(echo "$extra_args" | awk '{for(i=1;i<=NF;i++) printf "\"%s\", ", $i}')
]))
PYEOF
  echo "[run_all_laptop] ▶ $rid  (sail spark run)"
  "$SAIL" spark run -f "$driver"
  rm -f "$driver"
}


# ──────────── W0: depth 1, 2, 3 × {A, B, C, D}  ────────────
for depth in 1 2 3; do
  for cfg in A B; do
    run_spark_cli "w0" "$cfg" "--depth $depth"
  done
  for cfg in C D; do
    run_sail_cli  "w0" "$cfg" "--depth $depth"
  done
done

# ──────────── W1 / W2 / W3 × {A, B, C, D} ────────────
for wl in w1 w2 w3; do
  for cfg in A B; do
    run_spark_cli "$wl" "$cfg"
  done
  for cfg in C D; do
    run_sail_cli  "$wl" "$cfg"
  done
done

echo "[run_all_laptop] all 18 runs complete. Artefacts → $RESULTS_DIR"
echo "[run_all_laptop] next: python analysis/aggregate_results.py --results_dir $RESULTS_DIR"
