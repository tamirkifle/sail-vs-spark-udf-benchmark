#!/usr/bin/env bash
# Run the full 4×4 benchmark matrix on a laptop using a persistent Sail server.

set -euo pipefail

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"

export HF_HOME="$REPO_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

VENV="${VENV:-.venv}"
PY="$VENV/bin/python"
ITERATIONS="${ITERATIONS:-3}"

# Use an array for the command to correctly handle fallback to 'python -m'
if [ -f "$VENV/bin/sail" ]; then
    SAIL_CMD=("$VENV/bin/sail")
else
    SAIL_CMD=("$PY" "-m" "pysail.cli")
fi

CONFIG="${CONFIG:-config/laptop.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_RESULTS_DIR="results/laptop/$TIMESTAMP"
RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"

# Dynamic port allocation
BASE_PORT=50000
JOB_OFFSET=$(( ${SLURM_JOB_ID:-$$} % 1000 ))
PORT=$(( BASE_PORT + JOB_OFFSET ))
IP=127.0.0.1

mkdir -p "$RESULTS_DIR" data/laptop

echo "[laptop] prepping dataset"
"$PY" scripts/prep_dataset.py --config "$CONFIG" --force-synthetic

# 1. Start Sail Server in background
echo "[laptop] starting Sail server on $IP:$PORT..."

# Export offline environment for the Sail server (so it passes it to workers)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

"${SAIL_CMD[@]}" spark server --ip "$IP" --port "$PORT" > "$RESULTS_DIR/sail_server.log" 2>&1 &
SAIL_PID=$!

trap "echo '[laptop] stopping Sail server (PID $SAIL_PID)...'; kill $SAIL_PID 2>/dev/null || true" EXIT

echo "[laptop] waiting for server to wake up on port $PORT..."
for i in {1..60}; do
    if ! kill -0 $SAIL_PID 2>/dev/null; then
        echo "[laptop] ERROR: Sail server process died unexpectedly!"
        tail -n 20 "$RESULTS_DIR/sail_server.log"
        exit 1
    fi
    if python3 -c "import socket; s = socket.socket(); s.connect(('$IP', $PORT))" >/dev/null 2>&1; then
        echo "[laptop] server is UP on $IP:$PORT"
        break
    fi
    sleep 1
done

run_bench() {
  local workload="$1" execution="$2" extra_args="${3:-}"
  local base_rid="${workload}_${execution}$( [ -n "$extra_args" ] && echo "_$(echo "$extra_args" | tr -d ' -')" )"

  echo "[laptop] ▶ ${base_rid}_s1..s${ITERATIONS} ($ITERATIONS samples, shared session)"

  if [[ "$execution" == "C" || "$execution" == "D" ]]; then
    export SPARK_REMOTE="sc://$IP:$PORT"
  else
    unset SPARK_REMOTE
  fi

  "$PY" -m sail_vs_spark.runner.cli \
      --config "$CONFIG" \
      --workload "$workload" \
      --execution "$execution" \
      --results-dir "$RESULTS_DIR" \
      --run-id "$base_rid" \
      --samples "$ITERATIONS" \
      $extra_args
}

# W1 / W2 / W3 / W4 (AI Workloads)
for wl in w4 w1 w2 w3; do
  for cfg in C D A B; do
    run_bench "$wl" "$cfg"
  done
done

# W0: depth 1, 2, 3 (Trivial workloads)
for depth in 1 2 3; do
  for cfg in A B C D; do
    run_bench "w0" "$cfg" "--depth $depth"
  done
done



echo "[laptop] complete ($ITERATIONS samples each). results -> $RESULTS_DIR"
