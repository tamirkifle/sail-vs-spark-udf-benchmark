#!/usr/bin/env bash
# Run the full 4×4 benchmark matrix on GPU using a persistent Sail server.

set -euo pipefail

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"

export HF_HOME="$REPO_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

VENV="${VENV:-.venv_gpu}"
PY="$VENV/bin/python"
ITERATIONS="${ITERATIONS:-3}"

# Use an array for the command to correctly handle fallback to 'python -m'
if [ -f "$VENV/bin/sail" ]; then
    SAIL_CMD=("$VENV/bin/sail")
else
    SAIL_CMD=("$PY" "-m" "pysail.cli")
fi

CONFIG="${CONFIG:-config/gpu_h200.yaml}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_RESULTS_DIR="results/gpu/$TIMESTAMP"
RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"

# Dynamic port allocation
BASE_PORT=50000
JOB_OFFSET=$(( ${SLURM_JOB_ID:-$$} % 1000 ))
PORT=$(( BASE_PORT + JOB_OFFSET ))
IP=127.0.0.1

mkdir -p "$RESULTS_DIR" data/gpu

echo "[gpu] prepping dataset"
"$PY" scripts/prep_dataset.py --config "$CONFIG"

# 1. Start Sail Server in background
echo "[gpu] starting Sail server on $IP:$PORT..."
"${SAIL_CMD[@]}" spark server --ip "$IP" --port "$PORT" > "$RESULTS_DIR/sail_server.log" 2>&1 &
SAIL_PID=$!

trap "echo '[gpu] stopping servers...'; kill $SAIL_PID ${VLLM_PID:-} 2>/dev/null || true" EXIT

echo "[gpu] waiting for server to wake up on port $PORT..."
for i in {1..60}; do
    if ! kill -0 $SAIL_PID 2>/dev/null; then
        echo "[gpu] ERROR: Sail server process died unexpectedly!"
        tail -n 20 "$RESULTS_DIR/sail_server.log"
        exit 1
    fi
    if python3 -c "import socket; s = socket.socket(); s.connect(('$IP', $PORT))" >/dev/null 2>&1; then
        echo "[gpu] server is UP on $IP:$PORT"
        break
    fi
    sleep 1
done

# 2. Start vLLM Server in background
VLLM_MODEL=$("$PY" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['models']['generator']['name'])")
export VLLM_MODEL
source scripts/start_vllm_server.sh
# Update trap now that VLLM_PID is set
trap "echo '[gpu] stopping servers...'; kill $SAIL_PID $VLLM_PID 2>/dev/null || true" EXIT

run_bench() {
  local workload="$1" execution="$2" extra_args="${3:-}"
  local base_rid="${workload}_${execution}$( [ -n "$extra_args" ] && echo "_$(echo "$extra_args" | tr -d ' -')" )"
  
  for iter in $(seq 1 "$ITERATIONS"); do
    local rid="${base_rid}_s${iter}"
    echo "[gpu] ▶ $rid ($iter/$ITERATIONS)"
    
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
        --run-id "$rid" \
        $extra_args
  done
}

# W0: depth 1, 2, 3
for depth in 1 2 3; do
  for cfg in A B C D; do
    run_bench "w0" "$cfg" "--depth $depth"
  done
done

# W1 / W2 / W3 / W4
for wl in w1 w2 w3 w4; do
  for cfg in A B C D; do
    run_bench "$wl" "$cfg"
  done
done

echo "[gpu] complete ($ITERATIONS samples each). results -> $RESULTS_DIR"
