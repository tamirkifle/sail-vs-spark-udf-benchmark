#!/usr/bin/env bash
# Run the full 4×4 benchmark matrix on GPU using a persistent Sail server.

set -euo pipefail

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"

export HF_HOME="$REPO_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

VENV="${VENV:-.venv_gpu}"
PY="$VENV/bin/python"

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

# Read iterations and W0 depths from the yaml config (env var overrides yaml).
# This lets smoke configs set iterations=1 and depths=[1] without script changes.
_CFG_ITERS=$("$PY" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('runner',{}).get('iterations', 3))" 2>/dev/null || echo 3)
ITERATIONS="${ITERATIONS:-$_CFG_ITERS}"
W0_DEPTHS=$("$PY" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(' '.join(str(d) for d in c.get('workloads',{}).get('w0_chained',{}).get('depths',[1,2,3])))" 2>/dev/null || echo "1 2 3")

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
# Read all vllm settings from the yaml config so smoke vs full configs differ.
_vllm() { "$PY" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('vllm',{}).get('$1','$2'))" 2>/dev/null || echo "$2"; }
export VLLM_MODEL=$("$PY" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['models']['generator']['name'])")
export VLLM_HOST=$(_vllm host "127.0.0.1")
export VLLM_PORT=$(_vllm port "8000")
export VLLM_TP=$(_vllm tensor_parallel_size "1")
export VLLM_GPU_MEM=$(_vllm gpu_memory_utilization "0.90")
export VLLM_MAX_LEN=$(_vllm max_model_len "8192")
export VLLM_QUANTIZATION=$(_vllm quantization "fp8")
export VLLM_TOKENIZER=$(_vllm tokenizer "")
echo "[gpu] vLLM config: model=$VLLM_MODEL host=$VLLM_HOST:$VLLM_PORT tp=$VLLM_TP mem=$VLLM_GPU_MEM max_len=$VLLM_MAX_LEN quant=$VLLM_QUANTIZATION tokenizer=${VLLM_TOKENIZER:-<model>}"
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



# W1 / W2 / W3 / W4
for wl in w1 w2 w3 w4; do
  for cfg in C D A B; do
    run_bench "$wl" "$cfg"
  done
done

# W0: depths read from yaml (smoke: [1], full: [1, 2, 3])
echo "[gpu] W0 depths: $W0_DEPTHS  iterations: $ITERATIONS"
for depth in $W0_DEPTHS; do
  for cfg in A B C D; do
    run_bench "w0" "$cfg" "--depth $depth"
  done
done


echo "[gpu] complete ($ITERATIONS samples each). results -> $RESULTS_DIR"
