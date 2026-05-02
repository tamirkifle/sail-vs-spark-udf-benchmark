#!/usr/bin/env bash
# Unified benchmark runner for mock, CPU/macOS, CPU vLLM, and GPU/Linux runs.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_benchmark.sh [--mode mock|cpu|cpu_real|gpu] [--config path] [--venv path]

Common overrides:
  MODELS_DIR=...         HF/Transformers cache. Default: ./models
  RESULTS_DIR=...        Output root. Default: results/<mode>/<timestamp>
  ITERATIONS=...         Samples per workload/config cell.
  WORKLOADS="w0 w1"      Workloads to run. Default: w0 w1 w2 w3 w4
  EXECUTIONS="A B"       Execution configs. Default comes from config.
  W0_DEPTHS="1 2 3"      W0 depths. Default comes from config.
  FORCE_SYNTHETIC=1      Force synthetic prompts. Default for mock mode.
  SKIP_DATASET_PREP=1    Reuse existing dataset parquet from the config out_dir.
  START_SAIL=0|1         Override auto Sail server startup for C/D.
  START_VLLM=0|1         Override auto vLLM startup for gpu mode.
USAGE
}

MODE="${MODE:-mock}"
CONFIG="${CONFIG:-}"
VENV="${VENV:-.venv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
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
  mock|cpu|cpu_real|gpu) ;;
  *)
    echo "invalid mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"

case "$MODE" in
  mock) DEFAULT_CONFIG="config/mock.yaml" ;;
  cpu) DEFAULT_CONFIG="config/cpu.yaml" ;;
  cpu_real) DEFAULT_CONFIG="config/cpu_real.yaml" ;;
  gpu) DEFAULT_CONFIG="config/gpu_h200.yaml" ;;
esac
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[run] missing $VENV/bin/python. Run: scripts/setup_env.sh --mode $MODE --venv $VENV" >&2
  exit 1
fi
PY="$VENV/bin/python"

yaml_expr() {
  local expr="$1"
  local fallback="$2"
  "$PY" - "$CONFIG" "$expr" "$fallback" <<'PY' 2>/dev/null || printf '%s\n' "$fallback"
import sys, yaml
path, expr, fallback = sys.argv[1:]
with open(path) as fh:
    cfg = yaml.safe_load(fh) or {}
try:
    value = eval(expr, {"cfg": cfg})
except Exception:
    value = fallback
if isinstance(value, (list, tuple)):
    print(" ".join(str(v) for v in value))
elif value is None:
    print(fallback)
else:
    print(value)
PY
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_RESULTS_DIR="results/$MODE/$TIMESTAMP"
RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
ITERATIONS="${ITERATIONS:-$(yaml_expr "cfg.get('runner', {}).get('iterations', 1)" "1")}"
WORKLOADS="${WORKLOADS:-w0 w1 w2 w3 w4}"
EXECUTIONS="${EXECUTIONS:-$(yaml_expr "cfg.get('execution', {}).get('configs', ['A', 'B'])" "A B")}"
W0_DEPTHS="${W0_DEPTHS:-$(yaml_expr "cfg.get('workloads', {}).get('w0_chained', {}).get('depths', [1])" "1")}"
FORCE_SYNTHETIC="${FORCE_SYNTHETIC:-$([[ "$MODE" == "mock" ]] && echo 1 || echo 0)}"
START_VLLM="${START_VLLM:-$([[ "$MODE" == "gpu" ]] && echo 1 || echo 0)}"

MODELS_DIR="${MODELS_DIR:-$REPO_DIR/models}"
export HF_HOME="${HF_HOME:-$MODELS_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$MODELS_DIR/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$MODELS_DIR}"
mkdir -p "$MODELS_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$RESULTS_DIR"

if [[ "${SKIP_DATASET_PREP:-0}" == "1" ]]; then
  echo "[run] skipping dataset prep: config=$CONFIG"
  DATASET_PARQUET="$(yaml_expr "cfg.get('dataset', {}).get('out_dir', 'data')" "data")/prompts.parquet"
  if [[ ! -e "$DATASET_PARQUET" ]]; then
    echo "[run] missing pre-staged dataset: $DATASET_PARQUET" >&2
    echo "[run] run scripts/prep_dataset.py on a node with internet, or unset SKIP_DATASET_PREP" >&2
    exit 1
  fi
else
  prep_args=(--config "$CONFIG")
  if [[ "$FORCE_SYNTHETIC" == "1" ]]; then
    prep_args+=(--force-synthetic)
  fi
  echo "[run] preparing dataset: config=$CONFIG synthetic=$FORCE_SYNTHETIC"
  "$PY" scripts/prep_dataset.py "${prep_args[@]}"
fi

needs_sail=0
for execution in $EXECUTIONS; do
  if [[ "$execution" == "C" || "$execution" == "D" ]]; then
    needs_sail=1
  fi
done
START_SAIL="${START_SAIL:-$needs_sail}"

SAIL_PID=""
VLLM_PID="${VLLM_PID:-}"
cleanup() {
  if [[ -n "${SAIL_PID:-}" ]]; then
    echo "[run] stopping Sail server (PID $SAIL_PID)"
    kill "$SAIL_PID" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID:-}" ]]; then
    echo "[run] stopping vLLM server (PID $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$START_SAIL" == "1" ]]; then
  if [[ -x "$VENV/bin/sail" ]]; then
    SAIL_CMD=("$VENV/bin/sail")
  else
    SAIL_CMD=("$PY" "-m" "pysail.cli")
  fi
  BASE_PORT=50000
  JOB_OFFSET=$(( ${SLURM_JOB_ID:-$$} % 1000 ))
  PORT="${SAIL_PORT:-$(( BASE_PORT + JOB_OFFSET ))}"
  IP="${SAIL_IP:-127.0.0.1}"

  echo "[run] starting Sail server on $IP:$PORT"
  "${SAIL_CMD[@]}" spark server --ip "$IP" --port "$PORT" > "$RESULTS_DIR/sail_server.log" 2>&1 &
  SAIL_PID=$!

  for _ in {1..60}; do
    if ! kill -0 "$SAIL_PID" 2>/dev/null; then
      echo "[run] Sail server died. Last 30 log lines:" >&2
      tail -n 30 "$RESULTS_DIR/sail_server.log" >&2 || true
      exit 1
    fi
    if "$PY" - "$IP" "$PORT" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(1)
s.connect((sys.argv[1], int(sys.argv[2])))
PY
    then
      echo "[run] Sail server ready"
      break
    fi
    sleep 1
  done
  export SAIL_REMOTE_URL="sc://$IP:$PORT"
fi

if [[ "$START_VLLM" == "1" ]]; then
  _vllm() { yaml_expr "cfg.get('vllm', {}).get('$1', '$2')" "$2"; }
  export VLLM_MODEL="${VLLM_MODEL:-$(yaml_expr "cfg['models']['generator']['name']" "")}"
  export VLLM_HOST="${VLLM_HOST:-$(_vllm host "127.0.0.1")}"
  export VLLM_PORT="${VLLM_PORT:-$(_vllm port "8000")}"
  export VLLM_DEVICE="${VLLM_DEVICE:-$(_vllm device "gpu")}"
  export VLLM_DTYPE="${VLLM_DTYPE:-$(_vllm dtype "")}"
  export VLLM_TP="${VLLM_TP:-$(_vllm tensor_parallel_size "1")}"
  export VLLM_GPU_MEM="${VLLM_GPU_MEM:-$(_vllm gpu_memory_utilization "0.90")}"
  export VLLM_MAX_LEN="${VLLM_MAX_LEN:-$(_vllm max_model_len "8192")}"
  export VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-$(_vllm quantization "fp8")}"
  export VLLM_TOKENIZER="${VLLM_TOKENIZER:-$(_vllm tokenizer "")}"
  export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-$(_vllm cpu_kvcache_space "")}"
  export VLLM_CPU_NUM_OF_RESERVED_CPU="${VLLM_CPU_NUM_OF_RESERVED_CPU:-$(_vllm cpu_num_reserved "")}"
  echo "[run] starting vLLM: device=$VLLM_DEVICE model=$VLLM_MODEL url=http://$VLLM_HOST:$VLLM_PORT"
  source scripts/start_vllm_server.sh
fi

run_cell() {
  local workload="$1"
  local execution="$2"
  local depth="${3:-}"
  local run_id="${workload}_${execution}"
  local extra=()
  if [[ -n "$depth" ]]; then
    run_id="${run_id}_depth${depth}"
    extra+=(--depth "$depth")
  fi

  if [[ "$execution" == "C" || "$execution" == "D" ]]; then
    export SPARK_REMOTE="${SAIL_REMOTE_URL:-$(yaml_expr "cfg.get('runner', {}).get('sail_remote_url', '')" "")}"
  else
    unset SPARK_REMOTE
  fi

  echo "[run] cell=$run_id samples=$ITERATIONS"
  "$PY" -m sail_vs_spark.runner.cli \
    --config "$CONFIG" \
    --workload "$workload" \
    --execution "$execution" \
    --results-dir "$RESULTS_DIR" \
    --run-id "$run_id" \
    --samples "$ITERATIONS" \
    "${extra[@]}"
}

echo "[run] mode=$MODE executions=[$EXECUTIONS] workloads=[$WORKLOADS] results=$RESULTS_DIR"
for workload in $WORKLOADS; do
  if [[ "$workload" == "w0" ]]; then
    for depth in $W0_DEPTHS; do
      for execution in $EXECUTIONS; do
        run_cell "$workload" "$execution" "$depth"
      done
    done
  else
    for execution in $EXECUTIONS; do
      run_cell "$workload" "$execution"
    done
  fi
done

"$PY" analysis/aggregate_results.py --results_dir "$RESULTS_DIR"
echo "[run] complete"
echo "[run] results: $RESULTS_DIR"
echo "[run] report:  $RESULTS_DIR/report/aggregate.html"
