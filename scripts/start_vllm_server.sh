#!/usr/bin/env bash
# Source this file (don't execute it) from run_all_gpu.sh so VLLM_BASE_URL
# and VLLM_PID stay in the calling shell's scope.
#
# Usage in run_all_gpu.sh:
#   export VLLM_MODEL="Qwen/Qwen3.5-122B-A10B-FP8"
#   source scripts/start_vllm_server.sh

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"

echo "[vllm] starting server: ${VLLM_MODEL} on ${VLLM_BASE_URL}"
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${VLLM_TP:-1}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM:-0.90}" \
    --max-model-len "${VLLM_MAX_LEN:-8192}" \
    --quantization "${VLLM_QUANTIZATION:-fp8}" \
    --enforce-eager \
    > "$RESULTS_DIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
export VLLM_PID

echo "[vllm] server PID ${VLLM_PID}, waiting for readiness..."
for i in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[vllm] ERROR: server died. Last 20 lines of log:"
        tail -n 20 "$RESULTS_DIR/vllm_server.log" || true
        exit 1
    fi
    if "$PY" -c "
import urllib.request, sys
try:
    urllib.request.urlopen('${VLLM_BASE_URL}/health', timeout=2)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "[vllm] server is READY on ${VLLM_BASE_URL}"
        break
    fi
    sleep 2
done
