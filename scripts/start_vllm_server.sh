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

# Default to the bundled Qwen3 template. FP8-quantized Qwen3 models are often
# uploaded without copying the chat_template from the base tokenizer_config.json.
# transformers 4.44+ forbids the implicit default, so we must supply one explicitly.
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-${REPO_DIR}/config/qwen3_chat_template.jinja}"

# Some FP8-quantized third-party uploads include broken or missing tokenizer
# files. Set VLLM_TOKENIZER to a donor model (any Qwen3 with tokenizer.json)
# to override the tokenizer without changing the model weights.
# Leave unset to use the model's own tokenizer (default).
VLLM_TOKENIZER="${VLLM_TOKENIZER:-}"

# Kill any stale vLLM from a prior job that didn't clean up on exit.
# Without this, the old server holds the port and the new server's EngineCore
# starts its internal ZMQ on port+1, the health-check hits the OLD server,
# and requests go to the un-patched instance.
echo "[vllm] clearing port ${VLLM_PORT} of any stale server..."
fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3

# Mirror _resolve_model_path() in loaders.py: use local flat dir if present,
# otherwise fall back to the HF repo ID (requires internet or HF cache).
# download_models.py stores weights at $REPO_DIR/models/Org--Model/ (double-dash),
# matching Python's model_id.replace("/", "--").
_LOCAL_MODEL_DIR="${REPO_DIR}/models/$(echo "${VLLM_MODEL}" | sed 's|/|--|g')"
if [ -d "${_LOCAL_MODEL_DIR}" ]; then
    _VLLM_MODEL_PATH="${_LOCAL_MODEL_DIR}"
    echo "[vllm] resolved local model dir: ${_VLLM_MODEL_PATH}"
else
    _VLLM_MODEL_PATH="${VLLM_MODEL}"
fi

# Ensure the venv bin is in PATH so FlashInfer's JIT subprocess can find ninja.
export PATH="${REPO_DIR}/.venv_gpu/bin:${PATH}"

echo "[vllm] starting server: ${VLLM_MODEL} on ${VLLM_BASE_URL}"
_TOKENIZER_ARG=()
if [ -n "${VLLM_TOKENIZER}" ]; then
    echo "[vllm] using external tokenizer: ${VLLM_TOKENIZER}"
    _TOKENIZER_ARG=(--tokenizer "${VLLM_TOKENIZER}")
fi
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "${_VLLM_MODEL_PATH}" \
    --served-model-name "${VLLM_MODEL}" \
    "${_TOKENIZER_ARG[@]}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${VLLM_TP:-1}" \
    --gpu-memory-utilization "${VLLM_GPU_MEM:-0.90}" \
    --max-model-len "${VLLM_MAX_LEN:-8192}" \
    --quantization "${VLLM_QUANTIZATION:-fp8}" \
    --chat-template "${VLLM_CHAT_TEMPLATE}" \
    --limit-mm-per-prompt '{"image": 0}' \
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
