#!/usr/bin/env bash
# Create/update a local Python environment for benchmark runs.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/setup_env.sh [--mode mock|cpu|cpu_real|gpu|dev] [--venv .venv]

Modes:
  mock  Base benchmark dependencies only. Fastest path; models are mocked.
  cpu   Base deps plus torch/transformers/sentence-transformers.
  cpu_real
        CPU deps plus vLLM CPU wheel on Linux x86_64.
  gpu   CPU deps plus vLLM/accelerate/bitsandbytes for CUDA hosts.
  dev   Base deps for tests and development.
USAGE
}

MODE="${MODE:-mock}"
VENV="${VENV:-.venv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
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
  mock|cpu|cpu_real|gpu|dev) ;;
  *)
    echo "invalid mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_DIR"
if [[ "$VENV" != /* ]]; then
  VENV="$REPO_DIR/$VENV"
fi

MODELS_DIR="${MODELS_DIR:-$REPO_DIR/models}"
mkdir -p "$MODELS_DIR"
export HF_HOME="${HF_HOME:-$MODELS_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$MODELS_DIR/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$MODELS_DIR}"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install -e .
"$PY" -m pip install -r requirements.txt

if [[ "$MODE" == "cpu_real" ]]; then
  "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  "$PY" -m pip install transformers sentence-transformers accelerate
elif [[ "$MODE" == "cpu" || "$MODE" == "gpu" ]]; then
  "$PY" -m pip install torch transformers sentence-transformers accelerate
fi

if [[ "$MODE" == "cpu_real" ]]; then
  if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
    GLIBC_VERSION="$("$PY" - <<'PY'
import platform

print(platform.libc_ver()[1] or "0.0")
PY
)"
    GLIBC_OK="$("$PY" - "$GLIBC_VERSION" <<'PY'
import sys

def parts(version: str) -> tuple[int, int]:
    major, _, rest = version.partition(".")
    minor = rest.split(".", 1)[0] if rest else "0"
    return int(major or 0), int(minor or 0)

print("1" if parts(sys.argv[1]) >= (2, 35) else "0")
PY
)"
    VLLM_VERSION="${VLLM_VERSION:-$("$PY" - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("https://api.github.com/repos/vllm-project/vllm/releases/latest", timeout=15) as response:
    tag = json.load(response)["tag_name"]
print(tag.removeprefix("v"))
PY
)}"
    if [[ "${VLLM_CPU_INSTALL:-wheel}" == "source" || "$GLIBC_OK" != "1" ]]; then
      if [[ "${VLLM_CPU_INSTALL:-wheel}" != "source" ]]; then
        cat >&2 <<MSG
[setup] vLLM CPU wheel requires manylinux_2_35/glibc >= 2.35.
[setup] This host reports glibc $GLIBC_VERSION, so pip will reject the wheel.
[setup] Re-run with VLLM_CPU_INSTALL=source to build vLLM on this host.
MSG
        exit 1
      fi
      VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-$REPO_DIR/third_party/vllm_cpu}"
      if [[ ! -d "$VLLM_SOURCE_DIR/.git" ]]; then
        git clone --branch "v${VLLM_VERSION}" --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_SOURCE_DIR"
      fi
      cd "$VLLM_SOURCE_DIR"
      "$PY" -m pip install -v -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
      "$PY" -m pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
      CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON VLLM_TARGET_DEVICE=cpu "$PY" -m pip install . --no-build-isolation
      cd "$REPO_DIR"
    else
      "$PY" -m pip install \
        "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl" \
        --extra-index-url https://download.pytorch.org/whl/cpu
    fi
  else
    cat <<'MSG'
[setup] cpu_real needs a CPU-capable vLLM install.
[setup] Pre-built CPU wheels are available for Linux x86_64. Apple Silicon
[setup] CPU support is experimental and currently requires a source build.
MSG
  fi
fi

if [[ "$MODE" == "gpu" ]]; then
  "$PY" -m pip install vllm bitsandbytes
fi

echo "[setup] ready: mode=$MODE venv=$VENV python=$PY"
