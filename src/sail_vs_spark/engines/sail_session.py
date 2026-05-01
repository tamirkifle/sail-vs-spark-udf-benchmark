"""Sail session builder for configs C and D.

Sail exposes a Spark Connect-compatible server. The benchmark supports two
invocation modes — both are handled transparently by this builder:

1. ``sail spark run -f script.py``  (ephemeral server, pre-injected session)
   Inside such a script, calling ``SparkSession.getActiveSession()`` returns
   the pre-injected Connect session. We detect this and reuse it.

2. Direct launch: the user runs ``sail spark server --port 50051`` in one
   terminal, then invokes the CLI normally. The CLI builds a Spark Connect
   session pointed at that URL.

NOTE on worker environment:
  spark.executorEnv.* is a local-Spark concept; it is a no-op for Spark
  Connect remote sessions. Sail Python workers inherit the environment of the
  Sail server process at the time it was started. Critical variables
  (HF_HOME, HF_HUB_OFFLINE, CUDA_VISIBLE_DEVICES) must therefore be exported
  before `sail spark server` is launched — which run_all_laptop.sh does.
  As a belt-and-suspenders measure, config C/D process closures also set
  these variables explicitly at worker entry (see configs/config_c_sail_arrow.py).
"""

from __future__ import annotations

import os
from typing import Any

# Variables the Sail Python workers must have. We capture them from the
# current client process (which run_all_laptop.sh has already exported) so
# the config C/D closures can re-inject them explicitly inside the worker.
_WORKER_ENV: dict[str, str] = {
    k: os.environ[k]
    for k in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
              "HF_DATASETS_OFFLINE", "CUDA_VISIBLE_DEVICES", "VLLM_BASE_URL")
    if k in os.environ
}


def build_sail_session(cfg: dict[str, Any]) -> Any:
    from pyspark.sql import SparkSession

    npart = max(1, int(cfg.get("hardware", {}).get("num_partitions", 2)))

    # 1. Check for a pre-injected session (sail spark run)
    existing = SparkSession.getActiveSession()
    if existing is not None:
        existing.conf.set("spark.sql.shuffle.partitions", str(npart))
        return existing

    # 2. Check for SPARK_REMOTE env var (set by sail spark run)
    remote_url = os.environ.get("SPARK_REMOTE")

    # 3. Fallback to config or default
    if not remote_url:
        remote_url = (
            cfg.get("runner", {}).get("sail_remote_url") or "sc://127.0.0.1:50051"
        )

    # Force 127.0.0.1 if localhost is used (avoid IPv6 resolution hangs)
    remote_url = remote_url.replace("localhost", "127.0.0.1")

    # Increase gRPC inbound metadata limit before the channel is created.
    # Default is 16 KB. Sail returns Python tracebacks as gRPC error metadata;
    # a deep FP8/MoE error chain easily exceeds 16 KB, causing gRPC to replace
    # the real error with RESOURCE_EXHAUSTED. That then triggers a secondary
    # ValueError in PySpark's grpc_status.from_call (mismatched status codes),
    # making the root cause completely invisible. 4 MB is generous but harmless.
    try:
        import grpc as _grpc
        _orig_insecure = _grpc.insecure_channel
        _orig_secure   = _grpc.secure_channel

        def _patched_insecure(target, options=None, **kw):
            opts = list(options or [])
            if not any(k == "grpc.max_metadata_size" for k, _ in opts):
                opts.append(("grpc.max_metadata_size", 4 * 1024 * 1024))
            return _orig_insecure(target, options=opts, **kw)

        def _patched_secure(target, credentials, options=None, **kw):
            opts = list(options or [])
            if not any(k == "grpc.max_metadata_size" for k, _ in opts):
                opts.append(("grpc.max_metadata_size", 4 * 1024 * 1024))
            return _orig_secure(target, credentials, options=opts, **kw)

        _grpc.insecure_channel = _patched_insecure
        _grpc.secure_channel   = _patched_secure
        _patched = True
    except ImportError:
        _patched = False

    try:
        session = (
            SparkSession.builder
            .appName("sail_vs_spark-benchmark-sail")
            .remote(remote_url)
            .getOrCreate()
        )
    finally:
        if _patched:
            _grpc.insecure_channel = _orig_insecure
            _grpc.secure_channel   = _orig_secure

    session.conf.set("spark.sql.shuffle.partitions", str(npart))
    return session


def is_running_under_sail() -> bool:
    """Heuristic: returns True if a Spark Connect session is already active.

    Used by the runner to decide whether to launch a Sail server itself.
    """
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession() is not None
    except Exception:
        return False
