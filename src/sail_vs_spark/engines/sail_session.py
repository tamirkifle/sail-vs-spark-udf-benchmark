"""Sail session builder for configs C and D.

Sail exposes a Spark Connect-compatible server. The benchmark supports two
invocation modes — both are handled transparently by this builder:

1. ``sail spark run -f script.py``  (ephemeral server, pre-injected session)
   Inside such a script, calling ``SparkSession.getActiveSession()`` returns
   the pre-injected Connect session. We detect this and reuse it.

2. Direct launch: the user runs ``sail spark server --port 50051`` in one
   terminal, then invokes the CLI normally. The CLI builds a Spark Connect
   session pointed at that URL.
"""

from __future__ import annotations

from typing import Any


def build_sail_session(cfg: dict[str, Any]) -> Any:
    from pyspark.sql import SparkSession

    existing = SparkSession.getActiveSession()
    if existing is not None:
        # Case 1: running under ``sail spark run`` — reuse pre-injected session
        return existing

    remote_url = (
        cfg.get("runner", {}).get("sail_remote_url") or "sc://localhost:50051"
    )
    return (
        SparkSession.builder
        .appName("sail_vs_spark-benchmark-sail")
        .remote(remote_url)
        .getOrCreate()
    )


def is_running_under_sail() -> bool:
    """Heuristic: returns True if a Spark Connect session is already active.

    Used by the runner to decide whether to launch a Sail server itself.
    """
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession() is not None
    except Exception:
        return False
