"""Run-manifest helper.

Each (workload, config) invocation writes a small JSON manifest with metadata
and file pointers. The aggregator reads these plus the per-run boundary /
stats JSONs to build the comparison tables and plots.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import socket
from pathlib import Path
from typing import Any


def make_manifest(
    run_id: str,
    workload_code: str,
    execution_config: str,
    *,
    depth: int | None = None,
    cfg: dict[str, Any] | None = None,
    output_parquet: str | None = None,
    wall_clock_sec: float | None = None,
    output_rows: int | None = None,
    boundary_json: str | None = None,
    stats_json: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workload": workload_code,
        "execution": execution_config,
        "depth": depth,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "profile": (cfg or {}).get("profile"),
        "device": (cfg or {}).get("hardware", {}).get("device"),
        "num_partitions": (cfg or {}).get("hardware", {}).get("num_partitions"),
        "dataset_rows": (cfg or {}).get("dataset", {}).get("n_rows"),
        "output_parquet": output_parquet,
        "output_rows": output_rows,
        "wall_clock_sec": wall_clock_sec,
        "boundary_json": boundary_json,
        "stats_json": stats_json,
    }


def save_manifest(manifest: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(manifest, fh, indent=2)
