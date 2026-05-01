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


def get_setup_description(execution_config: str) -> str:
    descriptions = {
        "A": "Spark (Row/Pickle): Standard row-by-row PySpark UDF",
        "B": "Spark (Pandas/Arrow): Vectorized Pandas UDF via socket",
        "C": "Sail (Zero-Copy): Native Rust engine sharing memory with Python",
        "D": "Sail (SQL-Native): Direct UDTF execution inside Rust engine"
    }
    return descriptions.get(execution_config, f"Unknown config: {execution_config}")


def _get_hardware_details(requested_device: str | None) -> dict[str, Any]:
    details = {}
    import os
    details["cpu_cores"] = os.cpu_count()
    try:
        import psutil
        details["total_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    # Resolve device exactly like loaders do
    resolved = "cpu"
    if requested_device != "auto" and requested_device is not None:
        resolved = requested_device
    else:
        if os.path.exists("/dev/nvidia0"):
            resolved = "cuda"
        else:
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    resolved = "mps"
            except Exception:
                pass
    
    details["resolved_device"] = resolved
    
    if resolved == "cuda":
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                capture_output=True, text=True
            )
            if r.returncode == 0:
                details["gpu_models"] = r.stdout.strip().split("\n")
        except Exception:
            pass
    elif resolved == "mps":
        import platform
        details["gpu_models"] = [platform.processor()]
        
    return details


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
    trace_json: str | None = None,
) -> dict[str, Any]:
    req_device = (cfg or {}).get("hardware", {}).get("device")
    return {
        "run_id": run_id,
        "workload": workload_code,
        "execution": execution_config,
        "setup_description": get_setup_description(execution_config),
        "depth": depth,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware_details": _get_hardware_details(req_device),
        "profile": (cfg or {}).get("profile"),
        "device": req_device,
        "num_partitions": (cfg or {}).get("hardware", {}).get("num_partitions"),
        "dataset_rows": (cfg or {}).get("dataset", {}).get("n_rows"),
        "models": (cfg or {}).get("models", {}),
        "workload_config": (cfg or {}).get("workloads", {}),
        "output_parquet": output_parquet,
        "output_rows": output_rows,
        "wall_clock_sec": wall_clock_sec,
        "boundary_json": boundary_json,
        "stats_json": stats_json,
        "trace_json": trace_json,
    }


def save_manifest(manifest: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(manifest, fh, indent=2)
