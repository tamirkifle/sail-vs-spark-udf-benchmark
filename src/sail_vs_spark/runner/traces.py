"""Trace artifact helpers."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

TRACE_DIR = Path("/tmp/sail_traces")


def clear_trace_dir() -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    for path in glob.glob(str(TRACE_DIR / "*.jsonl")):
        try:
            os.remove(path)
        except OSError:
            pass


def collect_trace_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in glob.glob(str(TRACE_DIR / "*.jsonl")):
        try:
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        events.append(json.loads(line))
            os.remove(path)
        except Exception:
            pass
    return events


def save_trace_artifact(events: list[dict[str, Any]], path: str | Path) -> str | None:
    if not events:
        return None
    out_path = Path(path)
    with open(out_path, "w") as fh:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, fh, indent=2)
    return str(out_path)
