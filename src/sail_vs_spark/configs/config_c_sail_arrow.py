"""Compatibility wrappers for Config C (Sail mapInArrow)."""

from __future__ import annotations
from typing import Any

from sail_vs_spark.execution.registry import get_backend

_BACKEND = get_backend("C")


def _cfg_for_depth(depth: int) -> dict[str, Any]:
    return {"workloads": {"w0_chained": {"depth": depth}}}

# ── W0: Chained Trivial Compute ─────────────────────────────────────────────
def run_w0(spark: Any, parquet_path: str, depth: int,
           output_parquet: str | None = None) -> int:
    return _BACKEND.run_w0(spark, parquet_path, _cfg_for_depth(depth), output_parquet)

# ── W1: Best-of-N ──────────────────────────────────────────────────────────
def run_w1(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    return _BACKEND.run_w1(spark, parquet_path, cfg, output_parquet)

# ── W2: Batched Generation ──────────────────────────────────────────────────
def run_w2(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    return _BACKEND.run_w2(spark, parquet_path, cfg, output_parquet)

# ── W4: Agentic Loop (entire loop inside one Arrow closure — zero extra crossings)
def run_w4(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    return _BACKEND.run_w4(spark, parquet_path, cfg, output_parquet)

# ── W3: Embedding Similarity ────────────────────────────────────────────────
def run_w3(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    return _BACKEND.run_w3(spark, parquet_path, cfg, output_parquet)
