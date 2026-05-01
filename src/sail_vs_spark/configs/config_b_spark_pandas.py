"""Compatibility wrappers for Config B (Spark Pandas UDF)."""

from typing import Any, Optional

from sail_vs_spark.execution.registry import get_backend

_BACKEND = get_backend("B")


def _cfg_for_depth(depth: int) -> dict[str, Any]:
    return {"workloads": {"w0_chained": {"depth": depth}}}


# ── W0 ──────────────────────────────────────────────────────────────────────
def run_w0(spark: Any, parquet_path: str, depth: int,
           output_parquet: Optional[str] = None) -> int:
    return _BACKEND.run_w0(spark, parquet_path, _cfg_for_depth(depth), output_parquet)


# ── W1 — Best-of-N via a struct-returning pandas_udf ────────────────────────
def run_w1(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    return _BACKEND.run_w1(spark, parquet_path, cfg, output_parquet)


# ── W2 — Batched inference via Scalar pandas_udf ────────────────────────────
def run_w2(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    return _BACKEND.run_w2(spark, parquet_path, cfg, output_parquet)


# ── W4 — Agentic loop via pandas_udf (JVM boundary paid per batch)
def run_w4(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    return _BACKEND.run_w4(spark, parquet_path, cfg, output_parquet)


# ── W3 — Embedding + similarity via pandas_udf ──────────────────────────────
def run_w3(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    return _BACKEND.run_w3(spark, parquet_path, cfg, output_parquet)
