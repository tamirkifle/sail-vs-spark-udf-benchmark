"""Shared helpers for execution backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sail_vs_spark.workloads.registry import build_workload

_TYPE_TO_SQL = {
    "int32": "int",
    "int64": "long",
    "float32": "float",
    "string": "string",
}


def compact_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep closure payloads small while preserving workload config."""
    return {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
        "workloads": cfg.get("workloads", {}),
    }


def read_prompts_df(spark: Any, parquet_path: str) -> Any:
    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    return spark.read.parquet(parquet_path).repartition(n_part)


def write_output(df: Any, output_parquet: str | None) -> int:
    if output_parquet:
        df.write.mode("overwrite").parquet(output_parquet)
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return df.count()

        out_path = Path(output_parquet)
        files = sorted(out_path.glob("*.parquet"))
        if not files:
            return 0
        return sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    return df.count()


def build_initialized_workload(workload: str, cfg: dict[str, Any]):
    wl = build_workload(workload, cfg)
    wl.init(cfg)
    return wl


def workload_columns(workload: str, cfg: dict[str, Any]) -> list[tuple[str, str]]:
    return list(build_workload(workload, cfg).result.output_columns)


def workload_schema_sql(workload: str, cfg: dict[str, Any]) -> str:
    cols = workload_columns(workload, cfg)
    return ", ".join(f"{name} {_TYPE_TO_SQL[dtype]}" for name, dtype in cols)


def workload_struct_type(workload: str, cfg: dict[str, Any]):
    from pyspark.sql.types import (
        FloatType,
        IntegerType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    type_map = {
        "int32": IntegerType(),
        "int64": LongType(),
        "float32": FloatType(),
        "string": StringType(),
    }
    return StructType(
        [StructField(name, type_map[dtype], False) for name, dtype in workload_columns(workload, cfg)]
    )


def select_struct_output(df: Any, column_name: str, workload: str, cfg: dict[str, Any]) -> Any:
    cols = [f"{column_name}.{name}" for name, _ in workload_columns(workload, cfg)]
    return df.select(*cols)
