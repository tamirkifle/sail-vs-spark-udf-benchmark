"""Registry for execution backends."""

from __future__ import annotations

from typing import Any

from .backends import SailArrowBackend, SailUdtfBackend, SparkPandasBackend, SparkRowBackend

_BACKENDS = {
    "A": SparkRowBackend("A"),
    "B": SparkPandasBackend("B"),
    "C": SailArrowBackend("C"),
    "D": SailUdtfBackend("D"),
}

SUPPORTED_EXECUTIONS = tuple(_BACKENDS)


def get_backend(execution: str):
    execution = execution.upper()
    try:
        return _BACKENDS[execution]
    except KeyError as exc:
        raise ValueError(f"unknown execution {execution!r}") from exc


def run_workload(
    spark: Any,
    execution: str,
    workload: str,
    parquet_path: str,
    cfg: dict[str, Any],
    output_parquet: str | None,
) -> int:
    return get_backend(execution).run_workload(
        spark=spark,
        workload=workload.lower(),
        parquet_path=parquet_path,
        cfg=cfg,
        output_parquet=output_parquet,
    )
