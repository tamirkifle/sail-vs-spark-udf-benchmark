"""Spark session builder for configs A and B.

A local SparkSession is created with sensible defaults for benchmarking on a
laptop. Key knobs:
  - ``spark.python.worker.reuse=true`` to amortise the model load cost.
  - ``spark.sql.execution.arrow.pyspark.enabled=true`` for Pandas UDF (Config B).
  - Small shuffle partition count because our inputs are tiny.
  - Pin PYSPARK_PYTHON to the current interpreter so workers match driver
    version (avoids PYTHON_VERSION_MISMATCH when a venv Python differs from
    the system Python).
"""

from __future__ import annotations

import os
import sys
from typing import Any


def build_spark_session(cfg: dict[str, Any]) -> Any:
    """Build a local Spark session sized for this benchmark.

    We don't reuse a global session — each run of the CLI starts fresh so the
    MODEL_LOAD cost is cleanly attributed to this run.
    """
    # Pin worker Python to whatever's running the driver, unless the caller
    # has set it explicitly (e.g. in SLURM).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from pyspark.sql import SparkSession

    npart = int(cfg.get("hardware", {}).get("num_partitions", 2))
    builder = (
        SparkSession.builder
        .appName("sail_vs_spark-benchmark")
        .master(f"local[{max(1, npart)}]")
        .config("spark.sql.shuffle.partitions", str(max(1, npart)))
        .config("spark.python.worker.reuse", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        # Tight-ish memory so spills surface in disk I/O metrics
        .config("spark.driver.memory", "2g")
    )
    return builder.getOrCreate()
