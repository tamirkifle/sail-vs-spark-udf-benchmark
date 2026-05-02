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
from pathlib import Path
from typing import Any


def build_spark_session(cfg: dict[str, Any]) -> Any:
    """Build a local Spark session sized for this benchmark.

    We don't reuse a global session — each run of the CLI starts fresh so the
    MODEL_LOAD cost is cleanly attributed to this run.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["PYTHONNOUSERSITE"] = "1"

    # CUDA context isolation: Spark local mode uses os.fork() to create Python
    # workers. If the driver process has ever touched CUDA (even via import),
    # forked children inherit a locked CUDA context and crash with
    # "CUDA unknown error". Mask the GPU from the driver; workers get the real
    # device ID via spark.executorEnv so they initialize a fresh CUDA context.
    real_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

        # Ensure workers use the venv
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .config("spark.executorEnv.PYTHONNOUSERSITE", "1")

        # Forward real GPU IDs to workers (driver sees -1 above)
        .config("spark.executorEnv.CUDA_VISIBLE_DEVICES", real_cuda_devices)

        # Forward HuggingFace cache and offline flags to workers
        .config("spark.executorEnv.HF_HOME", os.environ.get("HF_HOME", ""))
        .config("spark.executorEnv.HF_HUB_CACHE", os.environ.get("HF_HUB_CACHE", ""))
        .config(
            "spark.executorEnv.SENTENCE_TRANSFORMERS_HOME",
            os.environ.get("SENTENCE_TRANSFORMERS_HOME", ""),
        )
        .config("spark.executorEnv.HF_HUB_OFFLINE", "1")
        .config("spark.executorEnv.TRANSFORMERS_OFFLINE", "1")
        .config("spark.executorEnv.HF_DATASETS_OFFLINE", "1")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")

        # Tight-ish memory so spills surface in disk I/O metrics
        .config("spark.driver.memory", "2g")
    )

    return builder.getOrCreate()
