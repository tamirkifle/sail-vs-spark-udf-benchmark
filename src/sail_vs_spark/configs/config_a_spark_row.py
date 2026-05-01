"""Config A — Spark Row (cloudpickle) UDF.

This is the classic pickle baseline. Each row crosses JVM → pickle → socket
→ Python → pickle → socket → JVM. For workloads with N_CANDIDATES=4 and
chained depth, serialization cost is paid repeatedly.

Implementation notes
────────────────────
* Every UDF closure captures a reference to the ``cfg`` dict and the
  ``workload_code``. torch is NEVER imported at module level — it's lazily
  imported inside the worker when models load.
* ``BoundaryTimer`` is created per-partition on the worker; its raw phase
  samples are returned alongside the payload output and merged in the driver.
* For W0 we use the fastest possible pure-int UDF to keep the measurement of
  Spark's per-call overhead clean.
"""

from __future__ import annotations

from typing import Any


# ── W0 — chained trivial UDF (depth 1/2/3) ──────────────────────────────────
def run_w0(spark: Any, parquet_path: str, depth: int,
           output_parquet: str | None = None) -> int:
    """Run W0 with Config A. Returns number of output rows."""
    from pyspark.sql.functions import udf
    from pyspark.sql.types import LongType

    @udf(returnType=LongType())
    def stage(x):
        return int(x) + 1

    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    out = df
    for _ in range(depth):
        out = out.withColumn("prompt_id", stage("prompt_id"))
    # Avoid triggering write unless asked — for W0, count is enough.
    if output_parquet:
        out.write.mode("overwrite").parquet(output_parquet)
        return spark.read.parquet(output_parquet).count()
    return out.count()


# ── W1 — Best-of-N via Row UDFs ──────────────────────────────────────────────
def run_w1(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    from pyspark.sql.functions import struct, udf
    from pyspark.sql.types import (FloatType, IntegerType, LongType, StringType,
                                    StructField, StructType)

    schema = StructType([
        StructField("prompt_id", LongType(), False),
        StructField("best_response", StringType(), False),
        StructField("best_reward", FloatType(), False),
        StructField("n_candidates", IntegerType(), False),
    ])
    ncands = int(cfg["workloads"]["w1_best_of_n"]["n_candidates"])
    # Snapshot only the slices of cfg the closure needs, so the pickled
    # blob stays small.
    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }

    @udf(returnType=schema)
    def best_of_n(pid, text):
        # Lazy import: the top-level module cannot import these unconditionally.
        from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
        wl = W1BestOfN(n_candidates=ncands)
        wl.init(closure_cfg)
        pid2, resp, reward, n = wl.apply(int(pid), str(text))
        return (pid2, resp, float(reward), int(n))

    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    out = (
        df
        .withColumn("_r", best_of_n("prompt_id", "prompt_text"))
        .select(
            "_r.prompt_id", "_r.best_response",
            "_r.best_reward", "_r.n_candidates",
        )
    )
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()


# ── W2 — Batched inference via Row UDF (one row at a time, worst case) ──────
def run_w2(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }

    @udf(returnType=StringType())
    def gen_one(text):
        if not hasattr(gen_one, "_timer"):
            from sail_vs_spark.profiling.boundary_timer import BoundaryTimer
            gen_one._timer = BoundaryTimer("config_a", enable_tracing=True)
            gen_one._row_count = 0
            
        with gen_one._timer.measure("UDF_ROW_EXECUTION"):
            from sail_vs_spark.workloads.w2_batched import W2Batched
            wl = W2Batched()
            wl.init(closure_cfg)
            _, resp = wl.apply(0, str(text))
            
        gen_one._row_count += 1
        if gen_one._row_count % 10 == 0:
            import os
            gen_one._timer.save_trace(f"/tmp/sail_traces/trace_{os.getpid()}.jsonl")
        return resp

    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    out = df.withColumn("response", gen_one("prompt_text")) \
            .select("prompt_id", "response")
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()


# ── W4 — Agentic loop via Row UDFs (JVM boundary paid per generate+score call)
def run_w4(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    from pyspark.sql.functions import struct, udf
    from pyspark.sql.types import (FloatType, IntegerType, LongType, StringType,
                                    StructField, StructType)

    schema = StructType([
        StructField("prompt_id", LongType(), False),
        StructField("final_response", StringType(), False),
        StructField("iterations", IntegerType(), False),
        StructField("best_reward", FloatType(), False),
    ])
    w4_cfg = cfg.get("workloads", {}).get("w4_agentic", {})
    max_iter = int(w4_cfg.get("max_iterations", 3))
    threshold = float(w4_cfg.get("reward_threshold", 0.5))
    ncands = int(w4_cfg.get("n_candidates", 2))
    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }

    @udf(returnType=schema)
    def agentic(pid, text):
        from sail_vs_spark.workloads.w4_agentic import W4Agentic
        wl = W4Agentic(max_iterations=max_iter, reward_threshold=threshold,
                       n_candidates=ncands)
        wl.init(closure_cfg)
        pid2, resp, iters, reward = wl.apply(int(pid), str(text))
        return (pid2, resp, iters, float(reward))

    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    out = (
        df
        .withColumn("_r", agentic("prompt_id", "prompt_text"))
        .select("_r.prompt_id", "_r.final_response",
                "_r.iterations", "_r.best_reward")
    )
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()


# ── W3 — Embedding + similarity via Row UDF ─────────────────────────────────
def run_w3(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    from pyspark.sql.functions import udf
    from pyspark.sql.types import (FloatType, IntegerType, LongType,
                                    StructField, StructType)

    schema = StructType([
        StructField("prompt_id", LongType(), False),
        StructField("best_query_idx", IntegerType(), False),
        StructField("best_similarity", FloatType(), False),
    ])
    n_queries = int(cfg["workloads"]["w3_embedding"]["n_queries"])
    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }


    @udf(returnType=schema)
    def embed_sim(pid, text):
        from sail_vs_spark.workloads.w3_embedding import W3Embedding
        wl = W3Embedding(n_queries=n_queries)
        wl.init(closure_cfg)
        return wl.apply(int(pid), str(text))

    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    out = (
        df
        .withColumn("_r", embed_sim("prompt_id", "prompt_text"))
        .select("_r.prompt_id", "_r.best_query_idx", "_r.best_similarity")
    )
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()
