"""Config C — Sail mapInArrow (zero-copy Arrow stream)."""

from __future__ import annotations
from typing import Any, Iterator

# Captured at import time (client process). Injected into every process closure
# so Sail Python workers have the correct HF cache / offline / CUDA environment
# regardless of when the Sail server was started.
# spark.executorEnv.* is a no-op for Spark Connect remote sessions — explicit
# os.environ injection inside the closure is the only reliable mechanism.
try:
    from sail_vs_spark.engines.sail_session import _WORKER_ENV as _ENV
except ImportError:
    import os as _os
    _ENV: dict = {k: _os.environ[k] for k in (
        "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE", "CUDA_VISIBLE_DEVICES",
    ) if k in _os.environ}

# ── W0: Chained Trivial Compute ─────────────────────────────────────────────
def run_w0(spark: Any, parquet_path: str, depth: int,
           output_parquet: str | None = None) -> int:
    import pyarrow as pa
    
    # Each stage performs exactly ONE increment on the 'value' column.
    def stage(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        for batch in batch_iter:
            # We only yield the data from THIS batch. 
            # If the input batch has 100 rows, we yield 100 rows.
            ids = batch.column("prompt_id")
            vals = batch.column("value")
            
            try:
                import pyarrow.compute as pc
                bumped = pc.add(vals, 1)
            except Exception:
                bumped = pa.array([int(x) + 1 for x in vals.to_pylist()], type=pa.int64())
            
            yield pa.RecordBatch.from_arrays([ids, bumped], names=["prompt_id", "value"])

    # Initial state: value = prompt_id
    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part).selectExpr("prompt_id", "prompt_id as value")
    
    # Apply the transformation 'depth' times to match Spark's orchestration.
    out = df
    schema = "prompt_id long, value long"
    for _ in range(depth):
        out = out.mapInArrow(stage, schema)

    if output_parquet:
        out.write.mode("overwrite").parquet(output_parquet)
        return spark.read.parquet(output_parquet).count()
    return out.count()

# ── W1: Best-of-N ──────────────────────────────────────────────────────────
def run_w1(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    import pyarrow as pa
    ncands = int(cfg["workloads"]["w1_best_of_n"]["n_candidates"])
    closure_cfg = {"models": cfg.get("models", {}), "hardware": cfg.get("hardware", {})}

    def process(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        import os
        os.environ.update(_ENV)
        from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
        wl = W1BestOfN(n_candidates=ncands)
        wl.init(closure_cfg)
        for batch in batch_iter:
            ids = batch.column("prompt_id").to_pylist()
            texts = batch.column("prompt_text").to_pylist()
            out = wl.apply_batch(ids, texts)
            yield pa.RecordBatch.from_arrays([
                pa.array(out["prompt_id"], type=pa.int64()),
                pa.array(out["best_response"], type=pa.string()),
                pa.array(out["best_reward"], type=pa.float32()),
                pa.array(out["n_candidates"], type=pa.int32()),
            ], names=["prompt_id", "best_response", "best_reward", "n_candidates"])

    schema = "prompt_id long, best_response string, best_reward float, n_candidates int"
    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    out = df.mapInArrow(process, schema)
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()

# ── W2: Batched Generation ──────────────────────────────────────────────────
def run_w2(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    import pyarrow as pa
    closure_cfg = {"models": cfg.get("models", {}), "hardware": cfg.get("hardware", {})}

    def process(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        import os
        os.environ.update(_ENV)
        from sail_vs_spark.workloads.w2_batched import W2Batched
        wl = W2Batched()
        wl.init(closure_cfg)
        for batch in batch_iter:
            ids = batch.column("prompt_id").to_pylist()
            texts = batch.column("prompt_text").to_pylist()
            out = wl.apply_batch(ids, texts)
            yield pa.RecordBatch.from_arrays([
                pa.array(out["prompt_id"], type=pa.int64()),
                pa.array(out["response"], type=pa.string()),
            ], names=["prompt_id", "response"])

    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    out = df.mapInArrow(process, "prompt_id long, response string")
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()

# ── W4: Agentic Loop (entire loop inside one Arrow closure — zero extra crossings)
def run_w4(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    import pyarrow as pa
    w4_cfg = cfg.get("workloads", {}).get("w4_agentic", {})
    max_iter = int(w4_cfg.get("max_iterations", 3))
    threshold = float(w4_cfg.get("reward_threshold", 0.5))
    ncands = int(w4_cfg.get("n_candidates", 2))
    closure_cfg = {"models": cfg.get("models", {}), "hardware": cfg.get("hardware", {})}

    def process(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        import os
        os.environ.update(_ENV)
        from sail_vs_spark.workloads.w4_agentic import W4Agentic
        wl = W4Agentic(max_iterations=max_iter, reward_threshold=threshold,
                       n_candidates=ncands)
        wl.init(closure_cfg)
        for batch in batch_iter:
            ids = batch.column("prompt_id").to_pylist()
            texts = batch.column("prompt_text").to_pylist()
            out = wl.apply_batch(ids, texts)
            yield pa.RecordBatch.from_arrays([
                pa.array(out["prompt_id"], type=pa.int64()),
                pa.array(out["final_response"], type=pa.string()),
                pa.array(out["iterations"], type=pa.int32()),
                pa.array(out["best_reward"], type=pa.float32()),
            ], names=["prompt_id", "final_response", "iterations", "best_reward"])

    schema = "prompt_id long, final_response string, iterations int, best_reward float"
    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    out = df.mapInArrow(process, schema)
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()

# ── W3: Embedding Similarity ────────────────────────────────────────────────
def run_w3(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    import pyarrow as pa
    n_queries = int(cfg["workloads"]["w3_embedding"]["n_queries"])
    closure_cfg = {"models": cfg.get("models", {}), "hardware": cfg.get("hardware", {})}

    def process(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        import os
        os.environ.update(_ENV)
        from sail_vs_spark.workloads.w3_embedding import W3Embedding
        from sail_vs_spark.profiling.boundary_timer import BoundaryTimer
        timer = BoundaryTimer("config_c", enable_tracing=True)
        wl = W3Embedding(n_queries=n_queries)
        wl.init(closure_cfg)
        for batch in batch_iter:
            with timer.measure("UDF_BATCH_EXECUTION"):
                ids = batch.column("prompt_id").to_pylist()
                texts = batch.column("prompt_text").to_pylist()
                out_wl = wl.apply_batch(ids, texts)
                out = pa.RecordBatch.from_arrays([
                    pa.array(out_wl["prompt_id"], type=pa.int64()),
                    pa.array(out_wl["best_query_idx"], type=pa.int32()),
                    pa.array(out_wl["best_similarity"], type=pa.float32()),
                ], names=["prompt_id", "best_query_idx", "best_similarity"])
            timer.save_trace(f"/tmp/sail_traces/trace_{os.getpid()}.jsonl")
            yield out

    schema = "prompt_id long, best_query_idx int, best_similarity float"
    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    out = df.mapInArrow(process, schema)
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()
