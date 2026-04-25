"""Config C — Sail mapInArrow (zero-copy Arrow stream).

Each partition is streamed to Python as an iterator of ``pa.RecordBatch``es
whose backing buffers are *owned by the Rust engine*. Reading a column via
``batch.column("x").to_pylist()`` dereferences that Rust buffer without a
memcpy (Arrow C Data Interface; confirmed in sail-python-udf/conversion.rs).

Expected: ``DATA_TRANSFER_IN avg_ms < 2`` and ``serialization_tax_pct ≈ 0``.
"""

from __future__ import annotations

from typing import Any, Iterator


# ── W0 ──────────────────────────────────────────────────────────────────────
def run_w0(spark: Any, parquet_path: str, depth: int,
           output_parquet: str | None = None) -> int:
    import pyarrow as pa

    d = int(depth)

    def process(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        for batch in batch_iter:
            ids = batch.column("prompt_id")
            # Chained trivial compute done in Arrow-land without Python lists:
            # add scalar ``d`` via compute.add. Falls back to Python if compute
            # kernel unavailable.
            try:
                import pyarrow.compute as pc
                bumped = pc.add(ids, d)
            except Exception:
                bumped = pa.array([int(x) + d for x in ids.to_pylist()],
                                  type=pa.int64())
            yield pa.RecordBatch.from_arrays(
                [bumped, bumped],
                names=["prompt_id", "value"],
            )

    df = spark.read.parquet(parquet_path)
    schema = "prompt_id long, value long"
    out = df.select("prompt_id").mapInArrow(process, schema)
    if output_parquet:
        out.write.mode("overwrite").parquet(output_parquet)
        return spark.read.parquet(output_parquet).count()
    return out.count()


# ── W1 — Best-of-N, fused in a single mapInArrow closure ────────────────────
def run_w1(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    import pyarrow as pa

    ncands = int(cfg["workloads"]["w1_best_of_n"]["n_candidates"])
    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }

    def process(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        # Lazy imports — these are only needed on the worker.
        from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
        wl = W1BestOfN(n_candidates=ncands)
        wl.init(closure_cfg)
        for batch in batch_iter:
            ids = batch.column("prompt_id").to_pylist()
            texts = batch.column("prompt_text").to_pylist()
            out = wl.apply_batch(ids, texts)
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(out["prompt_id"], type=pa.int64()),
                    pa.array(out["best_response"], type=pa.string()),
                    pa.array(out["best_reward"], type=pa.float32()),
                    pa.array(out["n_candidates"], type=pa.int32()),
                ],
                names=["prompt_id", "best_response",
                       "best_reward", "n_candidates"],
            )

    schema = (
        "prompt_id long, best_response string, "
        "best_reward float, n_candidates int"
    )
    df = spark.read.parquet(parquet_path)
    out = df.mapInArrow(process, schema)
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()


# ── W2 — Batched generation ──────────────────────────────────────────────────
def run_w2(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    import pyarrow as pa
    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }

    def process(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        from sail_vs_spark.workloads.w2_batched import W2Batched
        wl = W2Batched()
        wl.init(closure_cfg)
        for batch in batch_iter:
            ids = batch.column("prompt_id").to_pylist()
            texts = batch.column("prompt_text").to_pylist()
            out = wl.apply_batch(ids, texts)
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(out["prompt_id"], type=pa.int64()),
                    pa.array(out["response"], type=pa.string()),
                ],
                names=["prompt_id", "response"],
            )

    df = spark.read.parquet(parquet_path)
    out = df.mapInArrow(process, "prompt_id long, response string")
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()


# ── W3 — Embedding + similarity ─────────────────────────────────────────────
def run_w3(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    import pyarrow as pa

    n_queries = int(cfg["workloads"]["w3_embedding"]["n_queries"])
    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }


    def process(batch_iter: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        from sail_vs_spark.workloads.w3_embedding import W3Embedding
        wl = W3Embedding(n_queries=n_queries)
        wl.init(closure_cfg)
        for batch in batch_iter:
            ids = batch.column("prompt_id").to_pylist()
            texts = batch.column("prompt_text").to_pylist()
            out = wl.apply_batch(ids, texts)
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(out["prompt_id"], type=pa.int64()),
                    pa.array(out["best_query_idx"], type=pa.int32()),
                    pa.array(out["best_similarity"], type=pa.float32()),
                ],
                names=["prompt_id", "best_query_idx", "best_similarity"],
            )

    schema = "prompt_id long, best_query_idx int, best_similarity float"
    df = spark.read.parquet(parquet_path)
    out = df.mapInArrow(process, schema)
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()
