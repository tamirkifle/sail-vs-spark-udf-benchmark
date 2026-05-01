"""Config D — Sail Arrow UDTF via ``LATERAL`` (SQL-native per-partition-batch path).

W0 is genuinely row-level (trivial increment), which is what UDTFs are designed
for. W1/W2/W3 accumulate rows in eval() and flush the whole partition as one
batch in terminate() — matching the batch semantics of Config C's mapInArrow.
"""

from __future__ import annotations
from typing import Any

try:
    from sail_vs_spark.engines.sail_session import _WORKER_ENV as _ENV
except ImportError:
    import os as _os
    _ENV: dict = {k: _os.environ[k] for k in (
        "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE", "CUDA_VISIBLE_DEVICES",
    ) if k in _os.environ}

# ── W0: Chained Trivial UDTF ────────────────────────────────────────────────
def run_w0(spark: Any, parquet_path: str, depth: int,
           output_parquet: str | None = None) -> int:
    from pyspark.sql.functions import udtf

    @udtf(returnType="prompt_id long, value long")
    class TrivialUDTF:
        def eval(self, pid: int, val: int):    # noqa: D401
            yield (int(pid), int(val) + 1)

    spark.udtf.register("w0_stage", TrivialUDTF)

    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    df.createOrReplaceTempView("prompts")

    sql = "SELECT p.prompt_id, p.prompt_id as value FROM prompts p"
    for i in range(depth):
        sql = f"SELECT u.prompt_id, u.value FROM ({sql}) t, LATERAL w0_stage(t.prompt_id, t.value) u"

    out = spark.sql(sql)
    if output_parquet:
        out.write.mode("overwrite").parquet(output_parquet)
        return spark.read.parquet(output_parquet).count()
    return out.count()

# ── W1 — Best-of-N UDTF (batched per partition) ─────────────────────────────
def run_w1(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    from pyspark.sql.functions import udtf
    ncands = int(cfg["workloads"]["w1_best_of_n"]["n_candidates"])
    closure_cfg = {"models": cfg.get("models", {}), "hardware": cfg.get("hardware", {})}
    schema = "prompt_id long, best_response string, best_reward float, n_candidates int"

    @udtf(returnType=schema)
    class BestOfNUDTF:
        def __init__(self):
            self._wl = None
            self._buffer = []

        def eval(self, prompt_id: int, prompt_text: str):
            if self._wl is None:
                import os; os.environ.update(_ENV)
                from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
                self._wl = W1BestOfN(n_candidates=ncands)
                self._wl.init(closure_cfg)
            self._buffer.append((int(prompt_id), str(prompt_text)))

        def terminate(self):
            if not self._buffer:
                return
            ids = [r[0] for r in self._buffer]
            texts = [r[1] for r in self._buffer]
            out = self._wl.apply_batch(ids, texts)
            for pid, resp, reward, n in zip(
                out["prompt_id"], out["best_response"],
                out["best_reward"], out["n_candidates"]
            ):
                yield (int(pid), str(resp), float(reward), int(n))

    spark.udtf.register("best_of_n", BestOfNUDTF)
    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    df.createOrReplaceTempView("prompts")
    out = spark.sql("SELECT u.* FROM prompts, LATERAL best_of_n(prompt_id, prompt_text) u")
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()

# ── W2 — Batched generation UDTF (batched per partition) ────────────────────
def run_w2(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    from pyspark.sql.functions import udtf
    closure_cfg = {"models": cfg.get("models", {}), "hardware": cfg.get("hardware", {})}

    @udtf(returnType="prompt_id long, response string")
    class W2UDTF:
        def __init__(self):
            self._wl = None
            self._buffer = []
            from sail_vs_spark.profiling.boundary_timer import BoundaryTimer
            self._timer = BoundaryTimer("config_d", enable_tracing=True)

        def eval(self, prompt_id: int, prompt_text: str):
            if self._wl is None:
                import os; os.environ.update(_ENV)
                from sail_vs_spark.workloads.w2_batched import W2Batched
                self._wl = W2Batched()
                self._wl.init(closure_cfg)
            self._buffer.append((int(prompt_id), str(prompt_text)))

        def terminate(self):
            if not self._buffer:
                return
            import os
            ids = [r[0] for r in self._buffer]
            texts = [r[1] for r in self._buffer]
            with self._timer.measure("UDF_BATCH_EXECUTION"):
                out = self._wl.apply_batch(ids, texts)
            self._timer.save_trace(f"/tmp/sail_traces/trace_{os.getpid()}.jsonl")
            for pid, resp in zip(out["prompt_id"], out["response"]):
                yield (int(pid), str(resp))

    spark.udtf.register("w2_generate", W2UDTF)
    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    df.createOrReplaceTempView("prompts")
    out = spark.sql("SELECT u.* FROM prompts, LATERAL w2_generate(prompt_id, prompt_text) u")
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()

# ── W4 — Agentic loop UDTF (entire loop buffered and flushed once per partition)
def run_w4(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    from pyspark.sql.functions import udtf
    w4_cfg = cfg.get("workloads", {}).get("w4_agentic", {})
    max_iter = int(w4_cfg.get("max_iterations", 3))
    threshold = float(w4_cfg.get("reward_threshold", 0.5))
    ncands = int(w4_cfg.get("n_candidates", 2))
    closure_cfg = {"models": cfg.get("models", {}), "hardware": cfg.get("hardware", {})}
    schema = "prompt_id long, final_response string, iterations int, best_reward float"

    @udtf(returnType=schema)
    class W4UDTF:
        def __init__(self):
            self._wl = None
            self._buffer = []

        def eval(self, prompt_id: int, prompt_text: str):
            if self._wl is None:
                import os; os.environ.update(_ENV)
                from sail_vs_spark.workloads.w4_agentic import W4Agentic
                self._wl = W4Agentic(max_iterations=max_iter,
                                     reward_threshold=threshold,
                                     n_candidates=ncands)
                self._wl.init(closure_cfg)
            self._buffer.append((int(prompt_id), str(prompt_text)))

        def terminate(self):
            if not self._buffer:
                return
            ids = [r[0] for r in self._buffer]
            texts = [r[1] for r in self._buffer]
            out = self._wl.apply_batch(ids, texts)
            for pid, resp, iters, reward in zip(
                out["prompt_id"], out["final_response"],
                out["iterations"], out["best_reward"]
            ):
                yield (int(pid), str(resp), int(iters), float(reward))

    spark.udtf.register("w4_agentic", W4UDTF)
    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    df.createOrReplaceTempView("prompts")
    out = spark.sql("SELECT u.* FROM prompts, LATERAL w4_agentic(prompt_id, prompt_text) u")
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()

# ── W3 — Embedding + similarity UDTF (batched per partition) ────────────────
def run_w3(spark: Any, parquet_path: str, cfg: dict, output_parquet: str) -> int:
    from pyspark.sql.functions import udtf
    n_queries = int(cfg["workloads"]["w3_embedding"]["n_queries"])
    closure_cfg = {"models": cfg.get("models", {}), "hardware": cfg.get("hardware", {})}

    @udtf(returnType="prompt_id long, best_query_idx int, best_similarity float")
    class W3UDTF:
        def __init__(self):
            self._wl = None
            self._buffer = []

        def eval(self, prompt_id: int, prompt_text: str):
            if self._wl is None:
                import os; os.environ.update(_ENV)
                from sail_vs_spark.workloads.w3_embedding import W3Embedding
                self._wl = W3Embedding(n_queries=n_queries)
                self._wl.init(closure_cfg)
            self._buffer.append((int(prompt_id), str(prompt_text)))

        def terminate(self):
            if not self._buffer:
                return
            ids = [r[0] for r in self._buffer]
            texts = [r[1] for r in self._buffer]
            out = self._wl.apply_batch(ids, texts)
            for pid, idx, sim in zip(
                out["prompt_id"], out["best_query_idx"], out["best_similarity"]
            ):
                yield (int(pid), int(idx), float(sim))

    spark.udtf.register("w3_embed_sim", W3UDTF)
    n_part = int(spark.conf.get("spark.sql.shuffle.partitions", "2"))
    df = spark.read.parquet(parquet_path).repartition(n_part)
    df.createOrReplaceTempView("prompts")
    out = spark.sql("SELECT u.* FROM prompts, LATERAL w3_embed_sim(prompt_id, prompt_text) u")
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()
