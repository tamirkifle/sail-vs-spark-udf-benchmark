"""Config D — Sail Arrow UDTF via ``LATERAL`` (SQL-native per-row path).

CRITICAL SYNTAX (per prior learnings)
────────────────────────────────────
1. ``@udtf`` decorator MUST NOT include ``useArrow=True``. Doing so activates
   ``PySparkArrowTableUdf`` which drains the input iterator to check emptiness
   and deadlocks against Sail's streaming LATERAL model.
2. The SQL MUST use ``LATERAL fn(col, col)`` — NEVER wrap with ``TABLE()``.
   Expression resolvers reject ``ArrowTable`` kind; the bare LATERAL form is
   routed through the UDTF query resolver which accepts it.

So: ``SELECT u.* FROM prompts, LATERAL best_of_n(prompt_id, prompt_text) u``
"""

from __future__ import annotations

from typing import Any


# ── W0 — Chained trivial UDTF ────────────────────────────────────────────────
def run_w0(spark: Any, parquet_path: str, depth: int,
           output_parquet: str | None = None) -> int:
    from pyspark.sql.functions import udtf

    d = int(depth)

    @udtf(returnType="prompt_id long, value long")
    class TrivialUDTF:
        def eval(self, pid: int):    # noqa: D401
            x = int(pid)
            for _ in range(d):
                x = x + 1
            yield (int(pid), int(x))

    spark.udtf.register("w0_trivial", TrivialUDTF)

    df = spark.read.parquet(parquet_path)
    df.createOrReplaceTempView("prompts")
    sql = (
        "SELECT u.* FROM prompts, "
        "LATERAL w0_trivial(prompt_id) u"
    )
    out = spark.sql(sql)
    if output_parquet:
        out.write.mode("overwrite").parquet(output_parquet)
        return spark.read.parquet(output_parquet).count()
    return out.count()


# ── W1 — Best-of-N UDTF ──────────────────────────────────────────────────────
def run_w1(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    from pyspark.sql.functions import udtf

    ncands = int(cfg["workloads"]["w1_best_of_n"]["n_candidates"])
    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }
    schema = ("prompt_id long, best_response string, "
              "best_reward float, n_candidates int")

    # IMPORTANT: no useArrow=True — would deadlock under Sail's LATERAL path.
    @udtf(returnType=schema)
    class BestOfNUDTF:
        def __init__(self):
            # self._wl is loaded lazily so pickling doesn't need torch
            self._wl = None

        def eval(self, prompt_id: int, prompt_text: str):
            if self._wl is None:
                from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
                self._wl = W1BestOfN(n_candidates=ncands)
                self._wl.init(closure_cfg)
            pid2, resp, reward, n = self._wl.apply(
                int(prompt_id), str(prompt_text),
            )
            yield (int(pid2), str(resp), float(reward), int(n))

    spark.udtf.register("best_of_n", BestOfNUDTF)

    df = spark.read.parquet(parquet_path)
    df.createOrReplaceTempView("prompts")
    out = spark.sql(
        "SELECT u.* FROM prompts, "
        "LATERAL best_of_n(prompt_id, prompt_text) u"
    )
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()


# ── W2 — Batched generation via UDTF ────────────────────────────────────────
def run_w2(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    from pyspark.sql.functions import udtf

    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }

    @udtf(returnType="prompt_id long, response string")
    class W2UDTF:
        def __init__(self):
            self._wl = None

        def eval(self, prompt_id: int, prompt_text: str):
            if self._wl is None:
                from sail_vs_spark.workloads.w2_batched import W2Batched
                self._wl = W2Batched()
                self._wl.init(closure_cfg)
            pid, resp = self._wl.apply(int(prompt_id), str(prompt_text))
            yield (int(pid), str(resp))

    spark.udtf.register("w2_generate", W2UDTF)

    df = spark.read.parquet(parquet_path)
    df.createOrReplaceTempView("prompts")
    out = spark.sql(
        "SELECT u.* FROM prompts, "
        "LATERAL w2_generate(prompt_id, prompt_text) u"
    )
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()


# ── W3 — Embedding + similarity via UDTF ────────────────────────────────────
def run_w3(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    from pyspark.sql.functions import udtf
    n_queries = int(cfg["workloads"]["w3_embedding"]["n_queries"])
    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }


    schema = (
        "prompt_id long, best_query_idx int, best_similarity float"
    )

    @udtf(returnType=schema)
    class W3UDTF:
        def __init__(self):
            self._wl = None

        def eval(self, prompt_id: int, prompt_text: str):
            if self._wl is None:
                from sail_vs_spark.workloads.w3_embedding import W3Embedding
                self._wl = W3Embedding(n_queries=n_queries)
                self._wl.init(closure_cfg)
            pid, idx, sim = self._wl.apply(int(prompt_id), str(prompt_text))
            yield (int(pid), int(idx), float(sim))

    spark.udtf.register("w3_embed_sim", W3UDTF)
    df = spark.read.parquet(parquet_path)
    df.createOrReplaceTempView("prompts")
    out = spark.sql(
        "SELECT u.* FROM prompts, "
        "LATERAL w3_embed_sim(prompt_id, prompt_text) u"
    )
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()
