"""Config B — Spark Pandas (Arrow) UDF.

Uses ``pandas_udf`` so batches are delivered as ``pandas.Series`` over the
Arrow IPC boundary. Versus Config A this removes the per-row pickle cost;
versus Configs C/D there is still a socket boundary crossed.

Expected: ~1.9× faster than A (Databricks published benchmark).

NOTE: We deliberately do NOT use ``from __future__ import annotations`` here.
PySpark 4.1's ``pandas_udf`` introspects the function signature at runtime to
decide which eval kind to use, and stringified annotations confuse that
inference with ``UNSUPPORTED_SIGNATURE``. Keeping annotations evaluated at
definition time is a PySpark-4.1 requirement for pandas UDFs.
"""

from typing import Any, Optional


# ── W0 ──────────────────────────────────────────────────────────────────────
def run_w0(spark: Any, parquet_path: str, depth: int,
           output_parquet: Optional[str] = None) -> int:
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import LongType

    @pandas_udf(LongType())
    def stage(s: pd.Series) -> pd.Series:
        return s + 1

    df = spark.read.parquet(parquet_path)
    out = df
    for _ in range(depth):
        out = out.withColumn("prompt_id", stage("prompt_id"))
    if output_parquet:
        out.write.mode("overwrite").parquet(output_parquet)
        return spark.read.parquet(output_parquet).count()
    return out.count()


# ── W1 — Best-of-N via a struct-returning pandas_udf ────────────────────────
def run_w1(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import (FloatType, IntegerType, LongType, StringType,
                                    StructField, StructType)

    schema = StructType([
        StructField("prompt_id", LongType(), False),
        StructField("best_response", StringType(), False),
        StructField("best_reward", FloatType(), False),
        StructField("n_candidates", IntegerType(), False),
    ])
    ncands = int(cfg["workloads"]["w1_best_of_n"]["n_candidates"])
    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }

    @pandas_udf(schema)
    def best_of_n(pid_s: pd.Series, text_s: pd.Series) -> pd.DataFrame:
        from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
        wl = W1BestOfN(n_candidates=ncands)
        wl.init(closure_cfg)
        out = wl.apply_batch(pid_s.tolist(), text_s.tolist())
        return pd.DataFrame(out)

    df = spark.read.parquet(parquet_path)
    out = (
        df
        .withColumn("_r", best_of_n("prompt_id", "prompt_text"))
        .select("_r.prompt_id", "_r.best_response",
                "_r.best_reward", "_r.n_candidates")
    )
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()


# ── W2 — Batched inference via Scalar pandas_udf ────────────────────────────
def run_w2(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StringType

    closure_cfg = {
        "models": cfg.get("models", {}),
        "hardware": cfg.get("hardware", {}),
    }

    @pandas_udf(StringType())
    def gen_batch(text_s: pd.Series) -> pd.Series:
        from sail_vs_spark.workloads.w2_batched import W2Batched
        wl = W2Batched()
        wl.init(closure_cfg)
        out = wl.apply_batch(range(len(text_s)), text_s.tolist())
        return pd.Series(out["response"])

    df = spark.read.parquet(parquet_path)
    out = df.withColumn("response", gen_batch("prompt_text")) \
            .select("prompt_id", "response")
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()


# ── W3 — Embedding + similarity via pandas_udf ──────────────────────────────
def run_w3(spark: Any, parquet_path: str, cfg: dict,
           output_parquet: str) -> int:
    import pandas as pd
    from pyspark.sql.functions import pandas_udf
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


    @pandas_udf(schema)
    def embed_sim(pid_s: pd.Series, text_s: pd.Series) -> pd.DataFrame:
        from sail_vs_spark.workloads.w3_embedding import W3Embedding
        wl = W3Embedding(n_queries=n_queries)
        wl.init(closure_cfg)
        out = wl.apply_batch(pid_s.tolist(), text_s.tolist())
        return pd.DataFrame(out)

    df = spark.read.parquet(parquet_path)
    out = (
        df
        .withColumn("_r", embed_sim("prompt_id", "prompt_text"))
        .select("_r.prompt_id", "_r.best_query_idx", "_r.best_similarity")
    )
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()
