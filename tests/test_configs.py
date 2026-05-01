"""Integration tests that wire real Spark/Sail sessions to the benchmark code.

These are the closest thing to a full end-to-end check we can run on the
laptop: they spin up a local Spark session, run W0 and W1 through Configs A
and B, and verify the output schema and row counts.

Configs C/D are gated behind a running Sail server — if none is available we
skip them. On the cluster the ``sail spark run -f`` wrapper provides the
session, so tests here target A/B by default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pyspark = pytest.importorskip("pyspark")

from sail_vs_spark.dataset.prep import prepare


@pytest.fixture(scope="module")
def prompts_parquet(tmp_path_factory) -> str:
    d = tmp_path_factory.mktemp("prompts")
    path = prepare(d, n_rows=20, force_synthetic=True)
    return str(path)


@pytest.fixture(scope="module")
def spark():
    from sail_vs_spark.engines.spark_session import build_spark_session
    s = build_spark_session({"hardware": {"num_partitions": 2}})
    yield s
    s.stop()


def _cfg_mock() -> dict:
    return {
        "models": {
            "generator": {"prefer_mock": True, "max_new_tokens": 4},
            "scorer": {"prefer_mock": True},
            "embedder": {"prefer_mock": True, "dim": 16},
        },
        "hardware": {"device": "cpu", "num_partitions": 2},
        "workloads": {
            "w1_best_of_n": {"n_candidates": 2},
            "w3_embedding": {"n_queries": 3},
            "w4_agentic": {"max_iterations": 2, "reward_threshold": 0.2, "n_candidates": 2},
        },
    }



# ── Config A (Spark Row UDF) ────────────────────────────────────────────────
def test_config_a_w0_depth_2(spark, prompts_parquet):
    from sail_vs_spark.configs import config_a_spark_row as ca
    n = ca.run_w0(spark, prompts_parquet, depth=2, output_parquet=None)
    assert n == 20


def test_config_a_w1(spark, prompts_parquet, tmp_path):
    from sail_vs_spark.configs import config_a_spark_row as ca
    out = str(tmp_path / "a_w1.parquet")
    n = ca.run_w1(spark, prompts_parquet, _cfg_mock(), out)
    assert n == 20
    t = pq.read_table(out)
    assert set(t.column_names) == {
        "prompt_id", "best_response", "best_reward", "n_candidates"
    }


def test_config_a_w2(spark, prompts_parquet, tmp_path):
    from sail_vs_spark.configs import config_a_spark_row as ca
    out = str(tmp_path / "a_w2.parquet")
    n = ca.run_w2(spark, prompts_parquet, _cfg_mock(), out)
    assert n == 20
    t = pq.read_table(out)
    assert set(t.column_names) == {"prompt_id", "response"}


def test_config_a_w3(spark, prompts_parquet, tmp_path):
    from sail_vs_spark.configs import config_a_spark_row as ca
    out = str(tmp_path / "a_w3.parquet")
    n = ca.run_w3(spark, prompts_parquet, _cfg_mock(), out)
    assert n == 20
    t = pq.read_table(out)
    assert set(t.column_names) == {
        "prompt_id", "best_query_idx", "best_similarity"
    }


def test_config_a_w4(spark, prompts_parquet, tmp_path):
    from sail_vs_spark.configs import config_a_spark_row as ca
    out = str(tmp_path / "a_w4.parquet")
    n = ca.run_w4(spark, prompts_parquet, _cfg_mock(), out)
    assert n == 20
    t = pq.read_table(out)
    assert set(t.column_names) == {
        "prompt_id", "final_response", "iterations", "best_reward"
    }



# ── Config B (Spark Pandas/Arrow UDF) ───────────────────────────────────────
def test_config_b_w0_depth_2(spark, prompts_parquet):
    from sail_vs_spark.configs import config_b_spark_pandas as cb
    n = cb.run_w0(spark, prompts_parquet, depth=2, output_parquet=None)
    assert n == 20


def test_config_b_w1(spark, prompts_parquet, tmp_path):
    from sail_vs_spark.configs import config_b_spark_pandas as cb
    out = str(tmp_path / "b_w1.parquet")
    n = cb.run_w1(spark, prompts_parquet, _cfg_mock(), out)
    assert n == 20
    t = pq.read_table(out)
    assert set(t.column_names) == {
        "prompt_id", "best_response", "best_reward", "n_candidates"
    }


def test_config_b_w2(spark, prompts_parquet, tmp_path):
    from sail_vs_spark.configs import config_b_spark_pandas as cb
    out = str(tmp_path / "b_w2.parquet")
    n = cb.run_w2(spark, prompts_parquet, _cfg_mock(), out)
    assert n == 20


def test_config_b_w3(spark, prompts_parquet, tmp_path):
    from sail_vs_spark.configs import config_b_spark_pandas as cb
    out = str(tmp_path / "b_w3.parquet")
    n = cb.run_w3(spark, prompts_parquet, _cfg_mock(), out)
    assert n == 20


def test_config_b_w4(spark, prompts_parquet, tmp_path):
    from sail_vs_spark.configs import config_b_spark_pandas as cb
    out = str(tmp_path / "b_w4.parquet")
    n = cb.run_w4(spark, prompts_parquet, _cfg_mock(), out)
    assert n == 20



# ── Cross-config equivalence: W0 output row count is identical ──────────────
def test_w0_row_count_identical_across_configs(spark, prompts_parquet):
    from sail_vs_spark.configs import (
        config_a_spark_row as ca,
        config_b_spark_pandas as cb,
    )
    # The W0 output should be 20 rows (dataset size) regardless of depth/config.
    for depth in (1, 2, 3):
        na = ca.run_w0(spark, prompts_parquet, depth=depth, output_parquet=None)
        nb = cb.run_w0(spark, prompts_parquet, depth=depth, output_parquet=None)
        assert na == nb == 20


# ── Configs C/D are best-effort: skipped unless a Sail server is running ────
@pytest.mark.skipif(
    os.environ.get("SAIL_REMOTE_URL") is None,
    reason="Requires SAIL_REMOTE_URL pointing at a running sail server",
)
def test_config_c_w0(prompts_parquet):
    from sail_vs_spark.engines.sail_session import build_sail_session
    from sail_vs_spark.configs import config_c_sail_arrow as cc
    remote = os.environ["SAIL_REMOTE_URL"]
    spark = build_sail_session({"runner": {"sail_remote_url": remote}})
    try:
        n = cc.run_w0(spark, prompts_parquet, depth=2, output_parquet=None)
        assert n == 20
    finally:
        spark.stop()
