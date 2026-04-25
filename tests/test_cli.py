"""Smoke test of the CLI end-to-end: dataset prep → run_one on Config A → JSON artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pyspark = pytest.importorskip("pyspark")
import yaml

from sail_vs_spark.dataset.prep import prepare
from sail_vs_spark.runner.cli import run_one


@pytest.fixture
def mini_cfg(tmp_path) -> dict:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    prepare(data_dir, n_rows=10, force_synthetic=True)
    cfg = {
        "profile": "test",
        "hardware": {"device": "cpu", "num_partitions": 1},
        "dataset": {"out_dir": str(data_dir), "n_rows": 10},
        "models": {
            "generator": {"prefer_mock": True, "max_new_tokens": 4},
            "scorer": {"prefer_mock": True},
            "embedder": {"prefer_mock": True, "dim": 16},
        },
        "workloads": {
            "w0_chained": {"depth": 2},
            "w1_best_of_n": {"n_candidates": 2},
            "w3_embedding": {"n_queries": 3},
        },
        "runner": {
            "results_dir": str(tmp_path / "results"),
            "sample_interval_sec": 0.1,
        },
    }
    return cfg



def test_run_one_config_a_w0(mini_cfg, tmp_path):
    rdir = Path(mini_cfg["runner"]["results_dir"])
    man = run_one(mini_cfg, "w0", "A",
                  results_dir=rdir, run_id="smoke_w0_A")
    assert man["workload"] == "w0"
    assert man["execution"] == "A"
    assert man["output_rows"] == 10
    # Manifest + stats files exist
    assert (rdir / "smoke_w0_A_manifest.json").exists()
    assert (rdir / "smoke_w0_A_stats.json").exists()
    stats = json.loads((rdir / "smoke_w0_A_stats.json").read_text())
    assert stats["wall_clock_sec"] > 0
    assert "pipeline_continuity" in stats


def test_run_one_config_b_w1(mini_cfg):
    rdir = Path(mini_cfg["runner"]["results_dir"])
    man = run_one(mini_cfg, "w1", "B",
                  results_dir=rdir, run_id="smoke_w1_B")
    assert man["output_rows"] == 10
    out = Path(man["output_parquet"])
    assert out.exists()
