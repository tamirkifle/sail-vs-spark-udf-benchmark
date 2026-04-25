"""Tests for W0-W3 workloads using mock models.

These tests establish the *payload contract*: every workload must produce
the right schema and yield identical per-row and per-batch results.
"""

from __future__ import annotations

import pytest

from sail_vs_spark.models.loaders import reset_singletons
from sail_vs_spark.workloads.base import Workload
from sail_vs_spark.workloads.registry import make_workload
from sail_vs_spark.workloads.w0_chained import W0Chained
from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
from sail_vs_spark.workloads.w2_batched import W2Batched
from sail_vs_spark.workloads.w3_embedding import W3Embedding


def setup_function(_) -> None:
    reset_singletons()


# ── W0 ───────────────────────────────────────────────────────────────────────
def test_w0_depth_1():
    w = W0Chained(depth=1)
    w.init({})
    pid, v = w.apply(5, "anything")
    assert pid == 5
    assert v == 6


def test_w0_depth_3():
    w = W0Chained(depth=3)
    w.init({})
    pid, v = w.apply(5, "anything")
    assert v == 8   # 5 + 3


def test_w0_batch_equivalence():
    w = W0Chained(depth=2)
    w.init({})
    out = w.apply_batch([1, 2, 3], ["a", "b", "c"])
    assert out["value"] == [3, 4, 5]


def test_w0_depth_bounds():
    with pytest.raises(ValueError):
        W0Chained(depth=0)
    with pytest.raises(ValueError):
        W0Chained(depth=99)


# ── W1 ───────────────────────────────────────────────────────────────────────
def _cfg_mock() -> dict:
    return {
        "models": {
            "generator": {"prefer_mock": True, "max_new_tokens": 8},
            "scorer": {"prefer_mock": True},
            "embedder": {"prefer_mock": True, "dim": 16},
        },
        "hardware": {"device": "cpu"},
    }


def test_w1_apply_shape():
    w = W1BestOfN(n_candidates=3)
    w.init(_cfg_mock())
    pid, resp, reward, n = w.apply(1, "hello world")
    assert pid == 1
    assert isinstance(resp, str) and resp
    assert -1.0 <= reward <= 1.0
    assert n == 3


def test_w1_batch_equals_single_rows():
    w = W1BestOfN(n_candidates=2)
    w.init(_cfg_mock())
    single = [w.apply(i, f"prompt {i}") for i in range(3)]
    # reset generator/scorer seeds — they're deterministic singletons
    reset_singletons()
    w2 = W1BestOfN(n_candidates=2)
    w2.init(_cfg_mock())
    batch = w2.apply_batch([0, 1, 2], ["prompt 0", "prompt 1", "prompt 2"])
    for i, (pid, resp, reward, n) in enumerate(single):
        assert batch["prompt_id"][i] == pid
        assert batch["best_response"][i] == resp
        assert batch["best_reward"][i] == pytest.approx(reward, abs=1e-6)


# ── W2 ───────────────────────────────────────────────────────────────────────
def test_w2_apply():
    w = W2Batched()
    w.init(_cfg_mock())
    pid, resp = w.apply(42, "test")
    assert pid == 42
    assert isinstance(resp, str)


def test_w2_batch():
    w = W2Batched()
    w.init(_cfg_mock())
    out = w.apply_batch([1, 2], ["a", "b"])
    assert out["prompt_id"] == [1, 2]
    assert len(out["response"]) == 2


# ── W3 ───────────────────────────────────────────────────────────────────────
def test_w3_apply():
    w = W3Embedding(n_queries=4)
    w.init(_cfg_mock())
    pid, idx, sim = w.apply(7, "machine learning is great")
    assert pid == 7
    assert 0 <= idx < 4
    assert -1.0 <= sim <= 1.0


def test_w3_batch_equals_single():
    w = W3Embedding(n_queries=3)
    w.init(_cfg_mock())
    s = [w.apply(i, f"p{i}") for i in range(3)]
    reset_singletons()
    w2 = W3Embedding(n_queries=3)
    w2.init(_cfg_mock())
    b = w2.apply_batch([0, 1, 2], ["p0", "p1", "p2"])
    for i, (pid, idx, sim) in enumerate(s):
        assert b["prompt_id"][i] == pid
        assert b["best_query_idx"][i] == idx
        assert b["best_similarity"][i] == pytest.approx(sim, abs=1e-6)


# ── registry ─────────────────────────────────────────────────────────────────
def test_registry_all_four():
    base_cfg = _cfg_mock()
    base_cfg["workloads"] = {
        "w0_chained": {"depth": 2},
        "w1_best_of_n": {"n_candidates": 2},
        "w3_embedding": {"n_queries": 3},
    }
    for code, expected_cls in [
        ("w0", W0Chained), ("w1", W1BestOfN),
        ("w2", W2Batched), ("w3", W3Embedding),
    ]:
        wl = make_workload(code, base_cfg)
        assert isinstance(wl, Workload)
        assert isinstance(wl, expected_cls)


def test_registry_unknown_raises():
    with pytest.raises(ValueError):
        make_workload("w99", _cfg_mock())


def test_registry_w0_respects_depth():
    cfg = _cfg_mock()
    cfg["workloads"] = {"w0_chained": {"depth": 3}}
    wl = make_workload("w0", cfg)
    pid, v = wl.apply(10, "x")
    assert v == 13
