"""Tests for MockGenerator / MockScorer / MockEmbedder + lazy loaders."""

from __future__ import annotations

import math

from sail_vs_spark.models.loaders import (
    get_embedder, get_generator, get_scorer, reset_singletons,
)
from sail_vs_spark.models.mock import MockEmbedder, MockGenerator, MockScorer


def setup_function(_) -> None:
    reset_singletons()


def test_generator_deterministic():
    g = MockGenerator(seed=42)
    a = g.generate(["hello"], n=2, max_new_tokens=8)
    b = g.generate(["hello"], n=2, max_new_tokens=8)
    assert a == b
    assert len(a[0]) == 2   # N=2 candidates


def test_generator_different_for_different_prompts():
    g = MockGenerator(seed=0)
    out = g.generate(["a", "b"], n=1)
    assert out[0] != out[1]


def test_scorer_deterministic():
    s = MockScorer(seed=7)
    r1 = s.score(["p"], ["r"])
    r2 = s.score(["p"], ["r"])
    assert r1 == r2
    assert -1.0 <= r1[0] <= 1.0


def test_scorer_length_mismatch():
    s = MockScorer()
    try:
        s.score(["a"], ["b", "c"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_embedder_unit_norm():
    e = MockEmbedder(dim=32, seed=0)
    [v] = e.encode(["hello"])
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_embedder_cosine_self():
    e = MockEmbedder(dim=32, seed=0)
    [v] = e.encode(["x"])
    assert abs(MockEmbedder.cosine_similarity(v, v) - 1.0) < 1e-6


def test_get_generator_falls_back_to_mock():
    # transformers almost certainly not installed in the minimal test env —
    # but even if it is, allow_mock=True guarantees we still get a callable.
    gen = get_generator({"name": "does-not-exist", "allow_mock": True,
                         "prefer_mock": True, "device": "cpu"})
    out = gen.generate(["hi"], n=1)
    assert len(out) == 1 and isinstance(out[0][0], str)


def test_get_scorer_singleton():
    reset_singletons()
    s1 = get_scorer({"name": "x", "prefer_mock": True, "device": "cpu"})
    s2 = get_scorer({"name": "y", "prefer_mock": True, "device": "cpu"})
    assert s1 is s2   # cached


def test_get_embedder_falls_back_to_mock():
    emb = get_embedder({"name": "x", "prefer_mock": True, "device": "cpu",
                        "dim": 16})
    [v] = emb.encode(["hi"])
    assert len(v) == 16
