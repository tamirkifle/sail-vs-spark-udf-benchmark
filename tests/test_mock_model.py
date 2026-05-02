"""Tests for MockGenerator / MockScorer / MockEmbedder + lazy loaders."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from sail_vs_spark.models.adapters import _HFGenerator, _HFScorer, _VLLMGenerator
from sail_vs_spark.models.loaders import (
    get_embedder, get_generator, get_scorer, reset_singletons,
)
from sail_vs_spark.models.mock import MockEmbedder, MockGenerator, MockScorer
from sail_vs_spark.profiling.boundary_timer import BoundaryTimer


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
    gen = get_generator({"name": "does-not-exist", "allow_mock": True,
                         "prefer_mock": True, "device": "cpu"})
    out = gen.generate(["hi"], n=1)
    assert len(out) == 1 and isinstance(out[0][0], str)


def test_get_generator_uses_vllm_url(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    gen = get_generator(
        {
            "name": "served-model",
            "server_url": "http://127.0.0.1:8000",
            "allow_mock": False,
        }
    )
    assert isinstance(gen, _VLLMGenerator)
    assert gen.model_id == "served-model"
    assert gen.server_url == "http://127.0.0.1:8000"


def test_get_generator_uses_transformers_provider(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    class _FakeHFGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def bind_timer(self, timer):
            self.timer = timer
            return self

    monkeypatch.setattr("sail_vs_spark.models.factory._HFGenerator", _FakeHFGenerator)
    gen = get_generator(
        {
            "name": "local-model",
            "provider": "transformers",
            "allow_mock": False,
            "max_new_tokens": 12,
        }
    )
    assert isinstance(gen, _FakeHFGenerator)
    assert gen.kwargs["model_id"] == "local-model"
    assert gen.kwargs["max_new_tokens"] == 12


def test_get_generator_transformers_failure_is_clear(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    class _BrokenHFGenerator:
        def __init__(self, **kwargs):
            raise OSError("missing weights")

    monkeypatch.setattr("sail_vs_spark.models.factory._HFGenerator", _BrokenHFGenerator)
    with pytest.raises(RuntimeError, match="Transformers generator failed to load"):
        get_generator(
            {
                "name": "local-model",
                "provider": "transformers",
                "allow_mock": False,
            }
        )


def test_get_generator_requires_real_backend_when_mock_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    try:
        get_generator({"name": "served-model", "allow_mock": False})
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "provider=transformers" in str(exc)


def test_get_scorer_singleton():
    reset_singletons()
    s1 = get_scorer({"name": "x", "prefer_mock": True, "device": "cpu"})
    s2 = get_scorer({"name": "y", "prefer_mock": True, "device": "cpu"})
    assert s1 is s2   # cached


def test_hf_scorer_micro_batches_inputs():
    class _FakeTensor:
        def __init__(self, values):
            self.values = list(values)

        def to(self, device):
            return self

        def squeeze(self, dim=-1):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self.values)

    class _FakeTok:
        def __init__(self):
            self.calls = []

        def __call__(self, prompts, responses, **kwargs):
            prompts = list(prompts)
            responses = list(responses)
            self.calls.append((prompts, responses, kwargs["max_length"]))
            n_items = len(prompts)
            return {
                "input_ids": _FakeTensor(range(n_items)),
                "attention_mask": _FakeTensor([1] * n_items),
            }

    class _FakeModel:
        def __init__(self):
            self.device = "cpu"
            self.calls = []

        def eval(self):
            return None

        def __call__(self, **inputs):
            batch_size = len(inputs["input_ids"].values)
            self.calls.append(batch_size)
            start = 10 * (len(self.calls) - 1)
            logits = _FakeTensor(range(start, start + batch_size))
            return SimpleNamespace(logits=logits)

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTorch:
        def no_grad(self):
            return _FakeNoGrad()

    scorer = object.__new__(_HFScorer)
    scorer.tok = _FakeTok()
    scorer.mdl = _FakeModel()
    scorer._torch = _FakeTorch()
    scorer.score_batch_size = 4
    scorer.max_length = 384

    out = scorer.score(
        [f"p{i}" for i in range(10)],
        [f"r{i}" for i in range(10)],
    )

    assert scorer.tok.calls == [
        ([f"p{i}" for i in range(4)], [f"r{i}" for i in range(4)], 384),
        ([f"p{i}" for i in range(4, 8)], [f"r{i}" for i in range(4, 8)], 384),
        ([f"p{i}" for i in range(8, 10)], [f"r{i}" for i in range(8, 10)], 384),
    ]
    assert scorer.mdl.calls == [4, 4, 2]
    assert out == [0, 1, 2, 3, 10, 11, 12, 13, 20, 21]


def test_hf_generator_uses_cpu_generate_and_decodes_new_tokens():
    class _FakeTensor:
        def __init__(self, values):
            self.values = values

        @property
        def shape(self):
            return (len(self.values), len(self.values[0]))

        def to(self, device):
            return self

        def __getitem__(self, item):
            rows, cols = item
            selected = self.values[rows] if isinstance(rows, int) else self.values[rows]
            if rows == slice(None):
                selected = self.values
            return _FakeTensor([row[cols] for row in selected])

    class _FakeTok:
        pad_token_id = None
        eos_token = "<eos>"

        def __init__(self):
            self.pad_token = None

        def __call__(self, text, return_tensors="pt"):
            return {"input_ids": _FakeTensor([[1, 2, 3]])}

        def batch_decode(self, output_ids, skip_special_tokens=True):
            return [" decoded " for _ in output_ids.values]

    class _FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            n = kwargs["num_return_sequences"]
            return _FakeTensor([[1, 2, 3, 9, 8] for _ in range(n)])

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTorch:
        def no_grad(self):
            return _FakeNoGrad()

    gen = object.__new__(_HFGenerator)
    gen.tok = _FakeTok()
    gen.mdl = _FakeModel()
    gen._torch = _FakeTorch()
    gen.model_id = "fake"
    gen.max_new_tokens = 5
    gen.temperature = 0.7
    gen.top_p = 0.9
    gen.top_k = 50
    gen._timer = BoundaryTimer("hf-generator")

    out = gen.generate(["hello"], n=2, max_new_tokens=4)

    assert out == [["decoded", "decoded"]]
    call = gen.mdl.calls[0]
    assert call["num_return_sequences"] == 2
    assert call["max_new_tokens"] == 4
    assert call["do_sample"] is True
    report = gen._timer.report()
    assert report["phases"]["TOKENIZE"]["call_count"] == 1
    assert report["phases"]["INFERENCE"]["call_count"] == 1
    assert report["phases"]["DETOKENIZE"]["call_count"] == 1


def test_get_embedder_falls_back_to_mock():
    emb = get_embedder({"name": "x", "prefer_mock": True, "device": "cpu",
                        "dim": 16})
    [v] = emb.encode(["hi"])
    assert len(v) == 16


def test_mock_models_emit_phase_timing_when_timer_bound():
    timer = BoundaryTimer("mock-phases")
    gen = get_generator({"name": "g", "prefer_mock": True, "device": "cpu"}, timer=timer)
    scorer = get_scorer({"name": "s", "prefer_mock": True, "device": "cpu"}, timer=timer)
    embedder = get_embedder({"name": "e", "prefer_mock": True, "device": "cpu", "dim": 8}, timer=timer)

    gen.generate(["hello"], n=2)
    scorer.score(["hello"], ["world"])
    embedder.encode(["hello"])

    report = timer.report()
    assert report["phases"]["INFERENCE"]["call_count"] == 1
    assert report["phases"]["SCORE"]["call_count"] == 1
    assert report["phases"]["EMBED"]["call_count"] == 1
