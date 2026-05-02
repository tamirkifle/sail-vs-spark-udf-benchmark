"""Test the dataset prep helper.

We force the synthetic fallback so the test does not require network or HF
credentials; the schema guarantee is the thing we care about.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from sail_vs_spark.dataset.prep import prepare
import sail_vs_spark.dataset.prep as prep


def test_synthetic_prepare_writes_parquet(tmp_path):
    out = prepare(tmp_path, n_rows=25, force_synthetic=True)
    assert out.exists()
    t = pq.read_table(out)
    assert t.num_rows == 25
    # Exact schema required by the rest of the pipeline
    assert t.schema.field("prompt_id").type == pa.int64()
    assert t.schema.field("prompt_text").type == pa.string()
    # Meta file written alongside
    meta = json.loads((tmp_path / "prompts_meta.json").read_text())
    assert meta["source"] == "synthetic"
    assert meta["n_rows"] == 25


def test_synthetic_deterministic(tmp_path):
    """Same inputs → identical parquet content."""
    p1 = prepare(tmp_path / "a", n_rows=10, force_synthetic=True)
    p2 = prepare(tmp_path / "b", n_rows=10, force_synthetic=True)
    t1 = pq.read_table(p1).to_pylist()
    t2 = pq.read_table(p2).to_pylist()
    assert t1 == t2


def test_prompt_ids_are_dense(tmp_path):
    out = prepare(tmp_path, n_rows=50, force_synthetic=True)
    rows = pq.read_table(out).to_pylist()
    ids = [r["prompt_id"] for r in rows]
    assert ids == list(range(50))


def test_hf_loader_streams_only_requested_rows(monkeypatch):
    calls = []

    def fake_load_dataset(source, split, streaming=False):
        calls.append((source, split, streaming))
        assert streaming is True
        for i in range(1000):
            yield {"instruction": f"prompt {i}"}

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    rows = prep._load_from_hf("fake/source", "train", 3)

    assert rows == [
        {"prompt_id": 0, "prompt_text": "prompt 0"},
        {"prompt_id": 1, "prompt_text": "prompt 1"},
        {"prompt_id": 2, "prompt_text": "prompt 2"},
    ]
    assert calls == [("fake/source", "train", True)]
