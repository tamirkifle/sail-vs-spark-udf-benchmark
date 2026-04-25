"""Tests for BoundaryTimer — the pickle-safety and phase recording guarantees."""

from __future__ import annotations

import json
import pickle
import time

import pytest

from sail_vs_spark.profiling.boundary_timer import BoundaryTimer


def test_measure_records_phase_time():
    t = BoundaryTimer("test")
    with t.measure("INFERENCE"):
        time.sleep(0.02)
    r = t.report()
    assert r["phases"]["INFERENCE"]["call_count"] == 1
    assert r["phases"]["INFERENCE"]["total_sec"] >= 0.018


def test_record_direct():
    t = BoundaryTimer("test")
    t.record("SCORE", 0.05)
    t.record("SCORE", 0.03)
    r = t.report()
    assert r["phases"]["SCORE"]["call_count"] == 2
    # avg should be ~40ms
    assert 30 <= r["phases"]["SCORE"]["avg_ms"] <= 50


def test_unknown_phase_rejected():
    t = BoundaryTimer("test")
    with pytest.raises(ValueError):
        t.measure("BOGUS")
    with pytest.raises(ValueError):
        t.record("BOGUS", 0.01)


def test_picklable_roundtrip():
    """BoundaryTimer must survive cloudpickle since it flows inside UDF closures."""
    t = BoundaryTimer("pickle-test")
    t.record("INFERENCE", 0.1)
    t.record("DATA_TRANSFER_IN", 0.001)

    blob = pickle.dumps(t)
    t2: BoundaryTimer = pickle.loads(blob)

    # Data preserved
    r = t2.report()
    assert r["phases"]["INFERENCE"]["total_sec"] == pytest.approx(0.1, abs=1e-4)
    assert r["phases"]["DATA_TRANSFER_IN"]["call_count"] == 1

    # Lock recreated (thread-safe after unpickling)
    assert t2._lock is not None
    with t2.measure("SCORE"):
        pass


def test_serialization_tax_calculation():
    """serialization_tax_pct = (transfer_in + transfer_out) / wall.

    We use ``time.sleep`` to make wall-clock elapse predictably and ``record``
    to inject phase totals so the arithmetic is deterministic.
    """
    t = BoundaryTimer("serial-test")
    # Make wall elapse ~0.4s
    time.sleep(0.4)
    # Inject 0.1s total transfer time
    t.record("DATA_TRANSFER_IN", 0.05)
    t.record("DATA_TRANSFER_OUT", 0.05)
    t.record("INFERENCE", 0.3)
    r = t.report()
    # transfer = 0.1s, wall >= 0.4s => tax should be in [15%, 30%] approx
    assert 10 <= r["serialization_tax_pct"] <= 35, r
    # compute = 0.3s => compute_pct should be in [40%, 80%] approx
    assert 40 <= r["compute_pct"] <= 80, r


def test_save_writes_valid_json(tmp_path):
    t = BoundaryTimer("save-test")
    t.record("INFERENCE", 0.01)
    path = tmp_path / "boundary.json"
    t.save(path)
    with open(path) as fh:
        data = json.load(fh)
    assert data["config"] == "save-test"
    assert data["phases"]["INFERENCE"]["call_count"] == 1
    # Derived metrics present
    assert "serialization_tax_pct" in data
    assert "compute_pct" in data


def test_merge_from_dict():
    t1 = BoundaryTimer("main")
    t1.record("INFERENCE", 0.1)
    t2 = BoundaryTimer("worker")
    t2.record("INFERENCE", 0.2)
    t2.record("SCORE", 0.05)
    t1.merge_from_dict(t2.export_raw())
    r = t1.report()
    assert r["phases"]["INFERENCE"]["call_count"] == 2
    assert r["phases"]["INFERENCE"]["total_sec"] == pytest.approx(0.3)
    assert r["phases"]["SCORE"]["call_count"] == 1


def test_p95_handles_few_samples():
    t = BoundaryTimer("p95-test")
    t.record("INFERENCE", 0.01)
    r = t.report()
    # p95 of one sample is just that sample
    assert r["phases"]["INFERENCE"]["p95_ms"] > 0
