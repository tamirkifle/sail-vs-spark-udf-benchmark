"""Tests for MetricsCollector.

Runs on any machine (including those without an NVIDIA GPU) — GPU fields will
simply be zero when nvidia-smi is unavailable.
"""

from __future__ import annotations

import json
import time

import pytest

from sail_vs_spark.profiling.metrics_collector import MetricsCollector


def test_start_stop_produces_samples():
    c = MetricsCollector("unit-test", sample_interval_sec=0.05)
    c.start()
    time.sleep(0.5)
    stats = c.stop()
    assert stats["n_samples"] >= 5
    assert stats["wall_clock_sec"] >= 0.5
    # basic CPU/RAM fields should be populated on any machine
    assert stats["avg_cpu_pct"] >= 0
    assert stats["peak_rss_mb"] > 0


def test_save_produces_valid_json(tmp_path):
    c = MetricsCollector("save-test", sample_interval_sec=0.05)
    c.start()
    time.sleep(0.25)
    c.stop()
    path = tmp_path / "stats.json"
    c.save(path, extra={"output_rows": 42})
    with open(path) as fh:
        data = json.load(fh)
    assert data["config"] == "save-test"
    assert data["output_rows"] == 42
    assert "samples" in data


def test_report_includes_pipeline_continuity_key():
    c = MetricsCollector("pc-test", sample_interval_sec=0.05)
    c.start()
    time.sleep(0.25)
    c.stop()
    r = c.report()
    assert "pipeline_continuity" in r
    assert "bytes_written_delta" in r
    assert 0.0 <= r["pipeline_continuity"] <= 1.0


def test_disk_io_fields_are_numeric():
    c = MetricsCollector("io-test", sample_interval_sec=0.05)
    c.start()
    # Trigger some IO so disk delta shows up (if psutil supports it here)
    with open("/tmp/sail_vs_spark_io_test.bin", "wb") as f:
        f.write(b"x" * 1_000_000)
    time.sleep(0.25)
    c.stop()
    r = c.report()
    assert isinstance(r["bytes_written_delta"], int)
    assert r["bytes_written_delta"] >= 0
