"""Tests for MetricsCollector.

Runs on any machine (including those without an NVIDIA GPU) — GPU fields will
simply be zero when nvidia-smi is unavailable.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

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
    assert r["disk_counter_scope"] in {"process", "system", "unavailable"}
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


def test_report_can_use_system_disk_scope_without_process_io():
    c = MetricsCollector("io-scope-test", sample_interval_sec=0.05)
    c._start_time = time.perf_counter() - 1.0
    c._end_time = time.perf_counter()
    c._io_scope = "system"
    c._io_start = SimpleNamespace(read_bytes=100, write_bytes=200)
    c._samples = [{"read_bytes": 140, "write_bytes": 320}]
    r = c.report()
    assert r["disk_counter_scope"] == "system"
    assert r["bytes_read_delta"] == 40
    assert r["bytes_written_delta"] == 120
    assert r["mb_written_delta"] >= 0.0


def test_parse_vllm_prometheus_metrics_keeps_activity_signals():
    text = """
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen"} 2
vllm:num_requests_waiting{model_name="Qwen"} 1
vllm:gpu_cache_usage_perc{model_name="Qwen",gpu="0"} 0.42
vllm:prompt_tokens_total{model_name="Qwen"} 128
vllm:generation_tokens_total{model_name="Qwen"} 256
vllm:request_queue_time_seconds_sum{model_name="Qwen"} 0.75
vllm:request_queue_time_seconds_count{model_name="Qwen"} 3
"""
    parsed = MetricsCollector._parse_prometheus_metrics(text)
    assert parsed["vllm_requests_running"] == 2.0
    assert parsed["vllm_requests_waiting"] == 1.0
    assert parsed["vllm_gpu_cache_usage_pct"] == 0.42
    assert parsed["vllm_prompt_tokens_total"] == 128.0
    assert parsed["vllm_generation_tokens_total"] == 256.0
    assert parsed["vllm_request_queue_time_seconds_sum"] == 0.75
    assert parsed["vllm_request_queue_time_seconds_count"] == 3.0


def test_report_includes_vllm_summary_fields_from_samples():
    c = MetricsCollector("vllm-report-test", sample_interval_sec=0.05)
    c._start_time = time.perf_counter() - 1.0
    c._end_time = time.perf_counter()
    c._samples = [
        {"vllm_gpu_cache_usage_pct": 0.25, "vllm_requests_running": 1.0},
        {
            "vllm_gpu_cache_usage_pct": 0.75,
            "vllm_requests_running": 3.0,
            "vllm_requests_waiting": 2.0,
        },
    ]
    r = c.report()
    assert r["avg_vllm_gpu_cache_usage_pct"] == 0.5
    assert r["peak_vllm_gpu_cache_usage_pct"] == 0.75
    assert r["peak_vllm_requests_running"] == 3.0
    assert r["peak_vllm_requests_waiting"] == 2.0
