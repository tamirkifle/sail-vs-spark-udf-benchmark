from __future__ import annotations

import json
from pathlib import Path

import analysis.aggregate_results as aggregate_results


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def test_aggregate_uses_manifest_artifact_paths_and_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(aggregate_results, "_run_plot_scripts", lambda *_: None)

    trace_path = results_dir / "custom_trace_name.json"
    stats_path = results_dir / "custom_stats_name.json"
    manifest_path = results_dir / "w2_A_s1_manifest.json"

    _write_json(
        trace_path,
        {
            "traceEvents": [
                {"name": "UDF_BATCH_EXECUTION", "dur": 2_000_000},
                {"name": "DATA_TRANSFER_IN", "dur": 500_000},
                {"name": "INFERENCE", "dur": 250_000},
            ]
        },
    )
    _write_json(
        stats_path,
        {
            "wall_clock_sec": 4.0,
            "n_samples": 3,
            "sample_interval_sec": 0.5,
            "avg_cpu_pct": 12.5,
            "peak_rss_mb": 321.0,
            "avg_rss_mb": 300.0,
            "peak_host_ram_gb": 7.5,
            "avg_gpu_util_pct": 45.0,
            "peak_gpu_util_pct": 80.0,
            "avg_gpu_mem_util_pct": 35.0,
            "peak_gpu_mem_used_mb": 1024.0,
            "avg_gpu_power_w": 60.0,
            "pipeline_continuity": 0.75,
            "bytes_read_delta": 123,
            "bytes_written_delta": 4_000_000,
            "mb_written_delta": 4.0,
            "write_throughput_mb_s": 1.0,
            "output_rows": 100,
            "samples": [
                {"t_sec": 0.0, "cpu_pct": 10.0, "rss_mb": 290.0, "host_ram_pct": 50.0},
                {"t_sec": 0.5, "cpu_pct": 15.0, "rss_mb": 300.0, "host_ram_pct": 55.0},
            ],
        },
    )
    _write_json(
        manifest_path,
        {
            "run_id": "w2_A_s1",
            "workload": "w2",
            "execution": "A",
            "setup_description": "Spark (Row/Pickle)",
            "depth": None,
            "profile": "test",
            "device": "cpu",
            "dataset_rows": 100,
            "models": {
                "generator": {"name": "g-model"},
                "scorer": {"name": "s-model"},
                "embedder": {"name": "e-model"},
            },
            "hardware_details": {
                "cpu_cores": 8,
                "total_ram_gb": 16.0,
                "resolved_device": "cpu",
            },
            "stats_json": str(stats_path),
            "trace_json": str(trace_path),
            "output_rows": 100,
            "wall_clock_sec": 4.0,
        },
    )

    summary = aggregate_results.aggregate(str(results_dir))

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["Workload"] == "W2"
    assert row["Config"] == "A"
    assert row["UDF Time (s)"] == 2.0
    assert row["Transfer Time (s)"] == 0.5
    assert row["Trace Compute (s)"] == 0.25
    assert row["Peak RSS (MB)"] == 321.0
    assert row["Disk Write (MB)"] == 4.0
    assert row["Trace Events"] == 3

    report_dir = results_dir / "report"
    assert (report_dir / "aggregate_runs.csv").exists()
    assert (report_dir / "aggregate_summary.csv").exists()
    assert (report_dir / "aggregate_summary.json").exists()
    assert (report_dir / "aggregate.md").exists()
    assert (report_dir / "aggregate.html").exists()

    html = (report_dir / "aggregate.html").read_text()
    assert 'id="overhead-breakdown-chart"' in html
    assert 'id="overhead-breakdown-data"' in html
    assert 'id="memory-comparison-chart"' in html
    assert 'id="memory-comparison-data"' in html
    assert 'id="relative-speedups-chart"' in html
    assert 'id="relative-speedups-data"' in html
    assert "Overhead Breakdown" in html
    assert "Memory Comparison" in html
    assert "Relative Speedups" in html
    assert "Spark's batched path" in html


def test_aggregate_reads_per_run_artifact_folders(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    run_dir = results_dir / "runs" / "w0_A_s1"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(aggregate_results, "_run_plot_scripts", lambda *_: None)

    _write_json(run_dir / "trace.json", {"traceEvents": []})
    _write_json(
        run_dir / "stats.json",
        {
            "wall_clock_sec": 1.0,
            "output_rows": 10,
            "samples": [],
        },
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": "w0_A_s1",
            "workload": "w0",
            "execution": "A",
            "depth": 1,
            "stats_json": str(run_dir / "stats.json"),
            "trace_json": str(run_dir / "trace.json"),
            "wall_clock_sec": 1.0,
            "output_rows": 10,
        },
    )

    summary = aggregate_results.aggregate(str(results_dir))

    assert len(summary) == 1
    assert (results_dir / "report" / "aggregate.html").exists()
