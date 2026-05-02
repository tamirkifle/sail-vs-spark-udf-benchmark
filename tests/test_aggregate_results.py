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
                {"name": "UDF_BATCH_EXECUTION", "ts": 10_600_000, "dur": 2_000_000},
                {"name": "DATA_TRANSFER_IN", "ts": 10_100_000, "dur": 500_000},
                {"name": "INFERENCE", "ts": 12_700_000, "dur": 250_000},
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
            "run_start_wall_ts": 10.0,
            "run_end_wall_ts": 14.0,
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
    assert row["Measured Runtime Writes (MB)"] == 4.0
    assert row["Output Materialized (MB)"] == 0.0
    assert row["Disk Metric Kind"] == "runtime_writes"
    assert row["Trace Events"] == 3
    run_df = aggregate_results._build_run_df(results_dir)
    run_row = run_df.iloc[0]
    assert run_row["PreTrace_sec"] == 0.1
    assert run_row["TraceWindow_sec"] == 2.85
    assert run_row["UntracedInWindow_sec"] == 0.1
    assert run_row["PostTrace_sec"] == 1.05
    assert run_row["InferenceTime_sec"] == 0.25
    assert run_row["UntracedEngineRuntime_sec"] == 0.25
    assert run_row["BoundaryTax_sec"] == 3.5
    assert run_row["MeasuredDiskWrite_MB"] == 4.0
    assert run_row["MeasuredDiskScope"] == "process"
    assert run_row["OutputMaterialized_MB"] == 0.0
    assert run_row["DiskWriteSource"] == "measured"
    assert bool(run_row["DiskTelemetryAvailable"]) is True

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
    assert 'id="disk-io-chart"' in html
    assert 'id="disk-io-data"' in html
    assert "Runtime Budget Breakdown" in html
    assert "Disk Materialization &amp; IO" in html
    assert "Memory Comparison" in html
    assert "Relative Speedups" in html
    assert "Spark's batched path" in html
    assert "Serialization vs Compute" not in html
    assert "Direct conversion / transfer" in html
    assert "Derived boundary tax" in html
    assert "full boundary tax is broader than the direct conversion bar alone" in html
    assert "Small traced compute in mock-model CPU runs means little compute was performed" in html
    assert "Startup / dispatch" in html
    assert "Untraced active runtime" in html
    assert "Tail / materialization" in html
    assert ".overhead-startup { background: #7ea6d8; }" in html
    assert ".overhead-gap { background: #d38c6a; }" in html
    assert ".overhead-tail { background: #a993cf; }" in html
    assert 'startup: "#7ea6d8"' in html
    assert 'activeGap: "#d38c6a"' in html
    assert 'tail: "#a993cf"' in html
    assert "Pattern-filled bars indicate fallback output-footprint values" in html
    assert "disk_io.png" not in html


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


def test_aggregate_falls_back_to_output_materialization_without_runtime_disk_telemetry(
    tmp_path: Path, monkeypatch
) -> None:
    results_dir = tmp_path / "results"
    run_dir = results_dir / "runs" / "w1_C_s1"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(aggregate_results, "_run_plot_scripts", lambda *_: None)

    output_dir = run_dir / "output.parquet"
    output_dir.mkdir()
    payload = b"x" * 4096
    (output_dir / "part-0000.parquet").write_bytes(payload)

    _write_json(run_dir / "trace.json", {"traceEvents": []})
    _write_json(
        run_dir / "stats.json",
        {
            "wall_clock_sec": 0.5,
            "bytes_written_delta": 0,
            "mb_written_delta": 0.0,
            "write_throughput_mb_s": 0.0,
            "output_rows": 100,
            "samples": [],
        },
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": "w1_C_s1",
            "workload": "w1",
            "execution": "C",
            "stats_json": str(run_dir / "stats.json"),
            "trace_json": str(run_dir / "trace.json"),
            "output_parquet": str(output_dir),
            "output_rows": 100,
            "wall_clock_sec": 0.5,
        },
    )

    summary = aggregate_results.aggregate(str(results_dir))
    row = summary.iloc[0]
    assert row["Disk Write (MB)"] == round(len(payload) / 1e6, 3)
    assert row["Measured Runtime Writes (MB)"] == 0.0
    assert row["Output Materialized (MB)"] == round(len(payload) / 1e6, 3)
    assert row["Disk Metric Kind"] == "output_materialization"

    run_df = aggregate_results._build_run_df(results_dir)
    run_row = run_df.iloc[0]
    assert run_row["MeasuredDiskWrite_MB"] == 0.0
    assert run_row["MeasuredDiskScope"] == "unavailable"
    assert run_row["OutputMaterialized_MB"] == round(len(payload) / 1e6, 3)
    assert run_row["DiskWriteSource"] == "output_artifact_fallback"
    assert bool(run_row["DiskTelemetryAvailable"]) is False
    assert run_row["WriteThroughput_MBps"] == 0.0

    html = (results_dir / "report" / "aggregate.html").read_text()
    assert "Measured runtime write telemetry was unavailable" in html


def test_summarize_trace_splits_detailed_phases_and_window(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    _write_json(
        trace_path,
        {
            "traceEvents": [
                {"name": "DATA_TRANSFER_IN", "ts": 1_000_000, "dur": 200_000},
                {"name": "UDF_BATCH_EXECUTION", "ts": 1_250_000, "dur": 900_000},
                {"name": "MODEL_LOAD", "ts": 1_300_000, "dur": 100_000},
                {"name": "TOKENIZE", "ts": 1_420_000, "dur": 80_000},
                {"name": "INFERENCE", "ts": 1_520_000, "dur": 200_000},
                {"name": "DETOKENIZE", "ts": 1_760_000, "dur": 50_000},
                {"name": "OTHER", "ts": 1_900_000, "dur": 40_000},
            ]
        },
    )
    summary = aggregate_results._summarize_trace(trace_path)
    assert summary.transfer_time_sec == 0.2
    assert summary.udf_time_sec == 0.9
    assert summary.model_load_time_sec == 0.1
    assert summary.tokenize_time_sec == 0.08
    assert summary.inference_time_sec == 0.2
    assert summary.detokenize_time_sec == 0.05
    assert summary.other_time_sec == 0.04
    assert summary.compute_time_sec == 0.47
    assert summary.trace_window_sec == 1.15
    assert summary.untraced_in_window_sec == 0.0
