"""Aggregate benchmark artifacts into tabular and HTML summaries."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import webbrowser
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template



EXECUTION_LABELS = {
    "A": "Spark (Row/Pickle)",
    "B": "Spark (Pandas/Arrow)",
    "C": "Sail (Zero-Copy)",
    "D": "Sail (SQL-Native)",
}

WORKLOAD_DESCRIPTIONS = {
    "W0": "Repeated trivial transforms at configurable depth. Isolates pure orchestration overhead and quantifies the cost of repeating the same engine–Python boundary crossing multiple times.",
    "W1": "Best-of-N generation with reward-model scoring. N candidates are generated per prompt and ranked; the highest-scoring response is returned.",
    "W2": "Repeated batched generation across a fixed prompt corpus. Measures sustained throughput under steady-state engine and GPU load.",
    "W3": "Batch embedding with vector similarity computation. Exercises model calls that return dense numerical outputs rather than token sequences.",
    "W4": "Multi-step agentic loop: generate, evaluate, and conditionally repeat until a quality threshold is met. Models iterative inference flows with variable iteration count.",
}

WORKLOAD_LABELS = {
    "W0": "Chained Transforms",
    "W1": "Best-of-N Generation",
    "W2": "Batched Generation",
    "W3": "Embedding Pipeline",
    "W4": "Agentic Loop",
}

CONFIG_DESCRIPTIONS = {
    "A": "Row-level PySpark UDF. Each row is serialized via Python pickle, dispatched to a Python worker process, and the result is deserialized back to the executor. Maximum boundary-crossing frequency.",
    "B": "Pandas UDF with Arrow batch serialization. Reduces crossing frequency by grouping rows into columnar batches, but data still transits a socket boundary between executor and Python worker.",
    "C": "Sail Arrow path. Data remains in columnar Arrow format across the engine–Python boundary, eliminating socket hops and reducing serialization to a format-compatible handoff.",
    "D": "Sail SQL-native UDTF. Model invocation is expressed as a table function within the query plan, allowing the engine to manage batching and minimizing the Python mediation surface.",
}

WORKLOAD_SPECS = {
    "W0": ["Input rows", "+1 transform", "Repeat by depth", "Output rows"],
    "W1": ["Prompt", "Generate N candidates", "Score candidates", "Pick best", "Best response"],
    "W2": ["Prompt batch", "Batch generation", "Collect responses", "Output parquet"],
    "W3": ["Prompt batch", "Embed text", "Similarity / vector work", "Scored output"],
    "W4": ["Prompt", "Generate", "Evaluate / reward", "Repeat if needed", "Final answer"],
}

CONFIG_SPECS = {
    "A": ["Spark executor", "Row UDF call", "Python worker", "Return one row", "Back to Spark"],
    "B": ["Spark executor", "Arrow batch", "Socket hop", "Pandas UDF", "Arrow batch back", "Back to Spark"],
    "C": ["Sail engine", "Arrow batch", "Python batch apply", "Shared-memory output", "Sail engine"],
    "D": ["Sail engine", "SQL-native UDTF", "Buffered Python batch", "Rows back to engine", "Sail engine"],
}

TRACE_PHASES_OF_INTEREST = {
    "UDF_BATCH_EXECUTION",
    "UDF_ROW_EXECUTION",
    "DATA_TRANSFER_IN",
    "DATA_TRANSFER_OUT",
    "MODEL_LOAD",
    "INFERENCE",
    "SCORE",
    "EMBED",
    "SIMILARITY",
    "TOKENIZE",
    "DETOKENIZE",
    "TRIVIAL_COMPUTE",
    "OTHER",
}

TRACE_COMPUTE_PHASES = {
    "MODEL_LOAD",
    "INFERENCE",
    "SCORE",
    "EMBED",
    "SIMILARITY",
    "TOKENIZE",
    "DETOKENIZE",
    "TRIVIAL_COMPUTE",
    "OTHER",
}

PLOT_SCRIPTS: list[tuple[str, str]] = []

SAIL_SVG = """
<svg width="28" height="28" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <g fill="#3762e0">
    <path d="M108.833 29L57.5627 129.813H108.833V29Z" />
    <path d="M157.503 158.651L57.5627 173L29 144.312L163.219 144.581L157.503 158.651Z" />
    <path d="M124.662 57.8778V129.813H171.794C126.568 100.098 124.662 57.8778 124.662 57.8778Z" />
  </g>
</svg>
""".strip()

SPARK_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="-7.872 -3.87 66.541 66.28" aria-hidden="true">
  <g fill="#e25a1c" fill-rule="evenodd">
    <path d="M42.915 42.1l-.303-.644-6.62-12.55c-.224-.42-.196-.67.1-1.027l10.473-12.3c.122-.144.233-.298.28-.554l-3.058.8-12.7 3.375c-.394.106-.573-.01-.77-.336L23.105 6.848c-.125-.208-.26-.4-.525-.585l-.582 3.207-2.013 11.33-.208 1.224c-.034.4-.234.534-.586.645L4.24 27.394c-.22.07-.432.158-.667.372l12.3 4.884-.36.283L7.86 37.88a.81.81 0 0 1-.887.072l-9.146-4.076c-1.368-.6-2.6-1.423-3.558-2.595-2.167-2.65-1.74-5.667 1.148-7.5.945-.603 2.018-1.055 3.088-1.404l14.686-4.665c.4-.128.6-.312.677-.757l2-11.33c.364-2.018.558-4.1 1.54-5.938.377-.7.83-1.408 1.367-2.004 1.945-2.16 4.66-2.242 6.716-.186.694.694 1.3 1.513 1.807 2.353L34 10.986c.262.44.5.53.984.4L51.4 7.025c1.128-.298 2.27-.407 3.427-.2 2.52.472 3.623 2.4 2.77 4.826-.388 1.1-1.058 2.047-1.8 2.932L44.34 28.05c-.313.366-.32.63-.103 1.04l6.822 12.934c.544 1.032.96 2.103.97 3.288.027 2.696-1.944 4.902-4.623 5.294-1.5.22-2.894-.1-4.3-.534l-10.3-3.133c-.318-.095-.44-.22-.496-.563l-1.242-7.24c-.012-.066.008-.138.018-.286l11.82 3.26" />
    <path d="M15.863 32.65l2.18.95L15.62 52.9l13.244-14.68 2.233.627 2.75 8.36-3.574-1.26-9.248 9.984s-4.136 4.282-6.378 3.56-4.134-2.72-4.344-6.292-1.156-13.77-1.156-13.77l-4.52-3.258z" />
    <path d="M9.146 39.43S7.15 61.503 11.038 62.238s-.42-.105-.42-.105 2.207 2.523 10.93-6.832l8.724-9.354-17.448 10.3c.42-.526 2.207-18.394 2.207-18.394z" />
  </g>
</svg>
""".strip()


@dataclass(frozen=True)
class TraceSummary:
    udf_time_sec: float
    transfer_time_sec: float
    compute_time_sec: float
    model_load_time_sec: float
    tokenize_time_sec: float
    inference_time_sec: float
    detokenize_time_sec: float
    score_time_sec: float
    embed_time_sec: float
    similarity_time_sec: float
    trivial_compute_time_sec: float
    other_time_sec: float
    trace_event_count: int
    trace_start_ts: float | None
    trace_end_ts: float | None
    trace_window_sec: float
    untraced_in_window_sec: float


def get_label(cfg: str) -> str:
    return EXECUTION_LABELS.get(cfg, cfg)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def _parse_sample_idx(run_id: str) -> int:
    match = re.search(r"_s(\d+)$", run_id)
    return int(match.group(1)) if match else 1


def _resolve_artifact_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    field_name: str,
    fallback_suffix: str,
) -> Path | None:
    raw = manifest.get(field_name)
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            if candidate.exists():
                return candidate
            candidate = manifest_path.parent / candidate.name
        if candidate.exists():
            return candidate

    fallback = manifest_path.with_name(
        manifest_path.name.replace("_manifest.json", fallback_suffix)
    )
    return fallback if fallback.exists() else None


def _summarize_trace(trace_path: Path | None) -> TraceSummary:
    if trace_path is None or not trace_path.exists():
        return TraceSummary(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, None, None, 0.0, 0.0
        )

    try:
        trace_data = _read_json(trace_path)
    except Exception:
        return TraceSummary(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, None, None, 0.0, 0.0
        )

    udf_time_sec = 0.0
    transfer_time_sec = 0.0
    compute_time_sec = 0.0
    phase_totals = {phase: 0.0 for phase in TRACE_COMPUTE_PHASES}
    trace_event_count = 0
    trace_start_ts: float | None = None
    trace_end_ts: float | None = None

    for event in trace_data.get("traceEvents", []):
        phase = event.get("name")
        if phase not in TRACE_PHASES_OF_INTEREST:
            continue
        dur_sec = float(event.get("dur", 0.0)) / 1_000_000.0
        ts_raw = event.get("ts")
        if ts_raw is not None:
            start_ts = float(ts_raw)
            end_ts = start_ts + float(event.get("dur", 0.0))
            trace_start_ts = start_ts if trace_start_ts is None else min(trace_start_ts, start_ts)
            trace_end_ts = end_ts if trace_end_ts is None else max(trace_end_ts, end_ts)
        trace_event_count += 1
        if phase in {"UDF_BATCH_EXECUTION", "UDF_ROW_EXECUTION"}:
            udf_time_sec += dur_sec
        elif phase in {"DATA_TRANSFER_IN", "DATA_TRANSFER_OUT"}:
            transfer_time_sec += dur_sec
        else:
            compute_time_sec += dur_sec
            phase_totals[phase] += dur_sec

    trace_window_sec = 0.0
    if trace_start_ts is not None and trace_end_ts is not None:
        trace_window_sec = max(0.0, (trace_end_ts - trace_start_ts) / 1_000_000.0)
    traced_sum_sec = udf_time_sec + transfer_time_sec + compute_time_sec
    untraced_in_window_sec = max(0.0, trace_window_sec - traced_sum_sec)

    return TraceSummary(
        udf_time_sec=round(udf_time_sec, 6),
        transfer_time_sec=round(transfer_time_sec, 6),
        compute_time_sec=round(compute_time_sec, 6),
        model_load_time_sec=round(phase_totals["MODEL_LOAD"], 6),
        tokenize_time_sec=round(phase_totals["TOKENIZE"], 6),
        inference_time_sec=round(phase_totals["INFERENCE"], 6),
        detokenize_time_sec=round(phase_totals["DETOKENIZE"], 6),
        score_time_sec=round(phase_totals["SCORE"], 6),
        embed_time_sec=round(phase_totals["EMBED"], 6),
        similarity_time_sec=round(phase_totals["SIMILARITY"], 6),
        trivial_compute_time_sec=round(phase_totals["TRIVIAL_COMPUTE"], 6),
        other_time_sec=round(phase_totals["OTHER"], 6),
        trace_event_count=trace_event_count,
        trace_start_ts=trace_start_ts,
        trace_end_ts=trace_end_ts,
        trace_window_sec=round(trace_window_sec, 6),
        untraced_in_window_sec=round(untraced_in_window_sec, 6),
    )


def _sample_metric(samples: list[dict[str, Any]], key: str, fn) -> float:
    values = [float(s[key]) for s in samples if s.get(key) is not None]
    return round(fn(values), 3) if values else 0.0


def _path_size_bytes(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return sum(int(child.stat().st_size) for child in path.rglob("*") if child.is_file())


def _build_run_record(manifest_path: Path) -> dict[str, Any] | None:
    manifest = _read_json(manifest_path)
    stats_path = _resolve_artifact_path(
        manifest_path, manifest, "stats_json", "_stats.json"
    )
    if stats_path is None:
        return None
    stats = _read_json(stats_path)
    trace_path = _resolve_artifact_path(
        manifest_path, manifest, "trace_json", "_trace.json"
    )
    trace = _summarize_trace(trace_path)

    samples = stats.get("samples") or []
    hardware = manifest.get("hardware_details") or {}
    models = manifest.get("models") or {}
    output_parquet_path = None
    if manifest.get("output_parquet"):
        output_parquet_path = _resolve_artifact_path(
            manifest_path, manifest, "output_parquet", "_output.parquet"
        )

    wall_time = float(stats.get("wall_clock_sec", manifest.get("wall_clock_sec", 0.0)) or 0.0)
    run_start_wall_ts = float(stats.get("run_start_wall_ts", 0.0) or 0.0)
    run_end_wall_ts = float(stats.get("run_end_wall_ts", 0.0) or 0.0)
    pre_trace_sec = 0.0
    post_trace_sec = 0.0
    if trace.trace_start_ts is not None and run_start_wall_ts > 0:
        pre_trace_sec = max(0.0, trace.trace_start_ts / 1_000_000.0 - run_start_wall_ts)
    if trace.trace_end_ts is not None and run_end_wall_ts > 0:
        post_trace_sec = max(0.0, run_end_wall_ts - trace.trace_end_ts / 1_000_000.0)
    if trace.trace_event_count == 0:
        pre_trace_sec = 0.0
        post_trace_sec = 0.0
    remaining_wall = max(0.0, wall_time - pre_trace_sec - post_trace_sec)
    trace_window_sec = min(remaining_wall, trace.trace_window_sec)
    worker_wrapper_sec = max(0.0, trace.udf_time_sec - trace.compute_time_sec)
    residual_sec = max(
        0.0,
        wall_time
        - trace.transfer_time_sec
        - trace.compute_time_sec
        - worker_wrapper_sec
        - pre_trace_sec
        - post_trace_sec
        - trace.untraced_in_window_sec,
    )
    untraced_engine_runtime_sec = residual_sec
    udf_share = (trace.udf_time_sec / wall_time * 100.0) if wall_time > 0 else 0.0
    transfer_share = (trace.transfer_time_sec / wall_time * 100.0) if wall_time > 0 else 0.0
    compute_share = (trace.compute_time_sec / wall_time * 100.0) if wall_time > 0 else 0.0
    bytes_read_delta = int(stats.get("bytes_read_delta", 0) or 0)
    disk_counter_scope = str(stats.get("disk_counter_scope", "unavailable") or "unavailable")
    measured_bytes_written = int(stats.get("bytes_written_delta", 0) or 0)
    if disk_counter_scope == "unavailable" and measured_bytes_written > 0:
        disk_counter_scope = "process"
    measured_mb_written = float(stats.get("mb_written_delta", 0.0) or 0.0)
    output_materialized_bytes = int(
        stats.get(
            "output_materialized_bytes",
            manifest.get("output_materialized_bytes", 0),
        )
        or 0
    )
    if output_materialized_bytes <= 0:
        output_materialized_bytes = _path_size_bytes(output_parquet_path)
    output_materialized_mb = float(
        stats.get(
            "output_materialized_mb",
            manifest.get("output_materialized_mb", 0.0),
        )
        or 0.0
    )
    if output_materialized_mb <= 0.0 and output_materialized_bytes > 0:
        output_materialized_mb = round(output_materialized_bytes / 1e6, 3)
    disk_telemetry_available = disk_counter_scope != "unavailable"
    if disk_telemetry_available and measured_bytes_written > 0:
        bytes_written_delta = measured_bytes_written
        mb_written_delta = (
            measured_mb_written
            if measured_mb_written > 0.0
            else round(measured_bytes_written / 1e6, 3)
        )
        disk_write_source = "measured"
    elif output_materialized_bytes > 0:
        bytes_written_delta = output_materialized_bytes
        mb_written_delta = output_materialized_mb
        disk_write_source = "output_artifact_fallback"
    else:
        bytes_written_delta = 0
        mb_written_delta = 0.0
        disk_write_source = "none"
    write_throughput_mb_s = float(stats.get("write_throughput_mb_s", 0.0) or 0.0)
    pipeline_continuity = _float_or_none(stats.get("pipeline_continuity"))

    return {
        "RunID": manifest["run_id"],
        "Workload": str(manifest["workload"]).upper(),
        "Config": manifest["execution"],
        "Label": get_label(manifest["execution"]),
        "Depth": manifest.get("depth"),
        "SampleIdx": _parse_sample_idx(manifest["run_id"]),
        "WallTime": wall_time,
        "Rows": int(stats.get("output_rows", manifest.get("output_rows", 0)) or 0),
        "DatasetRows": manifest.get("dataset_rows"),
        "SetupDescription": manifest.get("setup_description", ""),
        "Profile": manifest.get("profile"),
        "DeviceRequested": manifest.get("device"),
        "DeviceResolved": hardware.get("resolved_device"),
        "CpuCores": hardware.get("cpu_cores"),
        "TotalRamGB": hardware.get("total_ram_gb"),
        "GpuModels": "; ".join(hardware.get("gpu_models", [])) if hardware.get("gpu_models") else "",
        "GeneratorModel": (models.get("generator") or {}).get("name", ""),
        "ScorerModel": (models.get("scorer") or {}).get("name", ""),
        "EmbedderModel": (models.get("embedder") or {}).get("name", ""),
        "AvgCPU_pct": float(stats.get("avg_cpu_pct", 0.0) or 0.0),
        "PeakRSS_MB": float(stats.get("peak_rss_mb", 0.0) or 0.0),
        "AvgRSS_MB": float(stats.get("avg_rss_mb", 0.0) or 0.0),
        "AvgProcessTreeCPU_pct": float(stats.get("avg_process_tree_cpu_pct", 0.0) or 0.0),
        "PeakProcessTreeRSS_MB": float(stats.get("peak_process_tree_rss_mb", 0.0) or 0.0),
        "AvgProcessTreeRSS_MB": float(stats.get("avg_process_tree_rss_mb", 0.0) or 0.0),
        "SampledChildProcesses": int(stats.get("sampled_child_processes", 0) or 0),
        "PeakHostRam_GB": float(stats.get("peak_host_ram_gb", 0.0) or 0.0),
        "AvgHostRam_pct": _sample_metric(samples, "host_ram_pct", lambda vs: sum(vs) / len(vs)),
        "PeakHostRam_pct": _sample_metric(samples, "host_ram_pct", max),
        "GpuTelemetryAvailable": bool(stats.get("gpu_telemetry_available", False)),
        "AvgGPUUtil_pct": float(stats.get("avg_gpu_util_pct", 0.0) or 0.0),
        "PeakGPUUtil_pct": float(stats.get("peak_gpu_util_pct", 0.0) or 0.0),
        "AvgGPUMemUtil_pct": float(stats.get("avg_gpu_mem_util_pct", 0.0) or 0.0),
        "PeakGPUMemUsed_MB": float(stats.get("peak_gpu_mem_used_mb", 0.0) or 0.0),
        "AvgGPUPower_W": float(stats.get("avg_gpu_power_w", 0.0) or 0.0),
        "PipelineContinuityAvailable": bool(stats.get("pipeline_continuity_available", pipeline_continuity is not None)),
        "PipelineContinuity": pipeline_continuity,
        "VllmTelemetryAvailable": bool(stats.get("vllm_telemetry_available", False)),
        "AvgVLLMGPUCacheUsage_pct": float(stats.get("avg_vllm_gpu_cache_usage_pct", 0.0) or 0.0),
        "PeakVLLMGPUCacheUsage_pct": float(stats.get("peak_vllm_gpu_cache_usage_pct", 0.0) or 0.0),
        "PeakVLLMRequestsRunning": float(stats.get("peak_vllm_requests_running", 0.0) or 0.0),
        "PeakVLLMRequestsWaiting": float(stats.get("peak_vllm_requests_waiting", 0.0) or 0.0),
        "MeasuredDiskRead_Bytes": bytes_read_delta,
        "MeasuredDiskRead_MB": round(bytes_read_delta / 1e6, 3),
        "MeasuredDiskWrite_Bytes": measured_bytes_written,
        "MeasuredDiskWrite_MB": round(
            measured_mb_written if measured_mb_written > 0.0 else measured_bytes_written / 1e6,
            3,
        ) if measured_bytes_written > 0 else 0.0,
        "MeasuredDiskScope": disk_counter_scope,
        "OutputMaterialized_Bytes": output_materialized_bytes,
        "OutputMaterialized_MB": round(output_materialized_mb, 3),
        "DiskTelemetryAvailable": bool(disk_telemetry_available),
        "DiskWriteSource": disk_write_source,
        "BytesReadDelta": bytes_read_delta,
        "BytesWrittenDelta": bytes_written_delta,
        "DiskRead_MB": round(bytes_read_delta / 1e6, 3),
        "DiskWrite_MB": mb_written_delta,
        "WriteThroughput_MBps": write_throughput_mb_s,
        "CollectorSamples": int(stats.get("n_samples", 0) or 0),
        "CollectorInterval_sec": float(stats.get("sample_interval_sec", 0.0) or 0.0),
        "NvidiaDmonLog": stats.get("nvidia_dmon_log"),
        "TracePath": str(trace_path) if trace_path else "",
        "TraceEventCount": trace.trace_event_count,
        "UDFTime_sec": trace.udf_time_sec,
        "TransferTime_sec": trace.transfer_time_sec,
        "TraceComputeTime_sec": trace.compute_time_sec,
        "ModelLoadTime_sec": trace.model_load_time_sec,
        "TokenizeTime_sec": trace.tokenize_time_sec,
        "InferenceTime_sec": trace.inference_time_sec,
        "DetokenizeTime_sec": trace.detokenize_time_sec,
        "ScoreTime_sec": trace.score_time_sec,
        "EmbedTime_sec": trace.embed_time_sec,
        "SimilarityTime_sec": trace.similarity_time_sec,
        "TrivialComputeTime_sec": trace.trivial_compute_time_sec,
        "OtherTraceTime_sec": trace.other_time_sec,
        "WorkerWrapperTime_sec": round(worker_wrapper_sec, 6),
        "PreTrace_sec": round(pre_trace_sec, 6),
        "TraceWindow_sec": round(trace_window_sec, 6),
        "UntracedInWindow_sec": trace.untraced_in_window_sec,
        "PostTrace_sec": round(post_trace_sec, 6),
        "UntracedEngineRuntime_sec": round(untraced_engine_runtime_sec, 6),
        "BoundaryTax_sec": round(
            trace.transfer_time_sec
            + worker_wrapper_sec
            + pre_trace_sec
            + trace.untraced_in_window_sec
            + post_trace_sec,
            6,
        ),
        "BoundaryTax_pct": round(
            (
                (
                    trace.transfer_time_sec
                    + worker_wrapper_sec
                    + pre_trace_sec
                    + trace.untraced_in_window_sec
                    + post_trace_sec
                )
                / wall_time
                * 100.0
            )
            if wall_time > 0
            else 0.0,
            3,
        ),
        "UDFShare_pct": round(udf_share, 3),
        "TransferShare_pct": round(transfer_share, 3),
        "TraceComputeShare_pct": round(compute_share, 3),
    }


def _manifest_paths(results_dir: Path) -> list[Path]:
    paths = set(results_dir.glob("*_manifest.json"))
    paths.update(results_dir.glob("runs/*/manifest.json"))
    return sorted(paths)


def _build_run_df(results_dir: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in _manifest_paths(results_dir):
        row = _build_run_record(manifest_path)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def _format_duration(mean: float, std: float) -> str:
    if pd.isna(std) or std < 0.001:
        return f"{mean:.3f}s"
    return f"{mean:.3f}s ±{std:.3f}"


def _build_summary_df(run_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = ["Workload", "Config", "Label", "Depth"]

    for keys, group in run_df.groupby(group_cols, dropna=False):
        group = group.sort_values("SampleIdx")
        cold = group[group["SampleIdx"] == 1]
        warm = group[group["SampleIdx"] > 1]
        has_warm = not warm.empty
        perf = warm if has_warm else group

        cold_wall = float(cold["WallTime"].iloc[0]) if not cold.empty else float(group["WallTime"].iloc[0])
        warm_mean = float(warm["WallTime"].mean()) if has_warm else float("nan")
        warm_std = float(warm["WallTime"].std()) if has_warm and len(warm) > 1 else 0.0
        perf_time = float(perf["WallTime"].mean())  # warm if available, else cold
        pipeline_continuity = _mean_or_none(group["PipelineContinuity"])
        udf_mean = float(perf["UDFTime_sec"].mean())
        transfer_mean = float(perf["TransferTime_sec"].mean())
        trace_compute_mean = float(perf["TraceComputeTime_sec"].mean())
        overhead_pct = max(0.0, ((perf_time - udf_mean) / perf_time * 100.0)) if perf_time > 0 else 0.0
        throughput = (float(perf["Rows"].mean()) / perf_time) if perf_time > 0 else 0.0
        disk_metric_kind = (
            "runtime_writes"
            if float(group["MeasuredDiskWrite_Bytes"].max()) > 0
            else "output_materialization"
        )

        summary_rows.append(
            {
                "Workload": keys[0],
                "Config": keys[1],
                "Setup": keys[2],
                "Depth": keys[3],
                "HasWarm": has_warm,
                "ColdWall_sec": cold_wall,
                "WarmMean_sec": warm_mean,
                "WarmStd_sec": warm_std,
                "Steady (Warm)": _format_duration(warm_mean, warm_std) if has_warm else "-",
                "Setup (Cold)": f"{cold_wall:.3f}s",
                "Rows": float(perf["Rows"].mean()),
                "RowsPerSec": round(throughput, 3),
                "UDF Time (s)": udf_mean,
                "Transfer Time (s)": transfer_mean,
                "Trace Compute (s)": trace_compute_mean,
                "Overhead Tax (%)": round(overhead_pct, 2),
                "UDF Share (%)": round(float(perf["UDFShare_pct"].mean()), 2),
                "Transfer Share (%)": round(float(perf["TransferShare_pct"].mean()), 2),
                "Peak RSS (MB)": round(float(group["PeakRSS_MB"].max()), 2),
                "Peak Process Tree RSS (MB)": round(float(group["PeakProcessTreeRSS_MB"].max()), 2),
                "Avg CPU (%)": round(float(group["AvgCPU_pct"].mean()), 2),
                "Avg Process Tree CPU (%)": round(float(group["AvgProcessTreeCPU_pct"].mean()), 2),
                "Peak GPU Util (%)": round(float(group["PeakGPUUtil_pct"].max()), 2),
                "GPU Telemetry Available": bool(group["GpuTelemetryAvailable"].any()),
                "Pipeline Continuity Available": bool(group["PipelineContinuityAvailable"].any()),
                "Pipeline Continuity": round(pipeline_continuity, 3) if pipeline_continuity is not None else None,
                "Avg GPU Power (W)": round(float(group["AvgGPUPower_W"].mean()), 2),
                "vLLM Telemetry Available": bool(group["VllmTelemetryAvailable"].any()),
                "Disk Write (MB)": round(float(group["DiskWrite_MB"].mean()), 3),
                "Measured Runtime Writes (MB)": round(float(group["MeasuredDiskWrite_MB"].mean()), 3),
                "Output Materialized (MB)": round(float(group["OutputMaterialized_MB"].mean()), 3),
                "Write Throughput (MB/s)": round(float(group["WriteThroughput_MBps"].mean()), 3),
                "Disk Metric Kind": disk_metric_kind,
                "Collector Samples": int(group["CollectorSamples"].max()),
                "Trace Events": int(group["TraceEventCount"].sum()),
                "Samples": int(len(group)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df

    baseline_config = "B"

    def _speedup(row: pd.Series) -> float:
        mask = (summary_df["Workload"] == row["Workload"]) & (summary_df["Config"] == baseline_config)
        if pd.isna(row["Depth"]):
            mask &= summary_df["Depth"].isna()
        else:
            mask &= summary_df["Depth"] == row["Depth"]
        base = summary_df[mask]
        if base.empty:
            return float("nan")
        row_has_warm = bool(row["HasWarm"])
        base_has_warm = bool(base["HasWarm"].iloc[0])
        if row_has_warm and base_has_warm:
            row_time = float(row["WarmMean_sec"])
            base_time = float(base["WarmMean_sec"].iloc[0])
        elif not row_has_warm and not base_has_warm:
            row_time = float(row["ColdWall_sec"])
            base_time = float(base["ColdWall_sec"].iloc[0])
        else:
            return float("nan")  # mixed warm/cold comparison — not comparable
        if row_time <= 0 or base_time <= 0:
            return float("nan")
        return base_time / row_time

    summary_df["Speedup_x"] = summary_df.apply(_speedup, axis=1)
    summary_df["Speedup"] = summary_df["Speedup_x"].map(
        lambda v: f"{v:.2f}x" if pd.notna(v) else "-"
    )
    summary_df["UDF Time"] = summary_df["UDF Time (s)"].map(lambda v: f"{v:.3f}s" if v > 0 else "-")
    summary_df["Transfer Time"] = summary_df["Transfer Time (s)"].map(lambda v: f"{v:.3f}s" if v > 0 else "-")
    summary_df["DepthSort"] = summary_df["Depth"].fillna(-1)
    summary_df = summary_df.sort_values(["Workload", "DepthSort", "Config"]).drop(columns=["DepthSort"])
    return summary_df


def _write_csvs(report_dir: Path, run_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(report_dir / "aggregate_runs.csv", index=False)
    summary_df.to_csv(report_dir / "aggregate_summary.csv", index=False)
    with open(report_dir / "aggregate_summary.json", "w") as fh:
        json.dump(summary_df.to_dict(orient="records"), fh, indent=2)


def _write_markdown(report_dir: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Sail vs Spark Benchmark Summary",
        "",
        "This report is derived from manifests, stats JSON, and trace JSON files.",
        "",
    ]

    for workload in sorted(summary_df["Workload"].unique()):
        lines.append(f"## Workload {workload}")
        lines.append("")
        df = summary_df[summary_df["Workload"] == workload].copy()
        df["Depth"] = df["Depth"].apply(lambda x: int(x) if pd.notna(x) else "-")
        view = df[
            [
                "Setup",
                "Depth",
                "Setup (Cold)",
                "Steady (Warm)",
                "RowsPerSec",
                "UDF Time",
                "Transfer Time",
                "Overhead Tax (%)",
                "Speedup",
                "Peak RSS (MB)",
                "Disk Write (MB)",
                "Samples",
            ]
        ].rename(columns={"Overhead Tax (%)": "Overhead %"})
        lines.append(view.to_markdown(index=False))
        lines.append("")

    (report_dir / "aggregate.md").write_text("\n".join(lines))


def _logo_svg(config: str) -> str:
    return SAIL_SVG if config in {"C", "D"} else SPARK_SVG


def _row_class(config: str) -> str:
    return "sail-row" if config in {"C", "D"} else "spark-row"


def _speedup_class(config: str, speedup_x: float) -> str:
    sail_or_spark = 'sail' if config.lower() in {'c', 'd'} else 'spark'
    if pd.isna(speedup_x):
        return f"speedup speedup-{sail_or_spark}-sm"
    if speedup_x >= 50:
        return f"speedup speedup-{sail_or_spark}-xl"
    if speedup_x >= 10:
        return f"speedup speedup-{sail_or_spark}-lg"
    if speedup_x >= 2:
        return f"speedup speedup-{sail_or_spark}-md"
    return f"speedup speedup-{sail_or_spark}-sm"


def _mermaid_init(accent: str, edge_label_bg: str) -> str:
    return (
        "%%{init: {'theme': 'base', 'flowchart': {'nodeSpacing': 50, 'rankSpacing': 60}, 'themeVariables': {"
        "'primaryColor': '#ffffff',"
        "'primaryBorderColor': '" + accent + "',"
        "'primaryTextColor': '#0f172a',"
        "'lineColor': '" + accent + "',"
        "'edgeLabelBackground': '" + edge_label_bg + "',"
        "'fontSize': '18px'"
        "}}}%%"
    )


def _render_flow_mermaid(steps: list[str], accent: str, edge_label_bg: str) -> str:
    init = _mermaid_init(accent, edge_label_bg)
    node_ids = [chr(65 + i) for i in range(len(steps))]
    nodes = " --> ".join(f"{nid}([{step}]):::box" for nid, step in zip(node_ids, steps))
    body = (
        "flowchart LR\n"
        f"    classDef box fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    {nodes}"
    )
    return f'<div class="mermaid">\n{init}\n{body}\n</div>'


def _render_w1_mermaid() -> str:
    accent = "#3762e0"
    edge_label_bg = "#eff4ff"
    init = _mermaid_init(accent, edge_label_bg)
    body = (
        "flowchart LR\n"
        f"    classDef box  fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef gate fill:{edge_label_bg},stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef done fill:#dbeafe,stroke:{accent},stroke-width:2.8px,color:#0f172a,font-weight:700\n"
        "\n"
        "    P([Prompt]):::box --> G([Generate N candidates]):::box --> S([Score candidates]):::box --> D{Best candidate<br/>meets threshold?}:::gate\n"
        "    D -- yes --> F([Return best response]):::done\n"
        "    D -- no --> R([Regenerate / widen beam]):::box --> G\n"
    )
    return f'<div class="mermaid">\n{init}\n{body}\n</div>'


def _render_w0_mermaid() -> str:
    accent = "#3762e0"
    edge_label_bg = "#eff4ff"
    init = _mermaid_init(accent, edge_label_bg)
    body = (
        "flowchart LR\n"
        f"    classDef box fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef gate fill:{edge_label_bg},stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef done fill:#dbeafe,stroke:{accent},stroke-width:2.8px,color:#0f172a,font-weight:700\n"
        "\n"
        "    I([Input row]):::box --> S([Add 1]):::box --> O([Output row]):::done\n"
        "    D([Depth = repeat count]):::gate -.->|apply the same step N times| S\n"
    )
    return f'<div class="mermaid">\n{init}\n{body}\n</div>'


def _render_config_mermaid(code: str) -> str:
    palette = {
        "A": ("#e25a1c", "#fff7ed"),
        "B": ("#e25a1c", "#fff7ed"),
        "C": ("#3762e0", "#eff6ff"),
        "D": ("#3762e0", "#eff6ff"),
    }
    accent, edge_label_bg = palette[code]
    init = (
        "%%{init: {'theme': 'base', 'flowchart': {'nodeSpacing': 62, 'rankSpacing': 72}, 'themeVariables': {"
        "'primaryColor': '#ffffff',"
        "'primaryBorderColor': '" + accent + "',"
        "'primaryTextColor': '#0f172a',"
        "'lineColor': '" + accent + "',"
        "'edgeLabelBackground': '" + edge_label_bg + "',"
        "'fontSize': '17px'"
        "}}}%%"
    )
    if code == "A":
        body = (
            "flowchart LR\n"
            f"    classDef box  fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
            f"    classDef gate fill:{edge_label_bg},stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
            "\n"
            "    E([Spark executor]):::box --> R([Serialize row with pickle]):::box --> P([Python worker]):::box --> M([Run model per row]):::box --> O([Deserialize result]):::box --> E\n"
            "    R -.->|one row at a time| P\n"
        )
    elif code == "B":
        body = (
            "flowchart LR\n"
            f"    classDef box  fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
            f"    classDef gate fill:{edge_label_bg},stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
            "\n"
            "    E([Spark executor]):::box --> A([Bundle Arrow batches]):::box --> S([Socket hop to Python]):::box --> U([Pandas UDF]):::box --> B([Arrow batch return]):::box --> E\n"
            "    S -.->|batch boundary| U\n"
        )
    elif code == "C":
        body = (
            "flowchart LR\n"
            f"    classDef box  fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
            f"    classDef gate fill:{edge_label_bg},stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
            "\n"
            "    E([Sail engine]):::box --> A([Arrow batch]):::box --> H([Shared-memory handoff]):::box --> P([Python batch apply]):::box --> R([Zero-copy return]):::box --> E\n"
            "    H -.->|no socket hop| P\n"
        )
    else:
        body = (
            "flowchart LR\n"
            f"    classDef box  fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
            f"    classDef gate fill:{edge_label_bg},stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
            "\n"
            "    E([Sail engine]):::box --> Q([SQL plan / UDTF]):::box --> B([Buffered Python batch]):::box --> R([Rows back to engine]):::box --> E\n"
            "    Q -.->|engine orchestrates batching| B\n"
    )
    return f'<div class="mermaid">\n{init}\n{body}\n</div>'


def _render_w2_mermaid() -> str:
    accent = "#3762e0"
    edge_label_bg = "#eff4ff"
    init = _mermaid_init(accent, edge_label_bg)
    body = (
        "flowchart LR\n"
        f"    classDef box  fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef gate fill:{edge_label_bg},stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef done fill:#dbeafe,stroke:{accent},stroke-width:2.8px,color:#0f172a,font-weight:700\n"
        "\n"
        "    P([Prompt corpus]):::box --> B([Batch prompts]):::box --> G([Generate batched responses]):::box --> C([Collect response rows]):::box --> W([Write results]):::done\n"
        "    B -.->|fixed batch size| G\n"
        "    G -.->|single response per prompt| C\n"
    )
    return f'<div class="mermaid">\n{init}\n{body}\n</div>'


def _render_w3_mermaid() -> str:
    accent = "#3762e0"
    edge_label_bg = "#eff4ff"
    init = _mermaid_init(accent, edge_label_bg)
    body = (
        "flowchart LR\n"
        f"    classDef box  fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef gate fill:{edge_label_bg},stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef done fill:#dbeafe,stroke:{accent},stroke-width:2.8px,color:#0f172a,font-weight:700\n"
        "\n"
        "    P([Prompt batch]):::box --> E([Tokenize / embed]):::box --> S([Compare to reference examples]):::box --> A([Pick best match]):::box --> O([Best match result]):::done\n"
        "    R([Reference examples]):::box --> S\n"
        "    S -.->|find strongest match| A\n"
    )
    return f'<div class="mermaid">\n{init}\n{body}\n</div>'


def _render_agentic_mermaid() -> str:
    accent = "#3762e0"
    edge_label_bg = "#eff4ff"
    init = (
        "%%{init: {'theme': 'base', 'flowchart': {'nodeSpacing': 80, 'rankSpacing': 90}, 'themeVariables': {"
        "'primaryColor': '#ffffff',"
        "'primaryBorderColor': '" + accent + "',"
        "'primaryTextColor': '#0f172a',"
        "'lineColor': '" + accent + "',"
        "'edgeLabelBackground': '" + edge_label_bg + "',"
        "'fontSize': '19px'"
        "}}}%%"
    )
    body = (
        "flowchart LR\n"
        f"    classDef box  fill:#ffffff,stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef gate fill:{edge_label_bg},stroke:{accent},stroke-width:2.4px,color:#0f172a\n"
        f"    classDef done fill:#dbeafe,stroke:{accent},stroke-width:2.8px,color:#0f172a,font-weight:700\n"
        "\n"
        "    P([Prompt]):::box --> G([Generate]):::box --> E([Evaluate / reward]):::box --> D{Good enough?}:::gate\n"
        "    D -- yes --> F([Final answer]):::done\n"
        "    D -- no --> G"
    )
    return f'<div class="mermaid">\n{init}\n{body}\n</div>'


def _render_workload_svg(code: str) -> str:
    if code == "W0":
        return _render_w0_mermaid()
    if code == "W1":
        return _render_w1_mermaid()
    if code == "W2":
        return _render_w2_mermaid()
    if code == "W3":
        return _render_w3_mermaid()
    if code == "W4":
        return _render_agentic_mermaid()
    return _render_flow_mermaid(WORKLOAD_SPECS[code], accent="#3762e0", edge_label_bg="#eff4ff")


def _render_config_svg(code: str) -> str:
    return _render_config_mermaid(code)


def _load_report_context(results_dir: Path) -> dict[str, Any]:
    manifest_paths = _manifest_paths(results_dir)
    if not manifest_paths:
        return {}
    manifest = _read_json(manifest_paths[0])
    hardware = manifest.get("hardware_details") or {}
    models = manifest.get("models") or {}
    workloads = []
    for code in ["W0", "W1", "W2", "W3", "W4"]:
        workloads.append(
            {
                "code": code,
                "label": WORKLOAD_LABELS[code],
                "description": WORKLOAD_DESCRIPTIONS[code],
                "svg": _render_workload_svg(code),
            }
        )
    configs = []
    for code in ["A", "B", "C", "D"]:
        configs.append(
            {
                "code": code,
                "label": EXECUTION_LABELS[code],
                "description": CONFIG_DESCRIPTIONS[code],
                "logo": _logo_svg(code),
                "row_class": _row_class(code),
                "svg": _render_config_svg(code),
            }
        )
    run_specs = [
        ("Profile", manifest.get("profile") or "-"),
        ("Platform", manifest.get("platform") or "-"),
        ("Python", manifest.get("python") or "-"),
        ("Host", manifest.get("host") or "-"),
        ("Device Requested", manifest.get("device") or "-"),
        ("Device Resolved", hardware.get("resolved_device") or "-"),
        ("CPU Cores", hardware.get("cpu_cores") or "-"),
        ("Total RAM", f"{hardware.get('total_ram_gb')} GB" if hardware.get("total_ram_gb") else "-"),
        ("Dataset Rows", manifest.get("dataset_rows") or "-"),
        ("Partitions", manifest.get("num_partitions") or "-"),
    ]
    model_specs = []
    for key in ["generator", "scorer", "embedder"]:
        info = models.get(key) or {}
        model_specs.append({"role": key.title(), "name": info.get("name", "-"), "details": info})
    workload_knobs = []
    for key, cfg in (manifest.get("workload_config") or {}).items():
        title = key.replace("_", " ").title()
        badge = key.split("_", 1)[0].upper()
        params = [{"key": k, "value": v} for k, v in cfg.items()]
        workload_knobs.append(
            {
                "name": title,
                "code": key,
                "badge": badge,
                "params": params,
            }
        )
    return {
        "experiment_blurb": "Five AI inference workloads are executed across four execution configurations, isolating the cost of data serialization, engine-to-Python boundary crossings, and framework overhead from net model compute time.",
        "workloads": workloads,
        "configs": configs,
        "run_specs": run_specs,
        "model_specs": model_specs,
        "workload_knobs": workload_knobs,
    }


def _build_tel_cards(rows: pd.DataFrame) -> tuple[list[dict], dict]:
    tel_maxes = {
        "Peak GPU Util (%)":   max(float(rows["Peak GPU Util (%)"].max()),  1e-9),
        "Pipeline Continuity": 1.0,
    }
    cards = []
    for workload in sorted(rows["Workload"].unique()):
        cfgs = rows[rows["Workload"] == workload].to_dict("records")
        if cfgs:
            best_gpu  = max(c["Peak GPU Util (%)"]  for c in cfgs)
            cont_candidates = [
                float(c["Pipeline Continuity"])
                for c in cfgs
                if not pd.isna(c.get("Pipeline Continuity"))
            ]
            best_cont = max(cont_candidates) if cont_candidates else None
            min_disk  = min(c["Disk Write (MB)"] for c in cfgs)
            for c in cfgs:
                c["BestGPU"]  = abs(c["Peak GPU Util (%)"]  - best_gpu)  < 0.05
                c["ContinuityDisplay"] = (
                    "N/A"
                    if pd.isna(c.get("Pipeline Continuity"))
                    else f"{float(c['Pipeline Continuity']):.3f}"
                )
                c["ContinuityBarPct"] = (
                    0
                    if pd.isna(c.get("Pipeline Continuity"))
                    else max(2, round(float(c["Pipeline Continuity"]) * 100))
                )
                c["BestCont"] = (
                    best_cont is not None
                    and not pd.isna(c.get("Pipeline Continuity"))
                    and abs(float(c["Pipeline Continuity"]) - best_cont) < 1e-6
                )
                c["BestDisk"] = abs(c["Disk Write (MB)"] - min_disk) < 1e-6
                c["DiskDisplayLabel"] = (
                    "Runtime writes"
                    if c.get("Disk Metric Kind") == "runtime_writes"
                    else "Output footprint"
                )
        cards.append({"workload": workload, "configs": cfgs})
    return cards, tel_maxes


def _build_overhead_breakdown_payload(run_df: pd.DataFrame) -> dict[str, Any]:
    configs = ["A", "B", "C", "D"]
    if run_df.empty:
        return {
            "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
            "workloads": [],
            "maxSharePct": 100.0,
        }

    rows = run_df[
        [
            "Workload",
            "Config",
            "WallTime",
            "TransferTime_sec",
            "TraceComputeTime_sec",
            "WorkerWrapperTime_sec",
            "PreTrace_sec",
            "UntracedInWindow_sec",
            "PostTrace_sec",
            "UntracedEngineRuntime_sec",
            "BoundaryTax_sec",
        ]
    ].copy()
    rows["Serialization_sec"] = rows["TransferTime_sec"].fillna(0.0).clip(lower=0.0)
    rows["Compute_sec"] = rows["TraceComputeTime_sec"].fillna(0.0).clip(lower=0.0)
    rows["Startup_sec"] = rows["PreTrace_sec"].fillna(0.0).clip(lower=0.0)
    rows["ActiveGap_sec"] = (
        rows["WorkerWrapperTime_sec"].fillna(0.0)
        + rows["UntracedInWindow_sec"].fillna(0.0)
        + rows["UntracedEngineRuntime_sec"].fillna(0.0)
    ).clip(lower=0.0)
    rows["Tail_sec"] = rows["PostTrace_sec"].fillna(0.0).clip(lower=0.0)
    agg = (
        rows.groupby(["Workload", "Config"], dropna=False, as_index=False)
        .mean(numeric_only=True)
        .sort_values(["Workload", "Config"])
    )

    workloads: list[dict[str, Any]] = []
    for workload in sorted(agg["Workload"].unique()):
        workload_df = agg[agg["Workload"] == workload].copy()
        workload_df["Config"] = pd.Categorical(
            workload_df["Config"], categories=configs, ordered=True
        )
        workload_df = workload_df.sort_values("Config")
        bars = []
        for _, row in workload_df.iterrows():
            config = str(row["Config"])
            wall_sec = round(float(row["WallTime"]), 6)
            serial_sec = round(min(wall_sec, float(row["Serialization_sec"])), 6)
            compute_cap = max(0.0, wall_sec - serial_sec)
            compute_sec = round(min(compute_cap, float(row["Compute_sec"])), 6)
            startup_cap = max(0.0, wall_sec - serial_sec - compute_sec)
            startup_sec = round(min(startup_cap, float(row["Startup_sec"])), 6)
            active_gap_cap = max(0.0, wall_sec - serial_sec - compute_sec - startup_sec)
            active_gap_sec = round(min(active_gap_cap, float(row["ActiveGap_sec"])), 6)
            tail_sec = round(
                max(0.0, wall_sec - serial_sec - compute_sec - startup_sec - active_gap_sec),
                6,
            )
            denom = max(
                wall_sec,
                serial_sec + compute_sec + startup_sec + active_gap_sec + tail_sec,
                1e-12,
            )
            bars.append(
                {
                    "workload": workload,
                    "config": config,
                    "label": get_label(config),
                    "wall_sec": wall_sec,
                    "serial_sec": serial_sec,
                    "compute_sec": compute_sec,
                    "startup_sec": startup_sec,
                    "active_gap_sec": active_gap_sec,
                    "tail_sec": tail_sec,
                    "serial_pct": round(serial_sec / denom * 100.0, 3),
                    "compute_pct": round(compute_sec / denom * 100.0, 3),
                    "startup_pct": round(startup_sec / denom * 100.0, 3),
                    "active_gap_pct": round(active_gap_sec / denom * 100.0, 3),
                    "tail_pct": round(tail_sec / denom * 100.0, 3),
                    "boundary_tax_sec": round(float(row["BoundaryTax_sec"]), 6),
                    "boundary_tax_pct": round(float(row["BoundaryTax_sec"]) / denom * 100.0, 3),
                }
            )
        workloads.append({"workload": workload, "bars": bars})

    return {
        "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
        "workloads": workloads,
        "maxSharePct": 100.0,
    }


def _build_overhead_breakdown_section(chart_data: dict[str, Any]) -> str:
    payload_json = json.dumps(chart_data, ensure_ascii=True).replace("</", "<\\/")
    legend_html = []
    for item in chart_data.get("configs", []):
        code = escape(str(item.get("code", "")))
        label = escape(str(item.get("label", "")))
        legend_html.append(
            f'<span class="overhead-config-pill"><strong>{code}</strong><span>{label}</span></span>'
        )
    legend_markup = "".join(legend_html)
    return """
<div class="card overhead-card">
  <h2>Runtime Budget Breakdown</h2>
  <p class="section-note">This consolidated D3 view replaces the older overhead and serialization plots with one trace-accounted runtime budget. Each workload panel compares execution configs by share of wall time, split into direct conversion / transfer, traced compute, startup / dispatch, untraced active-window runtime, and tail / materialization. Right-edge labels show mean wall time.</p>
  <p class="section-note">For Spark paths, the full boundary tax is broader than the direct conversion bar alone. It includes dispatch, worker/bootstrap cost, untraced runtime mediation, and tail materialization. Small traced compute in mock-model CPU runs means little compute was performed, not that one engine computes faster.</p>
  <div class="overhead-legend">
    <div class="overhead-legend-group">
      <span class="overhead-legend-title">Execution configs</span>
      <div class="overhead-config-grid">__LEGEND__</div>
    </div>
    <div class="overhead-legend-group">
      <span class="overhead-legend-title">Attribution buckets</span>
      <div class="overhead-component-grid">
        <span class="overhead-component-pill"><span class="overhead-swatch overhead-serial"></span>Direct conversion / transfer</span>
        <span class="overhead-component-pill"><span class="overhead-swatch overhead-compute"></span>Traced compute</span>
        <span class="overhead-component-pill"><span class="overhead-swatch overhead-startup"></span>Startup / dispatch</span>
        <span class="overhead-component-pill"><span class="overhead-swatch overhead-gap"></span>Untraced active runtime</span>
        <span class="overhead-component-pill"><span class="overhead-swatch overhead-tail"></span>Tail / materialization</span>
        <span class="overhead-component-pill"><span class="overhead-swatch overhead-total"></span>Total wall time label</span>
      </div>
    </div>
  </div>
  <div class="overhead-chart-shell">
    <script type="application/json" id="overhead-breakdown-data">__PAYLOAD__</script>
    <div id="overhead-breakdown-chart" class="overhead-chart-grid"></div>
  </div>
</div>
<style>
  .overhead-card { border: 1px solid #dbe7ff; background: linear-gradient(180deg, rgba(255,255,255,0.98), #f8fafc); }
  .overhead-legend { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 10px 0 16px; }
  .overhead-legend-group { border: 1px solid #e2e8f0; border-radius: 14px; background: #fff; padding: 12px 14px; }
  .overhead-legend-title { display: block; font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; margin-bottom: 10px; }
  .overhead-config-grid, .overhead-component-grid { display: flex; flex-wrap: wrap; gap: 8px; }
  .overhead-config-pill, .overhead-component-pill { display: inline-flex; align-items: center; gap: 8px; padding: 7px 10px; border-radius: 999px; border: 1px solid #e2e8f0; background: #f8fafc; color: #334155; font-size: 12px; line-height: 1.1; }
  .overhead-config-pill strong { color: #0f172a; }
  .overhead-config-pill span { white-space: nowrap; }
  .overhead-swatch { width: 11px; height: 11px; border-radius: 999px; display: inline-block; flex: 0 0 auto; }
  .overhead-serial { background: #d9a35f; }
  .overhead-compute { background: #6fb686; }
  .overhead-startup { background: #7ea6d8; }
  .overhead-gap { background: #d38c6a; }
  .overhead-tail { background: #a993cf; }
  .overhead-total { background: #94a3b8; }
  .overhead-chart-shell { margin-top: 6px; }
  .overhead-chart-grid { display: grid; gap: 16px; grid-template-columns: 1fr; }
  .overhead-panel { border: 1px solid #e2e8f0; border-radius: 16px; background: linear-gradient(180deg, #fff, #f8fafc); padding: 14px 14px 12px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04); }
  .overhead-panel h3 { margin: 0; font-size: 16px; color: #0f172a; }
  .overhead-panel .subtitle { margin: 3px 0 12px; font-size: 12px; color: #64748b; }
  .overhead-svg { width: 100%; height: auto; display: block; }
  .overhead-axis text { fill: #475569; font-size: 11px; }
  .overhead-axis path, .overhead-axis line { stroke: #cbd5e1; }
  .overhead-grid line { stroke: #e2e8f0; stroke-dasharray: 3 4; }
  .overhead-share-label { fill: #0f172a; font-size: 11px; font-weight: 700; pointer-events: none; }
  .overhead-wall-label { fill: #64748b; font-size: 11px; font-weight: 700; pointer-events: none; }
  .overhead-note { grid-column: 1 / -1; margin-top: -2px; font-size: 12px; color: #64748b; }
  .overhead-bar .segment { pointer-events: none; }
  .overhead-bar:hover .overhead-hit { fill: rgba(15, 23, 42, 0.03); }
  .overhead-tooltip { position: fixed; z-index: 9999; pointer-events: none; opacity: 0; transition: opacity 120ms ease; background: rgba(15, 23, 42, 0.96); color: #f8fafc; border-radius: 12px; padding: 10px 12px; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22); font-size: 12px; line-height: 1.45; max-width: 260px; }
  .overhead-tooltip strong { display: block; font-size: 13px; margin-bottom: 4px; color: #fff; }
  .overhead-tooltip .metric { display: flex; justify-content: space-between; gap: 12px; white-space: nowrap; }
  .overhead-tooltip .metric span:first-child { color: #cbd5e1; }
  .overhead-empty { padding: 20px; border: 1px dashed #cbd5e1; border-radius: 14px; color: #475569; background: #fff; }
  @media (max-width: 760px) {
    .overhead-legend { grid-template-columns: 1fr; }
  }
</style>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  const dataNode = document.getElementById("overhead-breakdown-data");
  const chartNode = document.getElementById("overhead-breakdown-chart");
  if (!dataNode || !chartNode || typeof d3 === "undefined") {
    return;
  }

  const payload = JSON.parse(dataNode.textContent);
  const tooltip = d3.select("body").selectAll(".overhead-tooltip")
    .data([null])
    .join("div")
    .attr("class", "overhead-tooltip");

  const palette = {
    serial: "#d9a35f",
    compute: "#6fb686",
    startup: "#7ea6d8",
    activeGap: "#d38c6a",
    tail: "#a993cf",
    grid: "#e2e8f0",
    axis: "#94a3b8",
  };

  function formatSeconds(value, compact = false) {
    if (!Number.isFinite(value)) {
      return "-";
    }
    if (compact && Math.abs(value) >= 100) {
      return `${Math.round(value)}s`;
    }
    if (compact && Math.abs(value) >= 10) {
      return `${value.toFixed(1)}s`;
    }
    return value >= 1 ? `${value.toFixed(2)}s` : `${value.toFixed(3)}s`;
  }

  function formatPercent(value) {
    if (!Number.isFinite(value)) {
      return "-";
    }
    return value >= 10 ? `${Math.round(value)}%` : `${value.toFixed(1)}%`;
  }

  function hideTooltip() {
    tooltip.style("opacity", 0);
  }

  function showTooltip(event, datum) {
    tooltip
      .style("opacity", 1)
      .html(`
        <strong>${datum.workload} · Config ${datum.config}</strong>
        <div class="metric"><span>Direct conversion / transfer</span><span>${formatSeconds(datum.serial_sec)} · ${formatPercent(datum.serial_pct)}</span></div>
        <div class="metric"><span>Traced compute</span><span>${formatSeconds(datum.compute_sec)} · ${formatPercent(datum.compute_pct)}</span></div>
        <div class="metric"><span>Startup / dispatch</span><span>${formatSeconds(datum.startup_sec)} · ${formatPercent(datum.startup_pct)}</span></div>
        <div class="metric"><span>Untraced active runtime</span><span>${formatSeconds(datum.active_gap_sec)} · ${formatPercent(datum.active_gap_pct)}</span></div>
        <div class="metric"><span>Tail / materialization</span><span>${formatSeconds(datum.tail_sec)} · ${formatPercent(datum.tail_pct)}</span></div>
        <div class="metric"><span>Total wall time</span><span>${formatSeconds(datum.wall_sec)}</span></div>
        <div class="metric"><span>Derived boundary tax</span><span>${formatSeconds(datum.boundary_tax_sec)} · ${formatPercent(datum.boundary_tax_pct)}</span></div>
      `);

    const pad = 18;
    const rect = tooltip.node().getBoundingClientRect();
    let left = event.clientX + 16;
    let top = event.clientY + 16;
    if (left + rect.width + pad > window.innerWidth) {
      left = event.clientX - rect.width - 16;
    }
    if (top + rect.height + pad > window.innerHeight) {
      top = event.clientY - rect.height - 16;
    }
    tooltip.style("left", `${Math.max(8, left)}px`).style("top", `${Math.max(8, top)}px`);
  }

  function render() {
    chartNode.innerHTML = "";
    if (!payload.workloads || payload.workloads.length === 0) {
      chartNode.innerHTML = '<div class="overhead-empty">No trace-accounted timing data was found for this run.</div>';
      return;
    }

    const containerWidth = chartNode.getBoundingClientRect().width || chartNode.clientWidth || 960;
    const panelWidth = containerWidth;
    const panelHeight = 264;
    const margin = { top: 16, right: 80, bottom: 36, left: 64 };
    const innerWidth = Math.max(180, panelWidth - margin.left - margin.right);
    const innerHeight = panelHeight - margin.top - margin.bottom;
    const xScale = d3.scaleLinear().domain([0, payload.maxSharePct || 100]).range([0, innerWidth]);
    const yScale = d3.scaleBand().domain(payload.configs.map((d) => d.code)).range([0, innerHeight]).padding(0.24);
    const barHeight = yScale.bandwidth();
    const svgWidth = innerWidth + margin.left + margin.right;
    const svgHeight = panelHeight;

    payload.workloads.forEach((workload) => {
      const panel = chartNode.appendChild(document.createElement("div"));
      panel.className = "overhead-panel";
      panel.innerHTML = `<h3>${workload.workload}</h3><div class="subtitle">Share of wall time by execution config</div>`;

      const svg = d3.select(panel)
        .append("svg")
        .attr("class", "overhead-svg")
        .attr("viewBox", `0 0 ${svgWidth} ${svgHeight}`)
        .attr("role", "img")
        .attr("aria-label", `Runtime budget breakdown for workload ${workload.workload}`);

      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

      g.append("g")
        .attr("class", "overhead-grid")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).tickValues([0, 25, 50, 75, 100]).tickSize(-innerHeight).tickFormat(""))
        .selectAll("line")
        .attr("stroke", palette.grid);

      g.append("g")
        .attr("class", "overhead-axis")
        .call(d3.axisLeft(yScale).tickSizeOuter(0));

      g.append("g")
        .attr("class", "overhead-axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).tickValues([0, 25, 50, 75, 100]).tickFormat((d) => `${d}%`));

      const barGroups = g.selectAll(".overhead-bar")
        .data(workload.bars)
        .join("g")
        .attr("class", "overhead-bar")
        .attr("transform", (d) => `translate(0,${yScale(d.config) || 0})`);

      barGroups.append("rect")
        .attr("class", "overhead-hit")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", innerWidth + margin.right)
        .attr("height", barHeight)
        .attr("fill", "transparent")
        .on("pointerenter", function(event, d) {
          d3.select(this.parentNode).selectAll("rect.segment").attr("opacity", 0.92);
          showTooltip(event, d);
        })
        .on("pointermove", function(event, d) {
          showTooltip(event, d);
        })
        .on("pointerleave", function() {
          d3.select(this.parentNode).selectAll("rect.segment").attr("opacity", 1);
          hideTooltip();
        });

      barGroups.append("rect")
        .attr("class", "segment")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", (d) => Math.max(0, xScale(d.serial_pct)))
        .attr("height", barHeight)
        .attr("fill", palette.serial);

      barGroups.append("rect")
        .attr("class", "segment")
        .attr("x", (d) => xScale(d.serial_pct))
        .attr("y", 0)
        .attr("width", (d) => Math.max(0, xScale(d.compute_pct)))
        .attr("height", barHeight)
        .attr("fill", palette.compute);

      barGroups.append("rect")
        .attr("class", "segment")
        .attr("x", (d) => xScale(d.serial_pct + d.compute_pct))
        .attr("y", 0)
        .attr("width", (d) => Math.max(0, xScale(d.startup_pct)))
        .attr("height", barHeight)
        .attr("fill", palette.startup);

      barGroups.append("rect")
        .attr("class", "segment")
        .attr("x", (d) => xScale(d.serial_pct + d.compute_pct + d.startup_pct))
        .attr("y", 0)
        .attr("width", (d) => Math.max(0, xScale(d.active_gap_pct)))
        .attr("height", barHeight)
        .attr("fill", palette.activeGap);

      barGroups.append("rect")
        .attr("class", "segment")
        .attr("x", (d) => xScale(d.serial_pct + d.compute_pct + d.startup_pct + d.active_gap_pct))
        .attr("y", 0)
        .attr("width", (d) => Math.max(0, xScale(d.tail_pct)))
        .attr("height", barHeight)
        .attr("fill", palette.tail);

      barGroups.filter((d) => d.serial_pct >= 9).append("text")
        .attr("class", "overhead-share-label")
        .attr("x", (d) => xScale(d.serial_pct / 2))
        .attr("y", barHeight / 2 + 4)
        .attr("text-anchor", "middle")
        .attr("fill", "#fff")
        .text((d) => formatPercent(d.serial_pct));

      barGroups.filter((d) => d.compute_pct >= 9).append("text")
        .attr("class", "overhead-share-label")
        .attr("x", (d) => xScale(d.serial_pct + d.compute_pct / 2))
        .attr("y", barHeight / 2 + 4)
        .attr("text-anchor", "middle")
        .attr("fill", "#fff")
        .text((d) => formatPercent(d.compute_pct));

      barGroups.filter((d) => d.startup_pct >= 8).append("text")
        .attr("class", "overhead-share-label")
        .attr("x", (d) => xScale(d.serial_pct + d.compute_pct + d.startup_pct / 2))
        .attr("y", barHeight / 2 + 4)
        .attr("text-anchor", "middle")
        .text((d) => formatPercent(d.startup_pct));

      barGroups.filter((d) => d.active_gap_pct >= 8).append("text")
        .attr("class", "overhead-share-label")
        .attr("x", (d) => xScale(d.serial_pct + d.compute_pct + d.startup_pct + d.active_gap_pct / 2))
        .attr("y", barHeight / 2 + 4)
        .attr("text-anchor", "middle")
        .text((d) => formatPercent(d.active_gap_pct));

      barGroups.filter((d) => d.tail_pct >= 8).append("text")
        .attr("class", "overhead-share-label")
        .attr("x", (d) => xScale(d.serial_pct + d.compute_pct + d.startup_pct + d.active_gap_pct + d.tail_pct / 2))
        .attr("y", barHeight / 2 + 4)
        .attr("text-anchor", "middle")
        .text((d) => formatPercent(d.tail_pct));

      barGroups.append("text")
        .attr("class", "overhead-wall-label")
        .attr("x", innerWidth + 10)
        .attr("y", barHeight / 2 + 4)
        .attr("text-anchor", "start")
        .text((d) => formatSeconds(d.wall_sec, true));

    });

    const note = document.createElement("div");
    note.className = "overhead-note";
    note.textContent = "Direct conversion / transfer is only the explicitly traced in-worker conversion cost. Derived boundary tax rolls up direct transfer, dispatch/startup, worker wrapper overhead, untraced active runtime, and tail/materialization. W0 values average depth 1/2/3 within each config before computing the time-budget split.";
    chartNode.appendChild(note);
  }

  render();

  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      window.requestAnimationFrame(render);
    });
    observer.observe(chartNode);
  } else {
    window.addEventListener("resize", render);
  }
})();
</script>
""".replace("__PAYLOAD__", payload_json).replace("__LEGEND__", legend_markup)


def _build_memory_payload(run_df: pd.DataFrame) -> dict[str, Any]:
    configs = ["A", "B", "C", "D"]
    if run_df.empty:
        return {
            "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
            "workloads": [],
            "maxPeakRssMb": 0.0,
        }

    rows = run_df[["Workload", "Config", "PeakRSS_MB"]].copy()
    agg = (
        rows.groupby(["Workload", "Config"], dropna=False, as_index=False)
        .mean(numeric_only=True)
        .sort_values(["Workload", "Config"])
    )

    workloads: list[dict[str, Any]] = []
    for workload in sorted(agg["Workload"].unique()):
        workload_df = agg[agg["Workload"] == workload].copy()
        workload_df["Config"] = pd.Categorical(
            workload_df["Config"], categories=configs, ordered=True
        )
        workload_df = workload_df.sort_values("Config")
        bars = []
        for _, row in workload_df.iterrows():
            config = str(row["Config"])
            bars.append(
                {
                    "config": config,
                    "label": get_label(config),
                    "peak_rss_mb": round(float(row["PeakRSS_MB"]), 3),
                }
            )
        workloads.append({"workload": workload, "bars": bars})

    return {
        "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
        "workloads": workloads,
        "maxPeakRssMb": round(float(agg["PeakRSS_MB"].max()), 3),
    }


def _build_speedup_payload(run_df: pd.DataFrame) -> dict[str, Any]:
    configs = ["A", "B", "C", "D"]
    baseline_config = "B"
    if run_df.empty:
        return {
            "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
            "workloads": [],
            "maxSpeedup": 0.0,
            "minSpeedup": 0.0,
        }

    rows = run_df[["Workload", "Config", "WallTime"]].copy()
    agg = (
        rows.groupby(["Workload", "Config"], dropna=False, as_index=False)
        .mean(numeric_only=True)
        .sort_values(["Workload", "Config"])
    )

    workloads: list[dict[str, Any]] = []
    speedup_values: list[float] = []
    for workload in sorted(agg["Workload"].unique()):
        workload_df = agg[agg["Workload"] == workload].copy()
        baseline_rows = workload_df[workload_df["Config"] == baseline_config]
        if baseline_rows.empty:
            continue
        baseline = float(baseline_rows["WallTime"].iloc[0])
        bars = []
        for cfg in configs:
            cfg_rows = workload_df[workload_df["Config"] == cfg]
            if cfg_rows.empty:
                continue
            wall = float(cfg_rows["WallTime"].iloc[0])
            if wall <= 0 or baseline <= 0:
                continue
            speedup = baseline / wall
            speedup_values.append(speedup)
            bars.append(
                {
                    "config": cfg,
                    "label": get_label(cfg),
                    "speedup": round(speedup, 3),
                    "wall_sec": round(wall, 6),
                }
            )
        if bars:
            workloads.append({"workload": workload, "baseline": round(baseline, 6), "bars": bars})

    if not speedup_values:
        min_speedup = 0.0
        max_speedup = 0.0
    else:
        min_speedup = round(min(speedup_values), 3)
        max_speedup = round(max(speedup_values), 3)

    return {
        "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
        "workloads": workloads,
        "minSpeedup": min_speedup,
        "maxSpeedup": max_speedup,
        "baselineConfig": baseline_config,
        "baselineLabel": get_label(baseline_config),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _build_gpu_utilization_payload(results_dir: Path) -> dict[str, Any]:
    configs = ["A", "B", "C", "D"]
    workloads_by_key: dict[str, list[dict[str, Any]]] = {}
    has_gpu_data = False
    has_vllm_data = False
    max_time_sec = 0.0

    for manifest_path in _manifest_paths(results_dir):
        manifest = _read_json(manifest_path)
        stats_path = _resolve_artifact_path(
            manifest_path, manifest, "stats_json", "_stats.json"
        )
        if stats_path is None:
            continue
        stats = _read_json(stats_path)
        workload = str(manifest.get("workload", stats.get("workload", ""))).upper()
        config = str(manifest.get("execution", stats.get("execution", ""))).upper()
        if not workload or not config:
            continue

        points: list[dict[str, Any]] = []
        for sample in stats.get("samples") or []:
            point = {
                "t_sec": round(float(sample.get("t_sec", 0.0) or 0.0), 3),
            }
            for key in [
                "gpu_util_pct",
                "gpu_power_w",
                "gpu_mem_used_mb",
                "gpu_mem_total_mb",
                "vllm_gpu_cache_usage_pct",
                "vllm_requests_running",
                "vllm_requests_waiting",
                "vllm_prompt_tokens_total",
                "vllm_generation_tokens_total",
            ]:
                value = _float_or_none(sample.get(key))
                if value is not None:
                    point[key] = round(value, 6)
            if "gpu_util_pct" in point or "gpu_power_w" in point or "gpu_mem_used_mb" in point:
                has_gpu_data = True
            if any(str(key).startswith("vllm_") for key in point):
                has_vllm_data = True
            max_time_sec = max(max_time_sec, float(point["t_sec"]))
            points.append(point)

        def counter_delta(key: str) -> float:
            values = [
                float(point[key])
                for point in points
                if point.get(key) is not None
            ]
            if len(values) < 2:
                return 0.0
            return max(0.0, values[-1] - values[0])

        wall_time = float(stats.get("wall_clock_sec", manifest.get("wall_clock_sec", 0.0)) or 0.0)
        max_time_sec = max(max_time_sec, wall_time)
        run = {
            "run_id": str(manifest.get("run_id", stats.get("config", ""))),
            "workload": workload,
            "config": config,
            "label": get_label(config),
            "depth": manifest.get("depth"),
            "sample_idx": _parse_sample_idx(str(manifest.get("run_id", stats.get("config", "")))),
            "wall_sec": round(wall_time, 6),
            "avg_gpu_util_pct": round(float(stats.get("avg_gpu_util_pct", 0.0) or 0.0), 3),
            "peak_gpu_util_pct": round(float(stats.get("peak_gpu_util_pct", 0.0) or 0.0), 3),
            "avg_gpu_power_w": round(float(stats.get("avg_gpu_power_w", 0.0) or 0.0), 3),
            "peak_gpu_mem_used_mb": round(float(stats.get("peak_gpu_mem_used_mb", 0.0) or 0.0), 3),
            "gpu_telemetry_available": bool(stats.get("gpu_telemetry_available", False)),
            "pipeline_continuity_available": bool(stats.get("pipeline_continuity_available", stats.get("pipeline_continuity") is not None)),
            "pipeline_continuity": (
                round(float(stats["pipeline_continuity"]), 3)
                if stats.get("pipeline_continuity") is not None
                else None
            ),
            "vllm_telemetry_available": bool(stats.get("vllm_telemetry_available", False)),
            "avg_vllm_gpu_cache_usage_pct": round(float(stats.get("avg_vllm_gpu_cache_usage_pct", 0.0) or 0.0), 6),
            "peak_vllm_gpu_cache_usage_pct": round(float(stats.get("peak_vllm_gpu_cache_usage_pct", 0.0) or 0.0), 6),
            "peak_vllm_requests_running": round(float(stats.get("peak_vllm_requests_running", 0.0) or 0.0), 3),
            "peak_vllm_requests_waiting": round(float(stats.get("peak_vllm_requests_waiting", 0.0) or 0.0), 3),
            "vllm_prompt_tokens_delta": round(counter_delta("vllm_prompt_tokens_total"), 3),
            "vllm_generation_tokens_delta": round(counter_delta("vllm_generation_tokens_total"), 3),
            "points": points,
        }
        workloads_by_key.setdefault(workload, []).append(run)

    workloads = []
    for workload in sorted(workloads_by_key):
        runs = sorted(
            workloads_by_key[workload],
            key=lambda item: (
                str(item["config"]),
                -1 if item["sample_idx"] is None else int(item["sample_idx"]),
                -1 if item["depth"] is None else int(item["depth"]),
            ),
        )
        sample_indexes = sorted(
            {
                int(run["sample_idx"])
                for run in runs
                if run.get("sample_idx") is not None
            }
        )
        workloads.append(
            {
                "workload": workload,
                "label": WORKLOAD_LABELS.get(workload, workload),
                "runs": runs,
                "sampleIndexes": sample_indexes,
            }
        )

    default_workload = ""
    workload_keys = [item["workload"] for item in workloads]
    if "W1" in workload_keys:
        default_workload = "W1"
    elif workload_keys:
        default_workload = workload_keys[0]

    return {
        "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
        "workloads": workloads,
        "defaultWorkload": default_workload,
        "hasGpuData": has_gpu_data,
        "hasVllmData": has_vllm_data,
        "maxTimeSec": round(max_time_sec, 3),
    }


def _build_gpu_utilization_section(chart_data: dict[str, Any]) -> str:
    payload_json = json.dumps(chart_data, ensure_ascii=True).replace("</", "<\\/")
    return """
<div class="card gpu-card">
  <h2>GPU Utilization &amp; vLLM Activity</h2>
  <p class="section-note">This D3 view tracks active accelerator work rather than reserved memory. vLLM intentionally occupies memory according to <code>gpu_memory_utilization</code>, so flat GPU memory mostly reflects reserved KV/model capacity; SM utilization, power, queue pressure, token movement, and KV-cache usage are the better activity signals.</p>
  <script type="application/json" id="gpu-utilization-data">__PAYLOAD__</script>
  <div class="gpu-controls">
    <label>Workload <select id="gpu-workload-select"></select></label>
    <label>Sample <select id="gpu-sample-select"></select></label>
    <div id="gpu-config-toggles" class="gpu-toggle-row" aria-label="Execution config toggles"></div>
  </div>
  <div id="gpu-utilization-chart" class="gpu-chart-shell"></div>
  <div id="gpu-summary-cards" class="gpu-summary-grid"></div>
  <p class="gpu-footnote">GPU memory is shown as reserved footprint when present. Treat it as capacity pressure, not as proof of active GPU execution.</p>
</div>
<style>
  .gpu-card { border-color: #dbe7ff; background: linear-gradient(180deg, rgba(255,255,255,0.98), #f8fafc); }
  .gpu-controls { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin: 12px 0 14px; }
  .gpu-controls label { display: inline-flex; align-items: center; gap: 8px; color: #334155; font-size: 13px; font-weight: 700; }
  .gpu-controls select { border: 1px solid #cbd5e1; border-radius: 10px; background: #fff; color: #0f172a; padding: 7px 10px; font: inherit; }
  .gpu-toggle-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .gpu-toggle { display: inline-flex; align-items: center; gap: 6px; border: 1px solid #dbe7ff; border-radius: 999px; background: #fff; padding: 7px 10px; color: #334155; font-size: 12px; font-weight: 800; }
  .gpu-toggle input { accent-color: #3762e0; }
  .gpu-chart-shell { min-height: 340px; border: 1px solid #e2e8f0; border-radius: 16px; background: #fff; padding: 12px; }
  .gpu-svg { width: 100%; height: auto; display: block; }
  .gpu-axis text { fill: #475569; font-size: 11px; }
  .gpu-axis path, .gpu-axis line { stroke: #cbd5e1; }
  .gpu-grid line { stroke: #e2e8f0; stroke-dasharray: 3 4; }
  .gpu-run-line { fill: none; stroke-width: 2.4; }
  .gpu-point { stroke: #fff; stroke-width: 1.2; }
  .gpu-empty { padding: 22px; border: 1px dashed #cbd5e1; border-radius: 14px; color: #475569; background: #fff; }
  .gpu-legend { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; color: #475569; font-size: 12px; }
  .gpu-legend-item { display: inline-flex; align-items: center; gap: 6px; }
  .gpu-swatch { width: 12px; height: 12px; border-radius: 999px; display: inline-block; }
  .gpu-summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-top: 12px; }
  .gpu-summary-card { border: 1px solid #e2e8f0; border-radius: 14px; background: #fff; padding: 12px; }
  .gpu-summary-card .k { display: block; font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 800; }
  .gpu-summary-card .v { display: block; color: #0f172a; font-size: 18px; font-weight: 850; margin-top: 4px; }
  .gpu-summary-card .s { display: block; color: #64748b; font-size: 12px; margin-top: 3px; line-height: 1.35; }
  .gpu-tooltip { position: fixed; z-index: 9999; pointer-events: none; opacity: 0; transition: opacity 120ms ease; background: rgba(15, 23, 42, 0.96); color: #f8fafc; border-radius: 12px; padding: 10px 12px; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22); font-size: 12px; line-height: 1.45; max-width: 280px; }
  .gpu-tooltip strong { display: block; font-size: 13px; margin-bottom: 4px; color: #fff; }
  .gpu-tooltip .metric { display: flex; justify-content: space-between; gap: 14px; white-space: nowrap; }
  .gpu-tooltip .metric span:first-child { color: #cbd5e1; }
  .gpu-footnote { color: #64748b; font-size: 12px; line-height: 1.45; margin: 10px 0 0; }
</style>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  const dataNode = document.getElementById("gpu-utilization-data");
  const chartNode = document.getElementById("gpu-utilization-chart");
  const cardsNode = document.getElementById("gpu-summary-cards");
  const workloadSelect = document.getElementById("gpu-workload-select");
  const sampleSelect = document.getElementById("gpu-sample-select");
  const togglesNode = document.getElementById("gpu-config-toggles");
  if (!dataNode || !chartNode || !cardsNode || typeof d3 === "undefined") {
    return;
  }
  const payload = JSON.parse(dataNode.textContent);
  const colors = { A: "#8a8f98", B: "#e25a1c", C: "#3762e0", D: "#0f9f7a" };
  const activeConfigs = new Set((payload.configs || []).map((d) => d.code));
  const tooltip = d3.select("body").selectAll(".gpu-tooltip").data([null]).join("div").attr("class", "gpu-tooltip");

  function fmtPct(value, digits = 0) {
    return Number.isFinite(value) ? `${value.toFixed(digits)}%` : "-";
  }

  function fmtPct01(value) {
    if (!Number.isFinite(value)) return "N/A";
    const pct = value <= 1 ? value * 100 : value;
    return `${pct.toFixed(pct >= 10 ? 0 : 1)}%`;
  }

  function fmtNumber(value, digits = 1) {
    return Number.isFinite(value) ? value.toFixed(digits) : "-";
  }

  function runLabel(run) {
    const depth = run.depth == null ? "" : ` d${run.depth}`;
    const sample = run.sample_idx == null ? "" : ` s${run.sample_idx}`;
    return `${run.config}${depth}${sample}`;
  }

  function setupControls() {
    workloadSelect.innerHTML = "";
    (payload.workloads || []).forEach((item) => {
      const option = document.createElement("option");
      option.value = item.workload;
      option.textContent = `${item.workload} · ${item.label}`;
      workloadSelect.appendChild(option);
    });
    if (payload.defaultWorkload) {
      workloadSelect.value = payload.defaultWorkload;
    }

    togglesNode.innerHTML = "";
    (payload.configs || []).forEach((cfg) => {
      const label = document.createElement("label");
      label.className = "gpu-toggle";
      label.innerHTML = `<input type="checkbox" value="${cfg.code}" checked><span style="color:${colors[cfg.code] || "#334155"}">${cfg.code}</span><span>${cfg.label}</span>`;
      label.querySelector("input").addEventListener("change", (event) => {
        if (event.target.checked) activeConfigs.add(cfg.code);
        else activeConfigs.delete(cfg.code);
        render();
      });
      togglesNode.appendChild(label);
    });
    workloadSelect.addEventListener("change", () => {
      syncSampleOptions();
      render();
    });
    sampleSelect.addEventListener("change", render);
    syncSampleOptions();
  }

  function selectedWorkload() {
    return (payload.workloads || []).find((item) => item.workload === workloadSelect.value) || (payload.workloads || [])[0];
  }

  function syncSampleOptions() {
    const workload = selectedWorkload();
    sampleSelect.innerHTML = "";
    const all = document.createElement("option");
    all.value = "all";
    all.textContent = "All samples";
    sampleSelect.appendChild(all);
    ((workload && workload.sampleIndexes) || []).forEach((idx) => {
      const option = document.createElement("option");
      option.value = String(idx);
      option.textContent = `Sample ${idx}`;
      sampleSelect.appendChild(option);
    });
  }

  function selectedRuns() {
    const workload = selectedWorkload();
    if (!workload) return [];
    const sampleValue = sampleSelect.value;
    return workload.runs.filter((run) => {
      if (!activeConfigs.has(run.config)) return false;
      if (sampleValue !== "all" && String(run.sample_idx) !== sampleValue) return false;
      return true;
    });
  }

  function showTooltip(event, run, point) {
    tooltip.style("opacity", 1).html(`
      <strong>${run.workload} · ${runLabel(run)}</strong>
      <div class="metric"><span>t</span><span>${fmtNumber(point.t_sec, 2)}s</span></div>
      <div class="metric"><span>SM util</span><span>${fmtPct(point.gpu_util_pct, 0)}</span></div>
      <div class="metric"><span>GPU power</span><span>${fmtNumber(point.gpu_power_w, 1)}W</span></div>
      <div class="metric"><span>vLLM KV cache</span><span>${fmtPct01(point.vllm_gpu_cache_usage_pct)}</span></div>
      <div class="metric"><span>Running / waiting</span><span>${fmtNumber(point.vllm_requests_running, 0)} / ${fmtNumber(point.vllm_requests_waiting, 0)}</span></div>
      <div class="metric"><span>Prompt / decode tokens</span><span>${fmtNumber(point.vllm_prompt_tokens_total, 0)} / ${fmtNumber(point.vllm_generation_tokens_total, 0)}</span></div>
    `);
    const rect = tooltip.node().getBoundingClientRect();
    let left = event.clientX + 16;
    let top = event.clientY + 16;
    if (left + rect.width + 18 > window.innerWidth) left = event.clientX - rect.width - 16;
    if (top + rect.height + 18 > window.innerHeight) top = event.clientY - rect.height - 16;
    tooltip.style("left", `${Math.max(8, left)}px`).style("top", `${Math.max(8, top)}px`);
  }

  function renderCards(runs) {
    cardsNode.innerHTML = "";
    if (!runs.length) return;
    const continuityValues = runs.map((d) => d.pipeline_continuity).filter((v) => Number.isFinite(v));
    const bestContinuity = continuityValues.length ? d3.max(continuityValues) : null;
    const bestPeak = d3.max(runs, (d) => d.peak_gpu_util_pct) || 0;
    const bestKv = d3.max(runs, (d) => d.peak_vllm_gpu_cache_usage_pct) || 0;
    const maxTokenDelta = d3.max(runs, (d) => (d.vllm_prompt_tokens_delta || 0) + (d.vllm_generation_tokens_delta || 0)) || 0;
    const cards = [
      { k: "Best continuity", v: fmtPct01(bestContinuity), s: "Share of GPU samples above active-util threshold." },
      { k: "Peak SM util", v: fmtPct(bestPeak, 0), s: "Burst ceiling from nvidia-smi utilization.gpu." },
      { k: "Peak vLLM KV cache", v: bestKv ? fmtPct01(bestKv) : "-", s: payload.hasVllmData ? "Cache pressure inside vLLM reserved memory." : "vLLM /metrics unavailable." },
      { k: "Max queue pressure", v: fmtNumber(d3.max(runs, (d) => d.peak_vllm_requests_waiting) || 0, 0), s: "Peak waiting requests from vLLM scheduler metrics." },
      { k: "Max token movement", v: fmtNumber(maxTokenDelta, 0), s: "Prompt + decode token counter delta within a run." },
    ];
    cards.forEach((card) => {
      const node = document.createElement("div");
      node.className = "gpu-summary-card";
      node.innerHTML = `<span class="k">${card.k}</span><span class="v">${card.v}</span><span class="s">${card.s}</span>`;
      cardsNode.appendChild(node);
    });
  }

  function render() {
    const runs = selectedRuns();
    chartNode.innerHTML = "";
    renderCards(runs);
    if (!payload.workloads || payload.workloads.length === 0) {
      chartNode.innerHTML = '<div class="gpu-empty">No benchmark runs were found for GPU utilization.</div>';
      return;
    }
    const lineRuns = runs.map((run) => ({
      run,
      points: (run.points || []).filter((point) => Number.isFinite(point.gpu_util_pct)),
    })).filter((item) => item.points.length > 0);
    if (!lineRuns.length) {
      chartNode.innerHTML = payload.hasVllmData
        ? '<div class="gpu-empty">vLLM activity metrics were recorded, but no GPU SM utilization samples were available for this selection.</div>'
        : '<div class="gpu-empty">No GPU/vLLM activity samples recorded for this selection.</div>';
      return;
    }

    const width = chartNode.getBoundingClientRect().width || 960;
    const height = 340;
    const margin = { top: 20, right: 30, bottom: 42, left: 58 };
    const innerWidth = Math.max(220, width - margin.left - margin.right);
    const innerHeight = height - margin.top - margin.bottom;
    const maxT = d3.max(lineRuns.flatMap((item) => item.points), (d) => d.t_sec) || 1;
    const x = d3.scaleLinear().domain([0, Math.max(1, maxT)]).range([0, innerWidth]);
    const y = d3.scaleLinear().domain([0, 100]).range([innerHeight, 0]);
    const line = d3.line().x((d) => x(d.t_sec)).y((d) => y(d.gpu_util_pct)).curve(d3.curveMonotoneX);
    const svg = d3.select(chartNode).append("svg")
      .attr("class", "gpu-svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("role", "img")
      .attr("aria-label", "GPU SM utilization timeline by benchmark run");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    g.append("g")
      .attr("class", "gpu-grid")
      .call(d3.axisLeft(y).tickValues([0, 25, 50, 75, 100]).tickSize(-innerWidth).tickFormat(""))
      .selectAll("line")
      .attr("stroke", "#e2e8f0");
    g.append("g").attr("class", "gpu-axis").call(d3.axisLeft(y).tickValues([0, 25, 50, 75, 100]).tickFormat((d) => `${d}%`));
    g.append("g").attr("class", "gpu-axis").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(6).tickFormat((d) => `${d}s`));
    g.append("text").attr("x", innerWidth / 2).attr("y", innerHeight + 36).attr("text-anchor", "middle").attr("fill", "#64748b").attr("font-size", 12).text("Wall-clock seconds since run start");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -innerHeight / 2).attr("y", -42).attr("text-anchor", "middle").attr("fill", "#64748b").attr("font-size", 12).text("GPU SM utilization");

    const groups = g.selectAll(".gpu-run").data(lineRuns).join("g").attr("class", "gpu-run");
    groups.append("path")
      .attr("class", "gpu-run-line")
      .attr("stroke", (d) => colors[d.run.config] || "#334155")
      .attr("opacity", 0.86)
      .attr("d", (d) => line(d.points));
    groups.selectAll("circle")
      .data((d) => d.points.map((point) => ({ point, run: d.run })))
      .join("circle")
      .attr("class", "gpu-point")
      .attr("cx", (d) => x(d.point.t_sec))
      .attr("cy", (d) => y(d.point.gpu_util_pct))
      .attr("r", 3.2)
      .attr("fill", (d) => colors[d.run.config] || "#334155")
      .on("pointerenter", (event, d) => showTooltip(event, d.run, d.point))
      .on("pointermove", (event, d) => showTooltip(event, d.run, d.point))
      .on("pointerleave", () => tooltip.style("opacity", 0));

    const legend = document.createElement("div");
    legend.className = "gpu-legend";
    lineRuns.forEach((item) => {
      const row = document.createElement("span");
      row.className = "gpu-legend-item";
      row.innerHTML = `<span class="gpu-swatch" style="background:${colors[item.run.config] || "#334155"}"></span>${runLabel(item.run)} · continuity ${fmtPct01(item.run.pipeline_continuity)}`;
      legend.appendChild(row);
    });
    chartNode.appendChild(legend);
  }

  setupControls();
  render();
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => window.requestAnimationFrame(render));
    observer.observe(chartNode);
  } else {
    window.addEventListener("resize", render);
  }
})();
</script>
""".replace("__PAYLOAD__", payload_json)


def _build_depth_runtime_payload(run_df: pd.DataFrame) -> dict[str, Any]:
    configs = ["A", "B", "C", "D"]
    empty = {
        "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
        "depths": [],
        "series": [],
        "summaries": [],
        "hasEnoughDepths": False,
        "maxRuntimeSec": 0.0,
    }
    if run_df.empty:
        return empty

    rows = run_df[run_df["Workload"] == "W0"].copy()
    rows = rows[rows["Depth"].notna()]
    if rows.empty:
        return empty

    rows["Depth"] = rows["Depth"].astype(int)
    depths = sorted(int(d) for d in rows["Depth"].unique())
    agg_rows: list[dict[str, Any]] = []
    for (config, depth), group in rows.groupby(["Config", "Depth"], dropna=False):
        cold_group = group[group["SampleIdx"] == 1]
        warm_group = group[group["SampleIdx"] != 1]
        cold_sec = (
            float(cold_group["WallTime"].mean())
            if not cold_group.empty
            else float(group["WallTime"].mean())
        )
        warm_sec = (
            float(warm_group["WallTime"].mean())
            if not warm_group.empty
            else float(group["WallTime"].mean())
        )
        rows_count = float(group["Rows"].mean()) if "Rows" in group else 0.0
        rows_per_sec = rows_count / warm_sec if warm_sec > 0 else 0.0
        agg_rows.append(
            {
                "config": str(config),
                "label": get_label(str(config)),
                "depth": int(depth),
                "cold_sec": round(cold_sec, 6),
                "warm_sec": round(warm_sec, 6),
                "rows_per_sec": round(rows_per_sec, 3),
                "boundary_tax_pct": round(float(group["BoundaryTax_pct"].mean()), 3),
                "samples": int(len(group)),
                "has_warm": bool(not warm_group.empty),
            }
        )

    baseline_by_depth = {
        row["depth"]: row["warm_sec"]
        for row in agg_rows
        if row["config"] == "B" and row["warm_sec"] > 0
    }
    max_runtime = 0.0
    for row in agg_rows:
        baseline = baseline_by_depth.get(row["depth"])
        row["speedup_vs_b"] = (
            round(baseline / row["warm_sec"], 3)
            if baseline and row["warm_sec"] > 0
            else None
        )
        max_runtime = max(max_runtime, row["cold_sec"], row["warm_sec"])

    series = []
    summaries = []
    for cfg in configs:
        points = sorted(
            [row for row in agg_rows if row["config"] == cfg],
            key=lambda row: row["depth"],
        )
        if not points:
            continue

        def slope(metric: str) -> float | None:
            valid = [p for p in points if p.get(metric) is not None]
            if len(valid) < 2:
                return None
            first, last = valid[0], valid[-1]
            depth_delta = last["depth"] - first["depth"]
            if depth_delta == 0:
                return None
            return round((float(last[metric]) - float(first[metric])) / depth_delta, 6)

        warm_slope = slope("warm_sec")
        cold_slope = slope("cold_sec")
        summaries.append(
            {
                "config": cfg,
                "label": get_label(cfg),
                "warm_slope_sec_per_depth": warm_slope,
                "cold_slope_sec_per_depth": cold_slope,
                "depth_count": len({p["depth"] for p in points}),
                "depth3_warm_sec": next(
                    (p["warm_sec"] for p in points if p["depth"] == 3),
                    None,
                ),
            }
        )
        series.append({"config": cfg, "label": get_label(cfg), "points": points})

    return {
        "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
        "depths": depths,
        "series": series,
        "summaries": summaries,
        "hasEnoughDepths": len(depths) >= 2,
        "maxRuntimeSec": round(max_runtime, 6),
    }


def _build_depth_runtime_section(chart_data: dict[str, Any]) -> str:
    payload_json = json.dumps(chart_data, ensure_ascii=True).replace("</", "<\\/")
    return """
<div class="card depth-card">
  <h2>Boundary Amplification: W0 Depth Scaling</h2>
  <p class="section-note">W0 is intentionally trivial. Any runtime growth with depth is mostly orchestration and engine–Python boundary-crossing cost, not model compute. This section shows whether repeated UDF stages amplify that tax.</p>
  <script type="application/json" id="depth-runtime-data">__PAYLOAD__</script>
  <div class="depth-controls">
    <label>Runtime view
      <select id="depth-runtime-mode">
        <option value="warm_sec" selected>Warm steady-state</option>
        <option value="cold_sec">Cold first run</option>
      </select>
    </label>
  </div>
  <div id="depth-runtime-chart" class="depth-chart-shell"></div>
  <div id="depth-summary-cards" class="depth-summary-grid"></div>
</div>
<style>
  .depth-card { border-color: #e7dcc7; background: linear-gradient(180deg, rgba(255,255,255,0.98), #fffaf0); }
  .depth-controls { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin: 12px 0 14px; }
  .depth-controls label { display: inline-flex; align-items: center; gap: 8px; color: #334155; font-size: 13px; font-weight: 700; }
  .depth-controls select { border: 1px solid #cbd5e1; border-radius: 10px; background: #fff; color: #0f172a; padding: 7px 10px; font: inherit; }
  .depth-chart-shell { min-height: 330px; border: 1px solid #e2e8f0; border-radius: 16px; background: #fff; padding: 12px; }
  .depth-svg { width: 100%; height: auto; display: block; }
  .depth-axis text { fill: #475569; font-size: 11px; }
  .depth-axis path, .depth-axis line { stroke: #cbd5e1; }
  .depth-grid line { stroke: #e2e8f0; stroke-dasharray: 3 4; }
  .depth-line { fill: none; stroke-width: 2.6; }
  .depth-point { stroke: #fff; stroke-width: 1.4; }
  .depth-empty { padding: 22px; border: 1px dashed #cbd5e1; border-radius: 14px; color: #475569; background: #fff; }
  .depth-legend { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; color: #475569; font-size: 12px; }
  .depth-legend-item { display: inline-flex; align-items: center; gap: 6px; }
  .depth-swatch { width: 12px; height: 12px; border-radius: 999px; display: inline-block; }
  .depth-summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin-top: 12px; }
  .depth-summary-card { border: 1px solid #e2e8f0; border-radius: 14px; background: #fff; padding: 12px; }
  .depth-summary-card .k { display: block; font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 800; }
  .depth-summary-card .v { display: block; color: #0f172a; font-size: 18px; font-weight: 850; margin-top: 4px; }
  .depth-summary-card .s { display: block; color: #64748b; font-size: 12px; margin-top: 3px; line-height: 1.35; }
  .depth-tooltip { position: fixed; z-index: 9999; pointer-events: none; opacity: 0; transition: opacity 120ms ease; background: rgba(15, 23, 42, 0.96); color: #f8fafc; border-radius: 12px; padding: 10px 12px; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22); font-size: 12px; line-height: 1.45; max-width: 280px; }
  .depth-tooltip strong { display: block; font-size: 13px; margin-bottom: 4px; color: #fff; }
  .depth-tooltip .metric { display: flex; justify-content: space-between; gap: 14px; white-space: nowrap; }
  .depth-tooltip .metric span:first-child { color: #cbd5e1; }
</style>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function() {
  const dataNode = document.getElementById("depth-runtime-data");
  const chartNode = document.getElementById("depth-runtime-chart");
  const cardsNode = document.getElementById("depth-summary-cards");
  const modeSelect = document.getElementById("depth-runtime-mode");
  if (!dataNode || !chartNode || !cardsNode || typeof d3 === "undefined") {
    return;
  }
  const payload = JSON.parse(dataNode.textContent);
  const colors = { A: "#8a8f98", B: "#e25a1c", C: "#3762e0", D: "#0f9f7a" };
  const tooltip = d3.select("body").selectAll(".depth-tooltip").data([null]).join("div").attr("class", "depth-tooltip");

  function fmtSec(value) {
    if (!Number.isFinite(value)) return "-";
    if (value >= 10) return `${value.toFixed(1)}s`;
    if (value >= 1) return `${value.toFixed(2)}s`;
    return `${(value * 1000).toFixed(1)}ms`;
  }

  function fmtRate(value) {
    if (!Number.isFinite(value)) return "-";
    return value >= 100 ? `${Math.round(value)}/s` : `${value.toFixed(1)}/s`;
  }

  function fmtSlope(value) {
    if (!Number.isFinite(value)) return "-";
    const sign = value > 0 ? "+" : "";
    return `${sign}${fmtSec(value)} / depth`;
  }

  function showTooltip(event, point, mode) {
    tooltip.style("opacity", 1).html(`
      <strong>Config ${point.config} · depth ${point.depth}</strong>
      <div class="metric"><span>Warm runtime</span><span>${fmtSec(point.warm_sec)}</span></div>
      <div class="metric"><span>Cold runtime</span><span>${fmtSec(point.cold_sec)}</span></div>
      <div class="metric"><span>Rows/sec</span><span>${fmtRate(point.rows_per_sec)}</span></div>
      <div class="metric"><span>Speedup vs B</span><span>${Number.isFinite(point.speedup_vs_b) ? `${point.speedup_vs_b.toFixed(2)}x` : "-"}</span></div>
      <div class="metric"><span>Boundary tax</span><span>${Number.isFinite(point.boundary_tax_pct) ? `${point.boundary_tax_pct.toFixed(1)}%` : "-"}</span></div>
      <div class="metric"><span>Plotted metric</span><span>${mode === "warm_sec" ? "Warm" : "Cold"}</span></div>
    `);
    const rect = tooltip.node().getBoundingClientRect();
    let left = event.clientX + 16;
    let top = event.clientY + 16;
    if (left + rect.width + 18 > window.innerWidth) left = event.clientX - rect.width - 16;
    if (top + rect.height + 18 > window.innerHeight) top = event.clientY - rect.height - 16;
    tooltip.style("left", `${Math.max(8, left)}px`).style("top", `${Math.max(8, top)}px`);
  }

  function renderCards(mode) {
    cardsNode.innerHTML = "";
    const summaries = payload.summaries || [];
    const spark = summaries.filter((d) => d.config === "A" || d.config === "B");
    const sail = summaries.filter((d) => d.config === "C" || d.config === "D");
    const slopeKey = mode === "warm_sec" ? "warm_slope_sec_per_depth" : "cold_slope_sec_per_depth";
    const sparkMax = d3.max(spark, (d) => d[slopeKey]);
    const sailMax = d3.max(sail, (d) => d[slopeKey]);
    const depth3Candidates = summaries.filter((d) => Number.isFinite(d.depth3_warm_sec));
    const bestDepth3 = depth3Candidates.sort((a, b) => a.depth3_warm_sec - b.depth3_warm_sec)[0];
    const cards = [
      { k: "Spark depth sensitivity", v: fmtSlope(sparkMax), s: "Largest A/B runtime slope across observed W0 depths." },
      { k: "Sail depth sensitivity", v: fmtSlope(sailMax), s: "Largest C/D runtime slope across observed W0 depths." },
      { k: "Best depth-3 config", v: bestDepth3 ? `Config ${bestDepth3.config}` : "-", s: bestDepth3 ? `${fmtSec(bestDepth3.depth3_warm_sec)} warm runtime at depth 3.` : "Depth 3 was not present." },
      { k: "Depth sweep coverage", v: payload.hasEnoughDepths ? `${payload.depths.length} depths` : "Insufficient", s: payload.hasEnoughDepths ? "Trend is based on multiple depths." : "Need at least two W0 depths to infer amplification." },
    ];
    cards.forEach((card) => {
      const node = document.createElement("div");
      node.className = "depth-summary-card";
      node.innerHTML = `<span class="k">${card.k}</span><span class="v">${card.v}</span><span class="s">${card.s}</span>`;
      cardsNode.appendChild(node);
    });
  }

  function render() {
    const mode = modeSelect.value || "warm_sec";
    chartNode.innerHTML = "";
    renderCards(mode);
    if (!payload.series || payload.series.length === 0) {
      chartNode.innerHTML = '<div class="depth-empty">No W0 depth-scaling rows were found for this run.</div>';
      return;
    }
    if (!payload.hasEnoughDepths) {
      chartNode.innerHTML = '<div class="depth-empty">Insufficient W0 depth sweep: only one depth was recorded. The boundary-amplification story needs at least two depths to avoid overclaiming.</div>';
      return;
    }

    const activeSeries = payload.series.map((series) => ({
      ...series,
      points: (series.points || []).filter((point) => Number.isFinite(point[mode])),
    })).filter((series) => series.points.length > 0);
    const width = chartNode.getBoundingClientRect().width || 960;
    const height = 330;
    const margin = { top: 20, right: 30, bottom: 42, left: 88 };
    const innerWidth = Math.max(220, width - margin.left - margin.right);
    const innerHeight = height - margin.top - margin.bottom;
    const maxRuntime = d3.max(activeSeries.flatMap((series) => series.points), (d) => d[mode]) || 1;
    const x = d3.scalePoint().domain((payload.depths || []).map(String)).range([0, innerWidth]).padding(0.35);
    const y = d3.scaleLinear().domain([0, maxRuntime * 1.12]).nice().range([innerHeight, 0]);
    const line = d3.line()
      .x((d) => x(String(d.depth)))
      .y((d) => y(d[mode]))
      .curve(d3.curveMonotoneX);
    const svg = d3.select(chartNode).append("svg")
      .attr("class", "depth-svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("role", "img")
      .attr("aria-label", "W0 runtime by pipeline depth");
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    g.append("g")
      .attr("class", "depth-grid")
      .call(d3.axisLeft(y).ticks(5).tickSize(-innerWidth).tickFormat(""))
      .selectAll("line")
      .attr("stroke", "#e2e8f0");
    g.append("g").attr("class", "depth-axis").call(d3.axisLeft(y).ticks(5).tickFormat((d) => fmtSec(Number(d))));
    g.append("g").attr("class", "depth-axis").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(x));
    g.append("text").attr("x", innerWidth / 2).attr("y", innerHeight + 36).attr("text-anchor", "middle").attr("fill", "#64748b").attr("font-size", 12).text("W0 chained-UDF pipeline depth");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -innerHeight / 2).attr("y", -70).attr("text-anchor", "middle").attr("fill", "#64748b").attr("font-size", 12).text(mode === "warm_sec" ? "Warm runtime" : "Cold runtime");

    const groups = g.selectAll(".depth-series").data(activeSeries).join("g").attr("class", "depth-series");
    groups.append("path")
      .attr("class", "depth-line")
      .attr("stroke", (d) => colors[d.config] || "#334155")
      .attr("d", (d) => line(d.points));
    groups.selectAll("circle")
      .data((d) => d.points.map((point) => ({ ...point, config: d.config, label: d.label })))
      .join("circle")
      .attr("class", "depth-point")
      .attr("cx", (d) => x(String(d.depth)))
      .attr("cy", (d) => y(d[mode]))
      .attr("r", 4)
      .attr("fill", (d) => colors[d.config] || "#334155")
      .on("pointerenter", (event, d) => showTooltip(event, d, mode))
      .on("pointermove", (event, d) => showTooltip(event, d, mode))
      .on("pointerleave", () => tooltip.style("opacity", 0));

    const legend = document.createElement("div");
    legend.className = "depth-legend";
    activeSeries.forEach((series) => {
      const row = document.createElement("span");
      row.className = "depth-legend-item";
      row.innerHTML = `<span class="depth-swatch" style="background:${colors[series.config] || "#334155"}"></span>Config ${series.config} · ${series.label}`;
      legend.appendChild(row);
    });
    chartNode.appendChild(legend);
  }

  modeSelect.addEventListener("change", render);
  render();
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => window.requestAnimationFrame(render));
    observer.observe(chartNode);
  } else {
    window.addEventListener("resize", render);
  }
})();
</script>
""".replace("__PAYLOAD__", payload_json)


def _build_disk_io_payload(run_df: pd.DataFrame) -> dict[str, Any]:
    configs = ["A", "B", "C", "D"]
    if run_df.empty:
        return {
            "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
            "workloads": [],
            "coverageRuns": 0,
            "totalRuns": 0,
            "metricKind": "output_materialization",
            "maxMetricMb": 0.0,
        }

    rows = run_df[
        [
            "Workload",
            "Config",
            "WallTime",
            "Rows",
            "MeasuredDiskWrite_MB",
            "MeasuredDiskWrite_Bytes",
            "MeasuredDiskScope",
            "OutputMaterialized_MB",
            "OutputMaterialized_Bytes",
            "WriteThroughput_MBps",
            "DiskTelemetryAvailable",
            "DiskWriteSource",
        ]
    ].copy()
    total_runs = int(len(rows))
    coverage_runs = int(rows["DiskTelemetryAvailable"].fillna(False).astype(bool).sum())
    positive_measured_runs = int((rows["MeasuredDiskWrite_Bytes"].fillna(0) > 0).sum())
    metric_kind = "runtime_writes" if positive_measured_runs > 0 else "output_materialization"
    agg = (
        rows.groupby(["Workload", "Config"], dropna=False, as_index=False)
        .agg(
            WallTime=("WallTime", "mean"),
            Rows=("Rows", "mean"),
            MeasuredDiskWrite_MB=("MeasuredDiskWrite_MB", "mean"),
            MeasuredDiskWrite_Bytes=("MeasuredDiskWrite_Bytes", "mean"),
            MeasuredDiskScopes=("MeasuredDiskScope", lambda vs: sorted({str(v) for v in vs if v})),
            OutputMaterialized_MB=("OutputMaterialized_MB", "mean"),
            OutputMaterialized_Bytes=("OutputMaterialized_Bytes", "mean"),
            WriteThroughput_MBps=("WriteThroughput_MBps", "mean"),
            DiskTelemetryCoverage=("DiskTelemetryAvailable", "mean"),
            DiskWriteSources=("DiskWriteSource", lambda vs: sorted({str(v) for v in vs if v})),
        )
        .sort_values(["Workload", "Config"])
    )

    workloads: list[dict[str, Any]] = []
    max_metric_mb = 0.0
    for workload in sorted(agg["Workload"].unique()):
        workload_df = agg[agg["Workload"] == workload].copy()
        workload_df["Config"] = pd.Categorical(workload_df["Config"], categories=configs, ordered=True)
        workload_df = workload_df.sort_values("Config")
        bars = []
        for _, row in workload_df.iterrows():
            config = str(row["Config"])
            measured_mb = round(float(row["MeasuredDiskWrite_MB"]), 3)
            output_mb = round(float(row["OutputMaterialized_MB"]), 3)
            primary_mb = measured_mb if metric_kind == "runtime_writes" else output_mb
            primary_source = "measured" if metric_kind == "runtime_writes" else "output_artifact_fallback"
            max_metric_mb = max(max_metric_mb, primary_mb)
            rows_value = float(row["Rows"])
            output_bytes = float(row["OutputMaterialized_Bytes"])
            bars.append(
                {
                    "config": config,
                    "label": get_label(config),
                    "wall_sec": round(float(row["WallTime"]), 6),
                    "rows": round(rows_value, 3),
                    "measured_write_mb": measured_mb,
                    "measured_write_bytes": int(round(float(row["MeasuredDiskWrite_Bytes"]))),
                    "write_throughput_mb_s": round(float(row["WriteThroughput_MBps"]), 3),
                    "output_materialized_mb": output_mb,
                    "output_materialized_bytes": int(round(output_bytes)),
                    "output_mb_per_1k_rows": round((output_mb / rows_value * 1000.0) if rows_value > 0 else 0.0, 6),
                    "disk_telemetry_coverage": round(float(row["DiskTelemetryCoverage"]) * 100.0, 1),
                    "disk_write_sources": list(row["DiskWriteSources"]),
                    "measured_scopes": list(row["MeasuredDiskScopes"]),
                    "primary_metric_mb": primary_mb,
                    "primary_source": primary_source,
                    "fallback": primary_source != "measured",
                }
            )
        workloads.append({"workload": workload, "bars": bars})

    return {
        "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
        "workloads": workloads,
        "coverageRuns": coverage_runs,
        "totalRuns": total_runs,
        "metricKind": metric_kind,
        "maxMetricMb": round(max_metric_mb, 3),
    }


def _build_disk_io_section(chart_data: dict[str, Any]) -> str:
    payload_json = json.dumps(chart_data, ensure_ascii=True).replace("</", "<\\/")
    legend_html = []
    for item in chart_data.get("configs", []):
        code = escape(str(item.get("code", "")))
        label = escape(str(item.get("label", "")))
        legend_html.append(
            f'<span class="disk-config-pill"><strong>{code}</strong><span>{label}</span></span>'
        )
    legend_markup = "".join(legend_html)
    subtitle = (
        "Measured runtime writes are available for __COVERAGE__ / __TOTAL__ runs. This D3 view compares true process-level write deltas where telemetry exists, and still shows final output materialization footprint for context."
        if chart_data.get("metricKind") == "runtime_writes"
        else "Measured runtime write telemetry was unavailable or zero for this run set, so the chart shows final output materialization footprint instead. This is artifact-size context, not observed spill or runtime disk pressure."
    )
    return """
<div class="card disk-card">
  <h2>Disk Telemetry &amp; Output Footprint</h2>
  <p class="section-note">__SUBTITLE__</p>
  <p class="section-note">Tooltips always separate measured runtime writes from output footprint and show provenance. Pattern-filled bars are not telemetry; they are fallback artifact-footprint values shown only so tiny completed runs do not look like missing data.</p>
  <div class="disk-legend">
    <div class="disk-legend-group">
      <span class="disk-legend-title">Execution configs</span>
      <div class="disk-config-grid">__LEGEND__</div>
    </div>
    <div class="disk-legend-group">
      <span class="disk-legend-title">Provenance</span>
      <div class="disk-config-grid">
        <span class="disk-config-pill"><span class="disk-swatch disk-solid"></span>Measured runtime writes</span>
        <span class="disk-config-pill"><span class="disk-swatch disk-pattern"></span>Output footprint fallback</span>
      </div>
    </div>
  </div>
  <div class="disk-chart-shell">
    <script type="application/json" id="disk-io-data">__PAYLOAD__</script>
    <div id="disk-io-chart" class="disk-chart-grid"></div>
  </div>
</div>
<style>
  .disk-card { border: 1px solid #dbe7ff; background: linear-gradient(180deg, rgba(255,255,255,0.98), #f8fafc); }
  .disk-legend { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 10px 0 16px; }
  .disk-legend-group { border: 1px solid #e2e8f0; border-radius: 14px; background: #fff; padding: 12px 14px; }
  .disk-legend-title { display: block; font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; margin-bottom: 10px; }
  .disk-config-grid { display: flex; flex-wrap: wrap; gap: 8px; }
  .disk-config-pill { display: inline-flex; align-items: center; gap: 8px; padding: 7px 10px; border-radius: 999px; border: 1px solid #e2e8f0; background: #f8fafc; color: #334155; font-size: 12px; line-height: 1.1; }
  .disk-config-pill strong { color: #0f172a; }
  .disk-config-pill span { white-space: nowrap; }
  .disk-swatch { width: 11px; height: 11px; border-radius: 999px; display: inline-block; flex: 0 0 auto; border: 1px solid #94a3b8; }
  .disk-solid { background: #7ea6d8; }
  .disk-pattern { background: repeating-linear-gradient(135deg, #d9a35f 0 3px, #fff7ed 3px 6px); }
  .disk-chart-grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
  .disk-panel { border: 1px solid #e2e8f0; border-radius: 16px; background: linear-gradient(180deg, #fff, #f8fafc); padding: 14px 14px 12px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04); }
  .disk-panel h3 { margin: 0; font-size: 16px; color: #0f172a; }
  .disk-panel .subtitle { margin: 3px 0 10px; font-size: 12px; color: #64748b; }
  .disk-svg { width: 100%; height: auto; display: block; }
  .disk-axis text { fill: #475569; font-size: 11px; }
  .disk-axis path, .disk-axis line { stroke: #cbd5e1; }
  .disk-grid line { stroke: #e2e8f0; stroke-dasharray: 3 4; }
  .disk-tooltip { position: fixed; z-index: 9999; pointer-events: none; opacity: 0; transition: opacity 120ms ease; background: rgba(15, 23, 42, 0.96); color: #f8fafc; border-radius: 12px; padding: 10px 12px; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22); font-size: 12px; line-height: 1.45; max-width: 290px; }
  .disk-tooltip strong { display: block; font-size: 13px; margin-bottom: 4px; color: #fff; }
  .disk-tooltip .metric { display: flex; justify-content: space-between; gap: 12px; white-space: nowrap; }
  .disk-tooltip .metric span:first-child { color: #cbd5e1; }
  .disk-empty { padding: 20px; border: 1px dashed #cbd5e1; border-radius: 14px; color: #475569; background: #fff; }
  .disk-bar:hover .disk-hit { fill: rgba(15, 23, 42, 0.03); }
  .disk-value-label { fill: #64748b; font-size: 11px; font-weight: 700; }
  @media (max-width: 760px) {
    .disk-legend { grid-template-columns: 1fr; }
  }
</style>
<script>
(function() {
  const dataNode = document.getElementById("disk-io-data");
  const chartNode = document.getElementById("disk-io-chart");
  if (!dataNode || !chartNode || typeof d3 === "undefined") {
    return;
  }

  const payload = JSON.parse(dataNode.textContent);
  const tooltip = d3.select("body").selectAll(".disk-tooltip")
    .data([null])
    .join("div")
    .attr("class", "disk-tooltip");
  const palette = { A: "#9e9e9e", B: "#e7a977", C: "#7ea6d8", D: "#7cc3b5" };

  function formatDiskValue(value) {
    if (!Number.isFinite(value)) {
      return "-";
    }
    if (value <= 0) {
      return "0 B";
    }
    if (value < 0.001) {
      return `${Math.round(value * 1_000_000)} B`;
    }
    if (value < 1) {
      return `${(value * 1000).toFixed(1)} KB`;
    }
    if (value < 10) {
      return `${value.toFixed(2)} MB`;
    }
    if (value < 100) {
      return `${value.toFixed(1)} MB`;
    }
    return `${Math.round(value)} MB`;
  }

  function showTooltip(event, datum) {
    const scopeLabel = datum.measured_scopes && datum.measured_scopes.length
      ? datum.measured_scopes.join(", ")
      : "unavailable";
    tooltip
      .style("opacity", 1)
      .html(`
        <strong>${datum.workload} · Config ${datum.config}</strong>
        <div class="metric"><span>Primary chart metric</span><span>${formatDiskValue(datum.primary_metric_mb)}</span></div>
        <div class="metric"><span>Measured runtime writes</span><span>${formatDiskValue(datum.measured_write_mb)}</span></div>
        <div class="metric"><span>Output materialized</span><span>${formatDiskValue(datum.output_materialized_mb)}</span></div>
        <div class="metric"><span>Provenance</span><span>${datum.primary_source === "measured" ? `measured (${scopeLabel})` : "output fallback"}</span></div>
        <div class="metric"><span>Runtime write throughput</span><span>${datum.write_throughput_mb_s.toFixed(3)} MB/s</span></div>
        <div class="metric"><span>Output / 1k rows</span><span>${datum.output_mb_per_1k_rows.toFixed(3)} MB</span></div>
        <div class="metric"><span>Wall time</span><span>${datum.wall_sec.toFixed(3)}s</span></div>
      `);
    const rect = tooltip.node().getBoundingClientRect();
    let left = event.clientX + 16;
    let top = event.clientY + 16;
    if (left + rect.width + 18 > window.innerWidth) left = event.clientX - rect.width - 16;
    if (top + rect.height + 18 > window.innerHeight) top = event.clientY - rect.height - 16;
    tooltip.style("left", `${Math.max(8, left)}px`).style("top", `${Math.max(8, top)}px`);
  }

  function hideTooltip() { tooltip.style("opacity", 0); }

  function render() {
    chartNode.innerHTML = "";
    if (!payload.workloads || payload.workloads.length === 0) {
      chartNode.innerHTML = '<div class="disk-empty">No disk telemetry or output materialization data was found for this run.</div>';
      return;
    }

    const containerWidth = chartNode.getBoundingClientRect().width || chartNode.clientWidth || 960;
    const minPanelWidth = 300;
    const gap = 16;
    const cols = Math.max(1, Math.min(3, Math.floor((containerWidth + gap) / minPanelWidth)));
    chartNode.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    const panelWidth = (containerWidth - gap * (cols - 1)) / cols;
    const panelHeight = 286;
    const margin = { top: 26, right: 20, bottom: 40, left: 84 };
    const innerWidth = Math.max(180, panelWidth - margin.left - margin.right);
    const innerHeight = panelHeight - margin.top - margin.bottom;
    const maxMetric = Math.max(0.001, (payload.maxMetricMb || 0) * 1.2);
    const yScale = d3.scaleLinear().domain([0, maxMetric]).nice().range([innerHeight, 0]);
    const xScale = d3.scaleBand().domain(payload.configs.map((d) => d.code)).range([0, innerWidth]).padding(0.22);
    const barWidth = xScale.bandwidth();
    const svgWidth = innerWidth + margin.left + margin.right;
    const tickCandidates = d3.ticks(0, maxMetric, maxMetric < 1 ? 6 : 5);

    payload.workloads.forEach((workload) => {
      const panel = chartNode.appendChild(document.createElement("div"));
      panel.className = "disk-panel";
      panel.innerHTML = `<h3>${workload.workload}</h3><div class="subtitle">${payload.metricKind === "runtime_writes" ? "Measured runtime disk writes when available" : "Final output materialization footprint"}</div>`;
      const svg = d3.select(panel)
        .append("svg")
        .attr("class", "disk-svg")
        .attr("viewBox", `0 0 ${svgWidth} ${panelHeight}`)
        .attr("role", "img")
        .attr("aria-label", `Disk comparison for workload ${workload.workload}`);
      const defs = svg.append("defs");
      workload.bars.forEach((bar) => {
        if (!bar.fallback) return;
        const patternId = `disk-pattern-${workload.workload.toLowerCase()}-${bar.config.toLowerCase()}`;
        const pattern = defs.append("pattern")
          .attr("id", patternId)
          .attr("patternUnits", "userSpaceOnUse")
          .attr("width", 8)
          .attr("height", 8)
          .attr("patternTransform", "rotate(135)");
        pattern.append("rect").attr("width", 8).attr("height", 8).attr("fill", "#fff7ed");
        pattern.append("rect").attr("width", 4).attr("height", 8).attr("fill", palette[bar.config] || "#d9a35f");
      });
      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
      g.append("g")
        .attr("class", "disk-grid")
        .call(d3.axisLeft(yScale).tickValues(tickCandidates).tickSize(-innerWidth).tickFormat(""))
        .selectAll("line")
        .attr("stroke", "#e2e8f0");
      g.append("g")
        .attr("class", "disk-axis")
        .call(d3.axisLeft(yScale).tickValues(tickCandidates).tickFormat((d) => formatDiskValue(Number(d))));
      g.append("g")
        .attr("class", "disk-axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).tickSizeOuter(0));
      const groups = g.selectAll(".disk-bar")
        .data(workload.bars)
        .join("g")
        .attr("class", "disk-bar")
        .attr("transform", (d) => `translate(${xScale(d.config)},0)`);
      groups.append("rect")
        .attr("class", "disk-hit")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", barWidth)
        .attr("height", innerHeight)
        .attr("fill", "transparent")
        .on("pointerenter", function(event, d) { showTooltip(event, { ...d, workload: workload.workload }); })
        .on("pointermove", function(event, d) { showTooltip(event, { ...d, workload: workload.workload }); })
        .on("pointerleave", hideTooltip);
      groups.append("rect")
        .attr("x", 0)
        .attr("y", (d) => yScale(d.primary_metric_mb))
        .attr("width", barWidth)
        .attr("height", (d) => Math.max(0, innerHeight - yScale(d.primary_metric_mb)))
        .attr("fill", (d) => {
          if (!d.fallback) return palette[d.config] || "#7ea6d8";
          return `url(#disk-pattern-${workload.workload.toLowerCase()}-${d.config.toLowerCase()})`;
        })
        .attr("stroke", (d) => palette[d.config] || "#7ea6d8")
        .attr("stroke-width", 1.2);
      groups.append("text")
        .attr("class", "disk-value-label")
        .attr("x", barWidth / 2)
        .attr("y", (d) => Math.max(12, yScale(d.primary_metric_mb) - 8))
        .attr("text-anchor", "middle")
        .text((d) => formatDiskValue(d.primary_metric_mb));
    });
  }

  render();
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => window.requestAnimationFrame(render));
    observer.observe(chartNode);
  } else {
    window.addEventListener("resize", render);
  }
})();
</script>
""".replace("__PAYLOAD__", payload_json).replace("__LEGEND__", legend_markup).replace(
        "__SUBTITLE__",
        subtitle.replace("__COVERAGE__", str(chart_data.get("coverageRuns", 0))).replace("__TOTAL__", str(chart_data.get("totalRuns", 0))),
    )


def _build_memory_section(chart_data: dict[str, Any]) -> str:
    payload_json = json.dumps(chart_data, ensure_ascii=True).replace("</", "<\\/")
    legend_html = []
    for item in chart_data.get("configs", []):
        code = escape(str(item.get("code", "")))
        label = escape(str(item.get("label", "")))
        legend_html.append(
            f'<span class="memory-config-pill"><strong>{code}</strong><span>{label}</span></span>'
        )
    legend_markup = "".join(legend_html)
    return """
<div class="card memory-card">
  <h2>Memory Comparison</h2>
  <p class="section-note">Peak RSS is averaged per workload and execution path, then rendered as a grouped D3 chart so memory differences stay readable across workloads without flattening small variations.</p>
  <div class="memory-legend">
    <span class="memory-legend-title">Execution configs</span>
    <div class="memory-config-grid">__LEGEND__</div>
  </div>
  <div class="memory-chart-shell">
    <script type="application/json" id="memory-comparison-data">__PAYLOAD__</script>
    <div id="memory-comparison-chart" class="memory-chart-grid"></div>
  </div>
</div>
<style>
  .memory-card { border: 1px solid #dbe7ff; background: linear-gradient(180deg, rgba(255,255,255,0.98), #f8fafc); }
  .memory-legend { border: 1px solid #e2e8f0; border-radius: 14px; background: #fff; padding: 12px 14px; margin: 10px 0 16px; }
  .memory-legend-title { display: block; font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; margin-bottom: 10px; }
  .memory-config-grid { display: flex; flex-wrap: wrap; gap: 8px; }
  .memory-config-pill { display: inline-flex; align-items: center; gap: 8px; padding: 7px 10px; border-radius: 999px; border: 1px solid #e2e8f0; background: #f8fafc; color: #334155; font-size: 12px; line-height: 1.1; }
  .memory-config-pill strong { color: #0f172a; }
  .memory-config-pill span { white-space: nowrap; }
  .memory-chart-shell { margin-top: 6px; }
  .memory-chart-grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
  .memory-panel { border: 1px solid #e2e8f0; border-radius: 16px; background: linear-gradient(180deg, #fff, #f8fafc); padding: 14px 14px 12px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04); }
  .memory-panel h3 { margin: 0; font-size: 16px; color: #0f172a; }
  .memory-panel .subtitle { margin: 3px 0 10px; font-size: 12px; color: #64748b; }
  .memory-svg { width: 100%; height: auto; display: block; }
  .memory-axis text { fill: #475569; font-size: 11px; }
  .memory-axis path, .memory-axis line { stroke: #cbd5e1; }
  .memory-grid line { stroke: #e2e8f0; stroke-dasharray: 3 4; }
  .memory-bar:hover .memory-hit { fill: rgba(15, 23, 42, 0.03); }
  .memory-tooltip { position: fixed; z-index: 9999; pointer-events: none; opacity: 0; transition: opacity 120ms ease; background: rgba(15, 23, 42, 0.96); color: #f8fafc; border-radius: 12px; padding: 10px 12px; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22); font-size: 12px; line-height: 1.45; max-width: 260px; }
  .memory-tooltip strong { display: block; font-size: 13px; margin-bottom: 4px; color: #fff; }
  .memory-tooltip .metric { display: flex; justify-content: space-between; gap: 12px; white-space: nowrap; }
  .memory-tooltip .metric span:first-child { color: #cbd5e1; }
  .memory-empty { padding: 20px; border: 1px dashed #cbd5e1; border-radius: 14px; color: #475569; background: #fff; }
  .memory-bar-fill-a { fill: #9e9e9e; }
  .memory-bar-fill-b { fill: #ff7043; }
  .memory-bar-fill-c { fill: #42a5f5; }
  .memory-bar-fill-d { fill: #26a69a; }
  @media (max-width: 760px) {
    .memory-legend { margin-bottom: 12px; }
  }
</style>
<script>
(function() {
  const dataNode = document.getElementById("memory-comparison-data");
  const chartNode = document.getElementById("memory-comparison-chart");
  if (!dataNode || !chartNode || typeof d3 === "undefined") {
    return;
  }

  const payload = JSON.parse(dataNode.textContent);
  const tooltip = d3.select("body").selectAll(".memory-tooltip")
    .data([null])
    .join("div")
    .attr("class", "memory-tooltip");
  const palette = { A: "#9e9e9e", B: "#ff7043", C: "#42a5f5", D: "#26a69a" };

  function formatMb(value) {
    return `${value.toFixed(1)} MB`;
  }

  function hideTooltip() {
    tooltip.style("opacity", 0);
  }

  function showTooltip(event, datum) {
    tooltip
      .style("opacity", 1)
      .html(`
        <strong>${datum.workload} · Config ${datum.config}</strong>
        <div class="metric"><span>Execution path</span><span>${datum.label}</span></div>
        <div class="metric"><span>Peak RSS</span><span>${formatMb(datum.peak_rss_mb)}</span></div>
      `);

    const pad = 18;
    const rect = tooltip.node().getBoundingClientRect();
    let left = event.clientX + 16;
    let top = event.clientY + 16;
    if (left + rect.width + pad > window.innerWidth) {
      left = event.clientX - rect.width - 16;
    }
    if (top + rect.height + pad > window.innerHeight) {
      top = event.clientY - rect.height - 16;
    }
    tooltip.style("left", `${Math.max(8, left)}px`).style("top", `${Math.max(8, top)}px`);
  }

  function render() {
    chartNode.innerHTML = "";
    if (!payload.workloads || payload.workloads.length === 0) {
      chartNode.innerHTML = '<div class="memory-empty">No memory data was found for this run.</div>';
      return;
    }

    const containerWidth = chartNode.getBoundingClientRect().width || chartNode.clientWidth || 960;
    const minPanelWidth = 300;
    const gap = 16;
    const cols = Math.max(1, Math.min(3, Math.floor((containerWidth + gap) / minPanelWidth)));
    chartNode.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;

    const panelWidth = (containerWidth - gap * (cols - 1)) / cols;
    const panelHeight = 286;
    const margin = { top: 26, right: 14, bottom: 38, left: 52 };
    const innerWidth = Math.max(180, panelWidth - margin.left - margin.right);
    const innerHeight = panelHeight - margin.top - margin.bottom;
    const maxPeak = Math.max(1, (payload.maxPeakRssMb || 0) * 1.2);
    const yScale = d3.scaleLinear().domain([0, maxPeak]).nice().range([innerHeight, 0]);
    const xScale = d3.scaleBand().domain(payload.configs.map((d) => d.code)).range([0, innerWidth]).padding(0.22);
    const barWidth = xScale.bandwidth();
    const svgWidth = innerWidth + margin.left + margin.right;

    payload.workloads.forEach((workload) => {
      const panel = chartNode.appendChild(document.createElement("div"));
      panel.className = "memory-panel";
      panel.innerHTML = `<h3>${workload.workload}</h3><div class="subtitle">Peak RSS by execution config</div>`;

      const svg = d3.select(panel)
        .append("svg")
        .attr("class", "memory-svg")
        .attr("viewBox", `0 0 ${svgWidth} ${panelHeight}`)
        .attr("role", "img")
        .attr("aria-label", `Memory comparison for workload ${workload.workload}`);

      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

      g.append("g")
        .attr("class", "memory-grid")
        .call(d3.axisLeft(yScale).ticks(4).tickSize(-innerWidth).tickFormat(""))
        .selectAll("line")
        .attr("stroke", "#e2e8f0");

      g.append("g")
        .attr("class", "memory-axis")
        .call(d3.axisLeft(yScale).ticks(4).tickFormat((d) => `${d} MB`));

      g.append("g")
        .attr("class", "memory-axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).tickSizeOuter(0));

      const barGroups = g.selectAll(".memory-bar")
        .data(workload.bars)
        .join("g")
        .attr("class", "memory-bar")
        .attr("transform", (d) => `translate(${xScale(d.config)},0)`);

      barGroups.append("rect")
        .attr("class", "memory-hit")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", barWidth)
        .attr("height", innerHeight)
        .attr("fill", "transparent")
        .on("pointerenter", function(event, d) {
          d3.select(this.parentNode).select("rect.memory-fill").attr("opacity", 0.92);
          showTooltip(event, d);
        })
        .on("pointermove", function(event, d) {
          showTooltip(event, d);
        })
        .on("pointerleave", function() {
          d3.select(this.parentNode).select("rect.memory-fill").attr("opacity", 1);
          hideTooltip();
        });

      barGroups.append("rect")
        .attr("class", (d) => `memory-fill memory-bar-fill-${d.config.toLowerCase()}`)
        .attr("x", 0)
        .attr("y", (d) => yScale(d.peak_rss_mb))
        .attr("width", barWidth)
        .attr("height", (d) => Math.max(0, innerHeight - yScale(d.peak_rss_mb)));

      barGroups.append("text")
        .attr("x", barWidth / 2)
        .attr("y", (d) => Math.max(12, yScale(d.peak_rss_mb) - 8))
        .attr("text-anchor", "middle")
        .attr("fill", "#64748b")
        .attr("font-size", 11)
        .attr("font-weight", 700)
        .text((d) => formatMb(d.peak_rss_mb));
    });
  }

  render();
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      window.requestAnimationFrame(render);
    });
    observer.observe(chartNode);
  } else {
    window.addEventListener("resize", render);
  }
})();
</script>
""".replace("__PAYLOAD__", payload_json).replace("__LEGEND__", legend_markup)


def _build_speedup_section(chart_data: dict[str, Any]) -> str:
    payload_json = json.dumps(chart_data, ensure_ascii=True).replace("</", "<\\/")
    legend_html = []
    for item in chart_data.get("configs", []):
        code = escape(str(item.get("code", "")))
        label = escape(str(item.get("label", "")))
        legend_html.append(
            f'<span class="speedup-config-pill"><strong>{code}</strong><span>{label}</span></span>'
        )
    legend_markup = "".join(legend_html)
    baseline_code = escape(str(chart_data.get("baselineConfig", "B")))
    baseline_label = escape(str(chart_data.get("baselineLabel", "Spark (Pandas/Arrow)")))
    return """
<div class="card speedup-card">
  <h2>Relative Speedups</h2>
  <p class="section-note">Speedup is computed against Spark's batched path (Config __BASELINE_CODE__, __BASELINE_LABEL__) for each workload and rendered on a log scale so both small regressions and large wins remain visible.</p>
  <div class="speedup-legend">
    <div class="speedup-legend-row">
      <span class="speedup-legend-title">Execution configs</span>
      <div class="speedup-config-grid">__LEGEND__</div>
    </div>
    <div class="speedup-legend-row">
      <span class="speedup-legend-title">Reference</span>
      <span class="speedup-baseline-pill"><span class="speedup-baseline-line"></span>1.0x baseline (Config __BASELINE_CODE__)</span>
    </div>
  </div>
  <div class="speedup-chart-shell">
    <script type="application/json" id="relative-speedups-data">__PAYLOAD__</script>
    <div id="relative-speedups-chart" class="speedup-chart-grid"></div>
  </div>
</div>
<style>
  .speedup-card { border: 1px solid #dbe7ff; background: linear-gradient(180deg, rgba(255,255,255,0.98), #f8fafc); }
  .speedup-legend { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 10px 0 16px; }
  .speedup-legend-row { border: 1px solid #e2e8f0; border-radius: 14px; background: #fff; padding: 12px 14px; }
  .speedup-legend-title { display: block; font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; margin-bottom: 10px; }
  .speedup-config-grid { display: flex; flex-wrap: wrap; gap: 8px; }
  .speedup-config-pill { display: inline-flex; align-items: center; gap: 8px; padding: 7px 10px; border-radius: 999px; border: 1px solid #e2e8f0; background: #f8fafc; color: #334155; font-size: 12px; line-height: 1.1; }
  .speedup-config-pill strong { color: #0f172a; }
  .speedup-config-pill span { white-space: nowrap; }
  .speedup-baseline-pill { display: inline-flex; align-items: center; padding: 7px 10px; border-radius: 999px; background: #eef2ff; color: #1e3a8a; border: 1px solid #c7d2fe; font-size: 12px; font-weight: 700; }
  .speedup-baseline-line { width: 18px; height: 0; border-top: 2px dashed #dc2626; margin-right: 8px; flex: 0 0 auto; }
  .speedup-chart-shell { margin-top: 6px; }
  .speedup-chart-grid { display: block; }
  .speedup-panel { border: 1px solid #e2e8f0; border-radius: 16px; background: linear-gradient(180deg, #fff, #f8fafc); padding: 14px 14px 12px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04); }
  .speedup-panel h3 { margin: 0; font-size: 16px; color: #0f172a; }
  .speedup-panel .subtitle { margin: 3px 0 10px; font-size: 12px; color: #64748b; }
  .speedup-svg { width: 100%; height: auto; display: block; }
  .speedup-axis text { fill: #475569; font-size: 11px; }
  .speedup-axis path, .speedup-axis line { stroke: #cbd5e1; }
  .speedup-grid line { stroke: #e2e8f0; stroke-dasharray: 3 4; }
  .speedup-bar:hover .speedup-hit { fill: rgba(15, 23, 42, 0.03); }
  .speedup-tooltip { position: fixed; z-index: 9999; pointer-events: none; opacity: 0; transition: opacity 120ms ease; background: rgba(15, 23, 42, 0.96); color: #f8fafc; border-radius: 12px; padding: 10px 12px; box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22); font-size: 12px; line-height: 1.45; max-width: 260px; }
  .speedup-tooltip strong { display: block; font-size: 13px; margin-bottom: 4px; color: #fff; }
  .speedup-tooltip .metric { display: flex; justify-content: space-between; gap: 12px; white-space: nowrap; }
  .speedup-tooltip .metric span:first-child { color: #cbd5e1; }
  .speedup-empty { padding: 20px; border: 1px dashed #cbd5e1; border-radius: 14px; color: #475569; background: #fff; }
  .speedup-bar-fill-b { fill: #ff7043; }
  .speedup-bar-fill-c { fill: #42a5f5; }
  .speedup-bar-fill-d { fill: #26a69a; }
  .speedup-reference-line { stroke: #dc2626; stroke-dasharray: 5 4; stroke-width: 1.5; }
  .speedup-reference-label { fill: #dc2626; font-size: 11px; font-weight: 700; }
  @media (max-width: 760px) {
    .speedup-legend { grid-template-columns: 1fr; }
  }
</style>
<script>
(function() {
  const dataNode = document.getElementById("relative-speedups-data");
  const chartNode = document.getElementById("relative-speedups-chart");
  if (!dataNode || !chartNode || typeof d3 === "undefined") {
    return;
  }

  const payload = JSON.parse(dataNode.textContent);
  const tooltip = d3.select("body").selectAll(".speedup-tooltip")
    .data([null])
    .join("div")
    .attr("class", "speedup-tooltip");
  const palette = { B: "#ff7043", C: "#42a5f5", D: "#26a69a" };

  function formatSpeedup(value) {
    return `${value.toFixed(2)}x`;
  }

  function hideTooltip() {
    tooltip.style("opacity", 0);
  }

  function showTooltip(event, datum) {
    tooltip
      .style("opacity", 1)
      .html(`
        <strong>${datum.workload} · Config ${datum.config}</strong>
        <div class="metric"><span>Speedup</span><span>${formatSpeedup(datum.speedup)}</span></div>
        <div class="metric"><span>Wall time</span><span>${datum.wall_sec.toFixed(3)}s</span></div>
      `);

    const pad = 18;
    const rect = tooltip.node().getBoundingClientRect();
    let left = event.clientX + 16;
    let top = event.clientY + 16;
    if (left + rect.width + pad > window.innerWidth) {
      left = event.clientX - rect.width - 16;
    }
    if (top + rect.height + pad > window.innerHeight) {
      top = event.clientY - rect.height - 16;
    }
    tooltip.style("left", `${Math.max(8, left)}px`).style("top", `${Math.max(8, top)}px`);
  }

  function render() {
    chartNode.innerHTML = "";
    if (!payload.workloads || payload.workloads.length === 0) {
      chartNode.innerHTML = '<div class="speedup-empty">No speedup data was found for this run.</div>';
      return;
    }

    const width = chartNode.getBoundingClientRect().width || chartNode.clientWidth || 960;
    const margin = { top: 60, right: 12, bottom: 72, left: 52 };
    const height = 420;
    const innerWidth = Math.max(200, width - margin.left - margin.right);
    const innerHeight = height - margin.top - margin.bottom;
    const workloads = payload.workloads.map((d) => d.workload);
    const configs = payload.configs.map((d) => d.code);
    const bars = payload.workloads.flatMap((d) => d.bars.map((bar) => ({ ...bar, workload: d.workload })));
    const minSpeedup = Math.max(0.5, Math.min(payload.minSpeedup || 0.5, 0.8) * 0.85);
    const maxSpeedup = Math.max(1.2, (payload.maxSpeedup || 1.2) * 1.4);
    const xScale = d3.scaleBand().domain(workloads).range([0, innerWidth]).padding(0.24);
    const barGap = 8;
    const barWidth = Math.min(40, Math.max(14, (xScale.bandwidth() - barGap * (configs.length - 1)) / configs.length));
    const barPitch = barWidth + barGap;
    const yScale = d3.scaleLog().domain([minSpeedup, maxSpeedup]).range([innerHeight, 0]).nice();
    const svgWidth = innerWidth + margin.left + margin.right;

    const panel = chartNode.appendChild(document.createElement("div"));
    panel.className = "speedup-panel";
    panel.innerHTML = '<h3>Relative Speedups</h3><div class="subtitle">Speedup vs Spark batched baseline, log scale</div>';

    const svg = d3.select(panel)
      .append("svg")
      .attr("class", "speedup-svg")
      .attr("viewBox", `0 0 ${svgWidth} ${height}`)
      .attr("role", "img")
      .attr("aria-label", "Relative speedups across workloads");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    let yTicks = [0.5, 1, 2, 4, 8, 16].filter((tick) => tick >= minSpeedup && tick <= maxSpeedup);
    if (yTicks.length < 3) {
      yTicks = [0.5, 1, 2].filter((tick) => tick >= minSpeedup && tick <= maxSpeedup);
    }

    g.append("g")
      .attr("class", "speedup-grid")
      .call(d3.axisLeft(yScale).tickValues(yTicks).tickSize(-innerWidth).tickFormat(""))
      .selectAll("line")
      .attr("stroke", "#e2e8f0");

    g.append("g")
      .attr("class", "speedup-axis")
      .call(d3.axisLeft(yScale).tickValues(yTicks).tickFormat((d) => `${d}x`));

    g.append("line")
      .attr("class", "speedup-reference-line")
      .attr("x1", 0)
      .attr("x2", innerWidth)
      .attr("y1", yScale(1))
      .attr("y2", yScale(1));

    const barGroups = g.selectAll(".speedup-bar")
      .data(bars)
      .join("g")
      .attr("class", "speedup-bar")
      .attr("transform", (d) => `translate(${xScale(d.workload) + configs.indexOf(d.config) * barPitch},0)`);

    barGroups.append("rect")
      .attr("class", "speedup-hit")
      .attr("x", 0)
      .attr("y", (d) => Math.min(yScale(d.speedup), innerHeight - 4))
      .attr("width", barWidth)
      .attr("height", (d) => Math.max(4, innerHeight - Math.min(yScale(d.speedup), innerHeight - 4)))
      .attr("fill", "transparent")
      .on("pointerenter", function(event, d) {
        d3.select(this.parentNode).select("rect.speedup-fill").attr("opacity", 0.92);
        showTooltip(event, d);
      })
      .on("pointermove", function(event, d) {
        showTooltip(event, d);
      })
      .on("pointerleave", function() {
        d3.select(this.parentNode).select("rect.speedup-fill").attr("opacity", 1);
        hideTooltip();
      });

    barGroups.append("rect")
      .attr("class", (d) => `speedup-fill speedup-bar-fill-${d.config.toLowerCase()}`)
      .attr("x", 0)
      .attr("y", (d) => yScale(d.speedup))
      .attr("width", barWidth)
      .attr("height", (d) => Math.max(0, innerHeight - yScale(d.speedup)))
      .style("pointer-events", "none");

    barGroups.append("text")
      .attr("x", barWidth / 2)
      .attr("y", (d) => Math.max(12, yScale(d.speedup) - 8))
      .attr("text-anchor", "middle")
      .attr("fill", "#64748b")
      .attr("font-size", 11)
      .attr("font-weight", 700)
      .text((d) => formatSpeedup(d.speedup));

    g.selectAll(".speedup-config-label")
      .data(bars)
      .join("text")
      .attr("class", "speedup-config-label")
      .attr("x", (d) => xScale(d.workload) + configs.indexOf(d.config) * barPitch + barWidth / 2)
      .attr("y", innerHeight + 18)
      .attr("text-anchor", "middle")
      .attr("fill", "#475569")
      .attr("font-size", 11)
      .attr("font-weight", 700)
      .text((d) => d.config);

    g.selectAll(".speedup-workload-label")
      .data(workloads)
      .join("text")
      .attr("class", "speedup-workload-label")
      .attr("x", (d) => xScale(d) + xScale.bandwidth() / 2)
      .attr("y", -14)
      .attr("text-anchor", "middle")
      .attr("fill", "#0f172a")
      .attr("font-size", 12)
      .attr("font-weight", 800)
      .text((d) => d);

  }

  render();
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      window.requestAnimationFrame(render);
    });
    observer.observe(chartNode);
  } else {
    window.addEventListener("resize", render);
  }
})();
</script>
""".replace("__PAYLOAD__", payload_json).replace("__LEGEND__", legend_markup).replace("__BASELINE_CODE__", baseline_code).replace("__BASELINE_LABEL__", baseline_label)


def _write_html(
    results_dir: Path,
    report_dir: Path,
    summary_df: pd.DataFrame,
    run_df: pd.DataFrame,
) -> None:
    rows = summary_df.copy()
    rows["DepthDisplay"] = rows["Depth"].apply(lambda x: int(x) if pd.notna(x) else "-")
    rows["LogoSVG"] = rows["Config"].map(_logo_svg)
    rows["RowClass"] = rows["Config"].map(_row_class)
    rows["SpeedupClass"] = rows.apply(lambda row: _speedup_class(row["Config"], row["Speedup_x"]), axis=1)
    tel_cards, tel_maxes = _build_tel_cards(rows)
    overhead_breakdown_section = _build_overhead_breakdown_section(
        _build_overhead_breakdown_payload(run_df)
    )
    disk_io_section = _build_disk_io_section(_build_disk_io_payload(run_df))
    memory_section = _build_memory_section(_build_memory_payload(run_df))
    speedup_section = _build_speedup_section(_build_speedup_payload(run_df))
    gpu_utilization_section = _build_gpu_utilization_section(
        _build_gpu_utilization_payload(results_dir)
    )
    depth_runtime_section = _build_depth_runtime_section(
        _build_depth_runtime_payload(run_df)
    )
    context = _load_report_context(results_dir)

    template = Template(
        """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sail vs Spark Benchmark Report</title>
  <style>
    body { font-family: 'Avenir Next', 'IBM Plex Sans', 'Segoe UI', sans-serif; margin: 0 auto; max-width: 1240px; padding: 24px; color: #1f2937; background: linear-gradient(180deg, #eef4ff 0%, #f8fafc 18%, #f8fafc 100%); }
    h1, h2, h3 { color: #0f172a; margin-top: 0; }
    .hero { background: radial-gradient(circle at top left, rgba(55, 98, 224, 0.18), rgba(255,255,255,0.96) 42%), white; border: 1px solid #dbe7ff; border-radius: 18px; padding: 28px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06); }
    .hero p { max-width: 900px; line-height: 1.68; color: #334155; }
    .hero-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }
    .hero-chip { background: rgba(255,255,255,0.9); border: 1px solid #dbe7ff; border-radius: 14px; padding: 12px 14px; }
    .hero-chip .k { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; margin-bottom: 4px; }
    .hero-chip .v { font-size: 18px; font-weight: 700; color: #0f172a; }
    .hero-configs { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }
    .card { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04); }
    .lede { color: #475569; margin-top: 6px; margin-bottom: 0; line-height: 1.55; }
    .grid-2 { display: grid; grid-template-columns: 1.1fr 1fr; gap: 18px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .mini-card { border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px; background: linear-gradient(180deg, #fff, #f8fafc); }
    .mini-card p { margin: 6px 0 0 0; color: #475569; line-height: 1.45; font-size: 14px; }
    .mini-card code { font-size: 12px; white-space: normal; word-break: break-word; }
    .config-card.sail-row { border-color: #c7d2fe; background: linear-gradient(180deg, rgba(55, 98, 224, 0.08), rgba(255,255,255,0.98)); }
    .config-card.spark-row { border-color: #fed7aa; background: linear-gradient(180deg, rgba(226, 90, 28, 0.08), rgba(255,255,255,0.98)); }
    .config-head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
    .section-note { color: #475569; line-height: 1.55; margin: 0 0 14px 0; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { border: 1px solid #e5e7eb; padding: 10px 12px; text-align: left; font-size: 14px; }
    th { background: #0f172a; color: white; }
    tr:nth-child(even) { background: #f8fafc; }
    tr.spark-row td:first-child, tr.spark-row td:nth-child(2) { font-weight: 600; }
    tr.sail-row { background: linear-gradient(90deg, rgba(55, 98, 224, 0.10), rgba(255, 255, 255, 0)); }
    tr.sail-row:nth-child(even) { background: linear-gradient(90deg, rgba(55, 98, 224, 0.14), rgba(248, 250, 252, 0.7)); }
    tr.sail-row td { border-color: #c7d2fe; }
    tr.sail-row td:first-child, tr.sail-row td:nth-child(2) { font-weight: 700; color: #173089; }
    .setup-cell { display: flex; align-items: center; gap: 10px; }
    .setup-cell svg { flex: 0 0 auto; }
    .setup-name { line-height: 1.2; }
    .speedup { display: inline-flex; align-items: center; justify-content: center; min-width: 72px; padding: 6px 10px; border-radius: 999px; font-weight: 800; letter-spacing: 0.01em; }
    .speedup-spark-sm { background: #e5e7eb; color: #374151; }
    .speedup-spark-md { background: #ffeadf; color: #c2410c; }
    .speedup-spark-lg { background: #ffd9c7; color: #9a3412; box-shadow: inset 0 0 0 1px rgba(226, 90, 28, 0.18); }
    .speedup-spark-xl { background: linear-gradient(135deg, #e25a1c, #fb923c); color: white; box-shadow: 0 8px 18px rgba(226, 90, 28, 0.24); }
    .speedup-sail-sm { background: #e5e7eb; color: #374151; }
    .speedup-sail-md { background: #dbeafe; color: #1d4ed8; }
    .speedup-sail-lg { background: #bfdbfe; color: #1e3a8a; box-shadow: inset 0 0 0 1px rgba(30, 64, 175, 0.15); }
    .speedup-sail-xl { background: linear-gradient(135deg, #1d4ed8, #3762e0); color: white; box-shadow: 0 8px 18px rgba(55, 98, 224, 0.25); }
    .speedup-cell { white-space: nowrap; }
    .img-container { margin-top: 16px; }
    .img-container img { width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; }
    .spec-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 18px; }
    .spec-item { border-bottom: 1px dashed #e5e7eb; padding-bottom: 8px; }
    .spec-item .k { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
    .spec-item .v { display: block; font-size: 15px; font-weight: 600; color: #0f172a; margin-top: 2px; }
    .knob-shell { background: linear-gradient(180deg, #fff, #f8fafc); border: 1px solid #e2e8f0; border-radius: 18px; padding: 18px; box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04); }
    .knob-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-top: 10px; }
    .knob-card { background: linear-gradient(180deg, #ffffff, #f8fafc); border: 1px solid #e2e8f0; border-radius: 16px; padding: 16px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04); }
    .knob-card-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
    .knob-card-head strong { font-size: 15px; color: #0f172a; }
    .knob-code { display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 999px; background: #e0f2fe; color: #075985; font-size: 11px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; }
    .knob-specs { display: grid; grid-template-columns: 1fr; gap: 8px; }
    .knob-spec { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 8px 10px; border-radius: 12px; background: #ffffff; border: 1px solid #e5e7eb; }
    .knob-spec .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
    .knob-spec .v { font-size: 13px; font-weight: 700; color: #0f172a; text-align: right; }
    .diagram-list { display: grid; grid-template-columns: 1fr; gap: 14px; }
    .diagram-card { padding: 28px 24px; border: 1px solid #e5e7eb; border-radius: 18px; background: linear-gradient(180deg, #fff, #f8fafc); margin-top: 12px; overflow-x: auto; }
    .diagram-card .mermaid { display: block; }
    .diagram-card .mermaid svg { display: block; width: 100%; height: auto; margin: 0 auto;}
    .tel-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; margin-top: 8px; }
    .tel-card { border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden; }
    .tel-card-header { background: #0f172a; color: #fff; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; padding: 8px 14px; }
    .tel-cfg-row { display: flex; align-items: center; gap: 14px; padding: 10px 14px; border-bottom: 1px solid #f1f5f9; }
    .tel-cfg-row:last-child { border-bottom: none; }
    .tel-cfg-row.sail-row { background: rgba(55,98,224,0.04); border-left: 3px solid #c7d2fe; }
    .tel-cfg-row.spark-row { border-left: 3px solid #fed7aa; }
    .tel-cfg-id { display: flex; align-items: center; gap: 6px; min-width: 68px; }
    .tel-cfg-id strong { font-size: 13px; color: #0f172a; }
    .tel-metrics { display: flex; align-items: flex-end; gap: 18px; flex: 1; flex-wrap: wrap; }
    .tel-m { display: flex; flex-direction: column; gap: 2px; }
    .tel-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; white-space: nowrap; }
    .tel-val { font-size: 13px; font-weight: 600; color: #334155; white-space: nowrap; }
    .tel-val.winner { color: #1d4ed8; font-weight: 800; }
    .tel-cfg-row.spark-row .tel-val.winner { color: #b45309; }
    .bar-track { width: 72px; height: 4px; background: #e2e8f0; border-radius: 2px; overflow: hidden; margin-top: 3px; }
    .bar-fill { height: 100%; background: #3762e0; border-radius: 2px; min-width: 3px; }
    .tel-cfg-row.spark-row .bar-fill { background: #e25a1c; }
  </style>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });
  </script>
</head>
<body>
  <div class="hero">
    <h1>Sail vs Spark Benchmark Report</h1>
    <p>{{ experiment_blurb }}</p>
    <div class="hero-strip">
      <div class="hero-chip"><span class="k">Workloads</span><span class="v">{{ workloads|length }}</span></div>
      <div class="hero-chip"><span class="k">Execution Paths</span><span class="v">{{ configs|length }}</span></div>
      <div class="hero-chip"><span class="k">Profile</span><span class="v">{{ run_specs[0][1] if run_specs else "-" }}</span></div>
      <div class="hero-chip"><span class="k">Dataset Rows</span><span class="v">{{ run_specs[8][1] if run_specs else "-" }}</span></div>
    </div>
  </div>
  <div class="grid-2">
    <div class="card">
      <h2>Benchmark Design</h2>
      <p class="lede">Each workload runs identical model logic across four execution configurations: two through standard Spark UDF mechanisms (row-level pickle and Pandas/Arrow batch) and two through Sail-native paths (Arrow shared-memory and SQL-native UDTF). The configurations vary in boundary-crossing frequency, serialization format, and the degree to which data movement remains inside the engine.</p>
    </div>
    <div class="card">
      <h2>Metrics</h2>
      <p class="lede">The summary table reports cold-start latency, steady-state runtime, rows per second, traced UDF time, traced transfer time, overhead tax (fraction of wall time outside model compute), speedup relative to Config A, and peak RSS. Plots decompose these into per-phase time budgets. Untimed wall-clock regions represent time outside explicitly instrumented compute and transfer spans.</p>
    </div>
  </div>
  <div class="card">
    <h2>Workloads</h2>
    <p class="section-note">The five workloads represent structurally distinct inference patterns — from trivial chained transforms that expose pure orchestration cost, to scored best-of-N generation, batched throughput, embedding pipelines, and multi-step agentic loops. Coverage across these shapes ensures results are not specific to a single access pattern.</p>
    <div class="diagram-list">
      {% for item in workloads %}
      <div class="mini-card">
        <div class="config-head"><strong>{{ item.label }}</strong><span class="knob-code">{{ item.code }}</span></div>
        <p>{{ item.description }}</p>
        <div class="diagram-card">
          {{ item.svg | safe }}
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
  <div class="card">
    <h2>Configurations</h2>
    <p class="section-note">Each configuration represents a distinct data transport strategy between the execution engine and the Python model runtime. Workload logic is held constant; what varies is boundary-crossing frequency, serialization format, and the degree to which data movement remains inside the engine.</p>
    <div class="diagram-list">
      {% for item in configs %}
      <div class="mini-card config-card {{ item.row_class }}">
        <div class="config-head">{{ item.logo | safe }}<strong>{{ item.code }}</strong></div>
        <div><strong>{{ item.label }}</strong></div>
        <p>{{ item.description }}</p>
        <div class="diagram-card">
          {{ item.svg | safe }}
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
  <div class="card">
    <h2>Run Configuration</h2>
    <p class="section-note">Hardware, software, and model configuration for this benchmark run. Workload knobs are included here because they define the exact workload shape used for the measurements.</p>
    <div class="grid-2">
      <div class="mini-card">
        <h3>Environment</h3>
        <div class="spec-grid">
          {% for key, value in run_specs %}
          <div class="spec-item">
            <span class="k">{{ key }}</span>
            <span class="v">{{ value }}</span>
          </div>
          {% endfor %}
        </div>
      </div>
      <div class="mini-card">
        <h3>Models</h3>
        {% for item in model_specs %}
        <div class="spec-item">
          <span class="k">{{ item.role }}</span>
          <span class="v">{{ item.name }}</span>
        </div>
        {% endfor %}
      </div>
    </div>
    <h3 style="margin-top:18px;">Workload Knobs</h3>
    <div class="knob-grid">
      {% for item in workload_knobs %}
      <div class="knob-card">
        <div class="knob-card-head">
          <strong>{{ item.name }}</strong>
          <span class="knob-code">{{ item.badge }}</span>
        </div>
        <div class="knob-specs">
          {% for param in item.params %}
          <div class="knob-spec">
            <span class="k">{{ param.key }}</span>
            <span class="v">{{ param.value }}</span>
          </div>
          {% endfor %}
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
  <div class="card">
    <h2>Performance Summary</h2>
    <p class="section-note">Cold reflects first-run latency including JIT compilation, model loading, and executor initialization. Warm is the steady-state runtime across subsequent iterations; — indicates a single sample was collected. Speedup is normalized to Spark's batched path (Config B) at the same workload and recursion depth.</p>
    <table>
      <thead>
        <tr>
          <th>Workload</th>
          <th>Setup</th>
          <th>Depth</th>
          <th>Cold</th>
          <th>Warm</th>
          <th>Rows/s</th>
          <th>UDF</th>
          <th>Transfer</th>
          <th>Overhead %</th>
          <th>Speedup</th>
          <th>Peak RSS (MB)</th>
        </tr>
      </thead>
      <tbody>
      {% for _, row in rows.iterrows() %}
        <tr class="{{ row["RowClass"] }}">
          <td>{{ row["Workload"] }}</td>
          <td>
            <div class="setup-cell">
              {{ row["LogoSVG"] | safe }}
              <span class="setup-name">{{ row["Setup"] }}</span>
            </div>
          </td>
          <td>{{ row["DepthDisplay"] }}</td>
          <td>{{ row["Setup (Cold)"] }}</td>
          <td>{{ row["Steady (Warm)"] }}</td>
          <td>{{ "%.2f"|format(row["RowsPerSec"]) }}</td>
          <td>{{ row["UDF Time"] }}</td>
          <td>{{ row["Transfer Time"] }}</td>
          <td>{{ "%.2f"|format(row["Overhead Tax (%)"]) }}</td>
          <td class="speedup-cell">
            <span class="{{ row["SpeedupClass"] }}">{{ row["Speedup"] }}</span>
          </td>
          <td>{{ "%.1f"|format(row["Peak RSS (MB)"]) }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  <div class="card">
    <h2>Runtime Telemetry</h2>
      <p class="section-note">Aggregated telemetry from the runtime collector. Surfaces secondary cost signals — GPU pipeline continuity, peak memory pressure, and measured runtime writes — that are not captured in wall-clock timing alone. When disk-write telemetry is unavailable or zero, the disk cell is labeled Output footprint and should be read as artifact-size context rather than telemetry. Bold values indicate the best result for that metric within each workload.</p>
    <div class="tel-grid">
      {% for card in tel_cards %}
      <div class="tel-card">
        <div class="tel-card-header">{{ card.workload }}</div>
        {% for cfg in card.configs %}
        {% set gpu_pct  = [(cfg["Peak GPU Util (%)"]  / tel_maxes["Peak GPU Util (%)"]  * 100)|round|int, 2]|max %}
        {% set cont_pct = cfg["ContinuityBarPct"] %}
        <div class="tel-cfg-row {{ cfg['RowClass'] }}">
          <div class="tel-cfg-id">
            {{ cfg['LogoSVG'] | safe }}
            <strong>{{ cfg['Config'] }}</strong>
          </div>
          <div class="tel-metrics">
            <div class="tel-m">
              <span class="tel-lbl">GPU Util</span>
              <div class="bar-track"><div class="bar-fill" style="width:{{ gpu_pct }}%"></div></div>
              <span class="tel-val {{ 'winner' if cfg['BestGPU'] else '' }}">{{ "%.0f"|format(cfg["Peak GPU Util (%)"]) }}%</span>
            </div>
            <div class="tel-m">
              <span class="tel-lbl">Continuity</span>
              <div class="bar-track"><div class="bar-fill" style="width:{{ cont_pct }}%"></div></div>
              <span class="tel-val {{ 'winner' if cfg['BestCont'] else '' }}">{{ cfg["ContinuityDisplay"] }}</span>
            </div>
            <div class="tel-m">
              <span class="tel-lbl">CPU</span>
              <span class="tel-val">{{ "%.0f"|format(cfg["Avg CPU (%)"]) }}%</span>
            </div>
            <div class="tel-m">
              <span class="tel-lbl">GPU Power</span>
              <span class="tel-val">{{ "%.0f"|format(cfg["Avg GPU Power (W)"]) }}W</span>
            </div>
            <div class="tel-m">
              <span class="tel-lbl">{{ cfg["DiskDisplayLabel"] }}</span>
              <span class="tel-val {{ 'winner' if cfg['BestDisk'] else '' }}">{{ "%.2f"|format(cfg["Disk Write (MB)"]) }}MB</span>
            </div>
            <div class="tel-m">
              <span class="tel-lbl">Trace Events</span>
              <span class="tel-val">{{ cfg["Trace Events"] }}</span>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
      {% endfor %}
    </div>
  </div>
  {{ depth_runtime_section | safe }}
  {{ gpu_utilization_section | safe }}
  {{ overhead_breakdown_section | safe }}
  {{ disk_io_section | safe }}
  {{ memory_section | safe }}
  {{ speedup_section | safe }}
</body>
</html>
        """
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "aggregate.html").write_text(
        template.render(
            rows=rows,
            tel_cards=tel_cards,
            tel_maxes=tel_maxes,
            depth_runtime_section=depth_runtime_section,
            gpu_utilization_section=gpu_utilization_section,
            overhead_breakdown_section=overhead_breakdown_section,
            disk_io_section=disk_io_section,
            memory_section=memory_section,
            speedup_section=speedup_section,
            **context,
        )
    )


def _run_plot_scripts(results_dir: Path, plots_dir: Path) -> None:
    if not PLOT_SCRIPTS:
        return
    print("Generating plots...", flush=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = plots_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["MPLBACKEND"] = "Agg"
    for script, filename in PLOT_SCRIPTS:
        script_path = Path(script)
        if not script_path.exists():
            continue
        cmd = [
            sys.executable,
            str(script_path),
            "--results_dir",
            str(results_dir),
            "--out",
            str(plots_dir / filename),
        ]
        print(f"Running {script}...", flush=True)
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: failed to run {script}: {exc}")


def aggregate(results_dir: str, report_dir: str | None = None) -> pd.DataFrame:
    path = Path(results_dir)
    out_dir = Path(report_dir) if report_dir else path / "report"
    run_df = _build_run_df(path)
    if run_df.empty:
        print("No results found.")
        return run_df

    summary_df = _build_summary_df(run_df)
    _write_csvs(out_dir, run_df, summary_df)
    _write_markdown(out_dir, summary_df)
    _run_plot_scripts(path, out_dir / "plots")
    _write_html(path, out_dir, summary_df, run_df)
    print(f"Wrote aggregate outputs under {out_dir}")
    return summary_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    aggregate(args.results_dir, report_dir=args.report_dir)
    report_path = Path(args.report_dir) / "aggregate.html" if args.report_dir else results_dir / "report" / "aggregate.html"
    if report_path.exists():
        webbrowser.open(report_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
