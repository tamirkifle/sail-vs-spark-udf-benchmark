"""Aggregate benchmark artifacts into tabular and HTML summaries."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
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
    "INFERENCE",
    "SCORE",
    "EMBED",
    "SIMILARITY",
    "TOKENIZE",
    "DETOKENIZE",
    "TRIVIAL_COMPUTE",
}

PLOT_SCRIPTS = [
    "analysis/plot_depth_runtime.py",
    "analysis/plot_gpu_timeline.py",
    "analysis/plot_disk_io.py",
    "analysis/plot_serialization.py",
]

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
    trace_event_count: int


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
        return TraceSummary(0.0, 0.0, 0.0, 0)

    try:
        trace_data = _read_json(trace_path)
    except Exception:
        return TraceSummary(0.0, 0.0, 0.0, 0)

    udf_time_sec = 0.0
    transfer_time_sec = 0.0
    compute_time_sec = 0.0
    trace_event_count = 0

    for event in trace_data.get("traceEvents", []):
        phase = event.get("name")
        if phase not in TRACE_PHASES_OF_INTEREST:
            continue
        dur_sec = float(event.get("dur", 0.0)) / 1_000_000.0
        trace_event_count += 1
        if phase in {"UDF_BATCH_EXECUTION", "UDF_ROW_EXECUTION"}:
            udf_time_sec += dur_sec
        elif phase in {"DATA_TRANSFER_IN", "DATA_TRANSFER_OUT"}:
            transfer_time_sec += dur_sec
        else:
            compute_time_sec += dur_sec

    return TraceSummary(
        udf_time_sec=round(udf_time_sec, 6),
        transfer_time_sec=round(transfer_time_sec, 6),
        compute_time_sec=round(compute_time_sec, 6),
        trace_event_count=trace_event_count,
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
    udf_share = (trace.udf_time_sec / wall_time * 100.0) if wall_time > 0 else 0.0
    transfer_share = (trace.transfer_time_sec / wall_time * 100.0) if wall_time > 0 else 0.0
    compute_share = (trace.compute_time_sec / wall_time * 100.0) if wall_time > 0 else 0.0
    bytes_read_delta = int(stats.get("bytes_read_delta", 0) or 0)
    bytes_written_delta = int(stats.get("bytes_written_delta", 0) or 0)
    if bytes_written_delta <= 0:
        bytes_written_delta = _path_size_bytes(output_parquet_path)
    mb_written_delta = float(stats.get("mb_written_delta", 0.0) or 0.0)
    if mb_written_delta <= 0.0 and bytes_written_delta > 0:
        mb_written_delta = round(bytes_written_delta / 1e6, 3)
    write_throughput_mb_s = float(stats.get("write_throughput_mb_s", 0.0) or 0.0)
    if write_throughput_mb_s <= 0.0 and wall_time > 0 and bytes_written_delta > 0:
        write_throughput_mb_s = round((bytes_written_delta / 1e6) / wall_time, 3)

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
        "PeakHostRam_GB": float(stats.get("peak_host_ram_gb", 0.0) or 0.0),
        "AvgHostRam_pct": _sample_metric(samples, "host_ram_pct", lambda vs: sum(vs) / len(vs)),
        "PeakHostRam_pct": _sample_metric(samples, "host_ram_pct", max),
        "AvgGPUUtil_pct": float(stats.get("avg_gpu_util_pct", 0.0) or 0.0),
        "PeakGPUUtil_pct": float(stats.get("peak_gpu_util_pct", 0.0) or 0.0),
        "AvgGPUMemUtil_pct": float(stats.get("avg_gpu_mem_util_pct", 0.0) or 0.0),
        "PeakGPUMemUsed_MB": float(stats.get("peak_gpu_mem_used_mb", 0.0) or 0.0),
        "AvgGPUPower_W": float(stats.get("avg_gpu_power_w", 0.0) or 0.0),
        "PipelineContinuity": float(stats.get("pipeline_continuity", 0.0) or 0.0),
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
        "UDFShare_pct": round(udf_share, 3),
        "TransferShare_pct": round(transfer_share, 3),
        "TraceComputeShare_pct": round(compute_share, 3),
    }


def _build_run_df(results_dir: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in sorted(results_dir.glob("*_manifest.json")):
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
        udf_mean = float(perf["UDFTime_sec"].mean())
        transfer_mean = float(perf["TransferTime_sec"].mean())
        trace_compute_mean = float(perf["TraceComputeTime_sec"].mean())
        overhead_pct = max(0.0, ((perf_time - udf_mean) / perf_time * 100.0)) if perf_time > 0 else 0.0
        throughput = (float(perf["Rows"].mean()) / perf_time) if perf_time > 0 else 0.0

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
                "Avg CPU (%)": round(float(group["AvgCPU_pct"].mean()), 2),
                "Peak GPU Util (%)": round(float(group["PeakGPUUtil_pct"].max()), 2),
                "Pipeline Continuity": round(float(group["PipelineContinuity"].mean()), 3),
                "Avg GPU Power (W)": round(float(group["AvgGPUPower_W"].mean()), 2),
                "Disk Write (MB)": round(float(group["DiskWrite_MB"].mean()), 3),
                "Write Throughput (MB/s)": round(float(group["WriteThroughput_MBps"].mean()), 3),
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


def _write_csvs(results_dir: Path, run_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    run_df.to_csv(results_dir / "aggregate_runs.csv", index=False)
    summary_df.to_csv(results_dir / "aggregate_summary.csv", index=False)
    with open(results_dir / "aggregate_summary.json", "w") as fh:
        json.dump(summary_df.to_dict(orient="records"), fh, indent=2)


def _write_markdown(results_dir: Path, summary_df: pd.DataFrame) -> None:
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

    (results_dir / "aggregate.md").write_text("\n".join(lines))


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
    manifest_paths = sorted(results_dir.glob("*_manifest.json"))
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
            best_cont = max(c["Pipeline Continuity"] for c in cfgs)
            min_disk  = min(c["Disk Write (MB)"]     for c in cfgs)
            for c in cfgs:
                c["BestGPU"]  = abs(c["Peak GPU Util (%)"]  - best_gpu)  < 0.05
                c["BestCont"] = abs(c["Pipeline Continuity"] - best_cont) < 1e-6
                c["BestDisk"] = abs(c["Disk Write (MB)"]    - min_disk)  < 1e-6
        cards.append({"workload": workload, "configs": cfgs})
    return cards, tel_maxes


def _build_overhead_breakdown_payload(run_df: pd.DataFrame) -> dict[str, Any]:
    configs = ["A", "B", "C", "D"]
    if run_df.empty:
        return {
            "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
            "workloads": [],
            "maxWallSec": 0.0,
        }

    rows = run_df[["Workload", "Config", "WallTime", "UDFTime_sec"]].copy()
    rows["Overhead_sec"] = rows.apply(
        lambda row: max(0.0, float(row["WallTime"]) - float(row["UDFTime_sec"])),
        axis=1,
    )
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
                    "wall_sec": round(float(row["WallTime"]), 6),
                    "udf_sec": round(float(row["UDFTime_sec"]), 6),
                    "overhead_sec": round(float(row["Overhead_sec"]), 6),
                }
            )
        workloads.append({"workload": workload, "bars": bars})

    max_wall = float(agg["WallTime"].max()) if not agg.empty else 0.0
    return {
        "configs": [{"code": cfg, "label": get_label(cfg)} for cfg in configs],
        "workloads": workloads,
        "maxWallSec": round(max_wall, 6),
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
  <h2>Overhead Breakdown</h2>
  <p class="section-note">This D3-based view keeps the original decomposition but packages it as an interactive HTML chart. Each panel shows mean wall time per execution config, stacked into traced UDF compute and residual framework overhead.</p>
  <div class="overhead-legend">
    <div class="overhead-legend-group">
      <span class="overhead-legend-title">Execution configs</span>
      <div class="overhead-config-grid">__LEGEND__</div>
    </div>
    <div class="overhead-legend-group">
      <span class="overhead-legend-title">Stacked components</span>
      <div class="overhead-component-grid">
        <span class="overhead-component-pill"><span class="overhead-swatch overhead-udf"></span>Traced UDF compute</span>
        <span class="overhead-component-pill"><span class="overhead-swatch overhead-overhead"></span>Framework overhead</span>
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
  .overhead-udf { background: #2f9e44; }
  .overhead-overhead { background: #e25a1c; }
  .overhead-total { background: #94a3b8; }
  .overhead-chart-shell { margin-top: 6px; }
  .overhead-chart-grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
  .overhead-panel { border: 1px solid #e2e8f0; border-radius: 16px; background: linear-gradient(180deg, #fff, #f8fafc); padding: 14px 14px 12px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04); }
  .overhead-panel h3 { margin: 0; font-size: 16px; color: #0f172a; }
  .overhead-panel .subtitle { margin: 3px 0 10px; font-size: 12px; color: #64748b; }
  .overhead-svg { width: 100%; height: auto; display: block; }
  .overhead-axis text { fill: #475569; font-size: 11px; }
  .overhead-axis path, .overhead-axis line { stroke: #cbd5e1; }
  .overhead-grid line { stroke: #e2e8f0; stroke-dasharray: 3 4; }
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
    udf: "#2f9e44",
    overhead: "#e25a1c",
    grid: "#e2e8f0",
    axis: "#94a3b8",
  };

  function formatSeconds(value) {
    if (!Number.isFinite(value)) {
      return "-";
    }
    return value >= 1 ? `${value.toFixed(2)}s` : `${value.toFixed(3)}s`;
  }

  function hideTooltip() {
    tooltip.style("opacity", 0);
  }

  function showTooltip(event, datum) {
    tooltip
      .style("opacity", 1)
      .html(`
        <strong>${datum.workload} · Config ${datum.config}</strong>
        <div class="metric"><span>UDF compute</span><span>${formatSeconds(datum.udf_sec)}</span></div>
        <div class="metric"><span>Framework overhead</span><span>${formatSeconds(datum.overhead_sec)}</span></div>
        <div class="metric"><span>Total wall time</span><span>${formatSeconds(datum.wall_sec)}</span></div>
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
      chartNode.innerHTML = '<div class="overhead-empty">No overhead timing data was found for this run.</div>';
      return;
    }

    const containerWidth = chartNode.getBoundingClientRect().width || chartNode.clientWidth || 960;
    const minPanelWidth = 300;
    const gap = 16;
    const cols = Math.max(1, Math.min(3, Math.floor((containerWidth + gap) / minPanelWidth)));
    chartNode.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;

    const panelWidth = (containerWidth - gap * (cols - 1)) / cols;
    const panelHeight = 286;
    const margin = { top: 26, right: 14, bottom: 38, left: 46 };
    const innerWidth = Math.max(180, panelWidth - margin.left - margin.right);
    const innerHeight = panelHeight - margin.top - margin.bottom;
    const maxWall = Math.max(3.0, (payload.maxWallSec || 0) * 1.25);
    const yScale = d3.scaleSymlog().constant(0.02).domain([0, maxWall]).nice().range([innerHeight, 0]);
    const xScale = d3.scaleBand().domain(payload.configs.map((d) => d.code)).range([0, innerWidth]).padding(0.24);
    const barWidth = xScale.bandwidth();

    const svgWidth = innerWidth + margin.left + margin.right;
    const svgHeight = panelHeight;

    payload.workloads.forEach((workload) => {
      const panel = chartNode.appendChild(document.createElement("div"));
      panel.className = "overhead-panel";
      panel.innerHTML = `<h3>${workload.workload}</h3><div class="subtitle">Mean wall time per execution config</div>`;

      const svg = d3.select(panel)
        .append("svg")
        .attr("class", "overhead-svg")
        .attr("viewBox", `0 0 ${svgWidth} ${svgHeight}`)
        .attr("role", "img")
        .attr("aria-label", `Overhead breakdown for workload ${workload.workload}`);

      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

      g.append("g")
        .attr("class", "overhead-grid")
        .call(d3.axisLeft(yScale).ticks(4).tickSize(-innerWidth).tickFormat(""))
        .selectAll("line")
        .attr("stroke", palette.grid);

      g.append("g")
        .attr("class", "overhead-axis")
        .call(d3.axisLeft(yScale).ticks(4).tickFormat((d) => formatSeconds(d)));

      g.append("g")
        .attr("class", "overhead-axis")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).tickSizeOuter(0));

      const barGroups = g.selectAll(".overhead-bar")
        .data(workload.bars)
        .join("g")
        .attr("class", "overhead-bar")
        .attr("transform", (d) => `translate(${xScale(d.config)},0)`);

      barGroups.append("rect")
        .attr("class", "overhead-hit")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", barWidth)
        .attr("height", innerHeight)
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
        .attr("y", (d) => yScale(d.udf_sec))
        .attr("width", barWidth)
        .attr("height", (d) => Math.max(0, innerHeight - yScale(d.udf_sec)))
        .attr("fill", palette.udf);

      barGroups.append("rect")
        .attr("class", "segment")
        .attr("x", 0)
        .attr("y", (d) => yScale(d.wall_sec))
        .attr("width", barWidth)
        .attr("height", (d) => Math.max(0, yScale(d.udf_sec) - yScale(d.wall_sec)))
        .attr("fill", palette.overhead);

      barGroups.append("text")
        .attr("x", barWidth / 2)
        .attr("y", (d) => Math.max(12, yScale(d.wall_sec) - 8))
        .attr("text-anchor", "middle")
        .attr("fill", palette.axis)
        .attr("font-size", 11)
        .attr("font-weight", 700)
        .text((d) => formatSeconds(d.wall_sec));
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


def _write_html(results_dir: Path, summary_df: pd.DataFrame, run_df: pd.DataFrame) -> None:
    rows = summary_df.copy()
    rows["DepthDisplay"] = rows["Depth"].apply(lambda x: int(x) if pd.notna(x) else "-")
    rows["LogoSVG"] = rows["Config"].map(_logo_svg)
    rows["RowClass"] = rows["Config"].map(_row_class)
    rows["SpeedupClass"] = rows.apply(lambda row: _speedup_class(row["Config"], row["Speedup_x"]), axis=1)
    tel_cards, tel_maxes = _build_tel_cards(rows)
    overhead_breakdown_section = _build_overhead_breakdown_section(
        _build_overhead_breakdown_payload(run_df)
    )
    memory_section = _build_memory_section(_build_memory_payload(run_df))
    speedup_section = _build_speedup_section(_build_speedup_payload(run_df))
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
    <p class="section-note">Aggregated telemetry from the runtime collector. Surfaces secondary cost signals — GPU pipeline continuity, peak memory pressure, and disk materialization volume — that are not captured in wall-clock timing alone. Bold values indicate the best result for that metric within each workload.</p>
    <div class="tel-grid">
      {% for card in tel_cards %}
      <div class="tel-card">
        <div class="tel-card-header">{{ card.workload }}</div>
        {% for cfg in card.configs %}
        {% set gpu_pct  = [(cfg["Peak GPU Util (%)"]  / tel_maxes["Peak GPU Util (%)"]  * 100)|round|int, 2]|max %}
        {% set cont_pct = [(cfg["Pipeline Continuity"] * 100)|round|int, 2]|max %}
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
              <span class="tel-val {{ 'winner' if cfg['BestCont'] else '' }}">{{ "%.3f"|format(cfg["Pipeline Continuity"]) }}</span>
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
              <span class="tel-lbl">Disk Write</span>
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
  {{ overhead_breakdown_section | safe }}
  {{ memory_section | safe }}
  {{ speedup_section | safe }}
  {% for plot in plots %}
  <div class="card">
    <h2>{{ plot.title }}</h2>
    <p class="section-note">{{ plot.blurb }}</p>
    <div class="img-container">
      <img src="{{ plot.filename }}" alt="{{ plot.title }}">
    </div>
  </div>
  {% endfor %}
</body>
</html>
        """
    )

    plots = [
        {
            "title": "GPU Timeline",
            "filename": "gpu_timeline.png",
            "blurb": "GPU utilization over time, per configuration. Gaps between compute bursts indicate scheduling or data-transfer stalls that limit accelerator efficiency.",
        },
        {
            "title": "Depth Runtime",
            "filename": "depth_runtime.png",
            "blurb": "Depth = number of sequential UDF pipeline stages, where each stage does a trivially cheap computation. W0 runtime as a function of chain depth. Slope quantifies the per-crossing overhead tax that accumulates with each additional engine–Python round trip.",
        },
        {
            "title": "Disk IO",
            "filename": "disk_io.png",
            "blurb": "Bytes written to disk per run, with automatic unit scaling. Values elevated relative to input size indicate intermediate materialization or unnecessary data duplication.",
        },
        {
            "title": "Serialization vs Compute",
            "filename": "serialization_pies.png",
            "blurb": "Time budget decomposition per workload and configuration: traced model compute, traced serialization and transfer, and residual framework time. Unattributed time is the difference between wall-clock and the sum of instrumented spans.",
        },
    ]

    (results_dir / "aggregate.html").write_text(
        template.render(
            rows=rows,
            plots=plots,
            tel_cards=tel_cards,
            tel_maxes=tel_maxes,
            overhead_breakdown_section=overhead_breakdown_section,
            memory_section=memory_section,
            speedup_section=speedup_section,
            **context,
        )
    )


def _run_plot_scripts(results_dir: Path) -> None:
    print("Generating plots...")
    mpl_dir = results_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["MPLBACKEND"] = "Agg"
    for script in PLOT_SCRIPTS:
        script_path = Path(script)
        if not script_path.exists():
            continue
        cmd = [sys.executable, str(script_path), "--results_dir", str(results_dir)]
        print(f"Running {script}...")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: failed to run {script}: {exc}")


def aggregate(results_dir: str) -> pd.DataFrame:
    path = Path(results_dir)
    run_df = _build_run_df(path)
    if run_df.empty:
        print("No results found.")
        return run_df

    summary_df = _build_summary_df(run_df)
    _write_csvs(path, run_df, summary_df)
    _write_markdown(path, summary_df)
    _run_plot_scripts(path)
    _write_html(path, summary_df, run_df)
    print(f"Wrote aggregate outputs under {path}")
    return summary_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    aggregate(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
