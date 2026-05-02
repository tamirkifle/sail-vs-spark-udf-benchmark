"""Benchmark run orchestration."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sail_vs_spark.execution.registry import run_workload


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    output_parquet: str
    stats_json: str
    manifest_json: str
    trace_json: str
    nvidia_dmon_log: str

    @classmethod
    def for_run(cls, results_dir: Path, run_id: str) -> "RunArtifacts":
        run_dir = results_dir / "runs" / run_id
        return cls(
            run_dir=run_dir,
            output_parquet=str(run_dir / "output.parquet"),
            stats_json=str(run_dir / "stats.json"),
            manifest_json=str(run_dir / "manifest.json"),
            trace_json=str(run_dir / "trace.json"),
            nvidia_dmon_log=str(run_dir / "nvidia_dmon.log"),
        )


def resolve_run_id(workload: str, execution: str, run_id: str | None) -> str:
    return run_id or f"{workload}_{execution}_{uuid.uuid4().hex[:6]}"


def resolve_prompts_parquet(cfg: dict[str, Any]) -> str:
    parquet_path = str(Path(cfg["dataset"]["out_dir"]) / "prompts.parquet")
    if not Path(parquet_path).exists():
        raise FileNotFoundError(
            f"prompts parquet not found: {parquet_path}. Run scripts/prep_dataset.py first."
        )
    return parquet_path


def workload_depth(cfg: dict[str, Any], workload: str) -> int | None:
    if workload != "w0":
        return None
    return int(cfg.get("workloads", {}).get("w0_chained", {}).get("depth", 1))


def execute_run(
    spark: Any,
    cfg: dict[str, Any],
    workload: str,
    execution: str,
    *,
    results_dir: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    from sail_vs_spark.profiling.metrics_collector import MetricsCollector
    from sail_vs_spark.runner.manifest import get_setup_description, make_manifest, save_manifest
    from sail_vs_spark.runner.traces import clear_trace_dir, collect_trace_events, save_trace_artifact

    workload = workload.lower()
    execution = execution.upper()
    run_id = resolve_run_id(workload, execution, run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    clear_trace_dir()

    parquet_path = resolve_prompts_parquet(cfg)
    artifacts = RunArtifacts.for_run(results_dir, run_id)
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    depth = workload_depth(cfg, workload)

    n_rows = None
    collector = MetricsCollector(
        run_id,
        sample_interval_sec=cfg.get("runner", {}).get("sample_interval_sec", 0.5),
        nvidia_dmon_path=artifacts.nvidia_dmon_log,
    )
    collector.start()
    t0 = time.perf_counter()
    try:
        n_rows = run_workload(
            spark=spark,
            execution=execution,
            workload=workload,
            parquet_path=parquet_path,
            cfg=cfg,
            output_parquet=artifacts.output_parquet,
        )
    finally:
        wall = time.perf_counter() - t0
        collector.stop()
        collector.save(
            artifacts.stats_json,
            extra={
                "workload": workload,
                "execution": execution,
                "setup_description": get_setup_description(execution),
                "depth": depth,
                "wall_clock_sec": round(wall, 3),
                "output_rows": int(n_rows) if isinstance(n_rows, int) else None,
            },
        )

    trace_path = save_trace_artifact(collect_trace_events(), artifacts.trace_json)
    manifest = make_manifest(
        run_id=run_id,
        workload_code=workload,
        execution_config=execution,
        depth=depth,
        cfg=cfg,
        output_parquet=artifacts.output_parquet if Path(artifacts.output_parquet).exists() else None,
        wall_clock_sec=round(wall, 3),
        output_rows=int(n_rows) if isinstance(n_rows, int) else None,
        boundary_json=None,
        stats_json=artifacts.stats_json,
        trace_json=trace_path,
    )
    save_manifest(manifest, artifacts.manifest_json)

    if trace_path is not None:
        print(f"[cli] saved trace to {trace_path}")
    print(f"[cli] done - {run_id}  wall={wall:.2f}s  rows={n_rows}")
    return manifest
