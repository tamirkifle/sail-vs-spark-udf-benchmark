"""CLI entry point for running one (workload, execution) combination.

Usage
─────
    python -m sail_vs_spark.runner.cli \
        --config config/laptop.yaml --workload w0 --execution A --depth 1

The CLI:
  1. Loads + merges the YAML config (and any command-line overrides).
  2. Instantiates the right SparkSession / Sail session.
  3. Starts the MetricsCollector in the background.
  4. Dispatches to the right ``run_wX`` function on the right Config module.
  5. Stops the collector + saves boundary timer, stats, manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml


def _load_cfg(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    # Depth override for W0
    if args.depth is not None:
        cfg.setdefault("workloads", {}).setdefault("w0_chained", {})
        cfg["workloads"]["w0_chained"]["depth"] = int(args.depth)
    # Device override
    if args.device:
        cfg.setdefault("hardware", {})["device"] = args.device
    # Dataset rows override
    if args.n_rows is not None:
        cfg.setdefault("dataset", {})["n_rows"] = int(args.n_rows)
        
    # Auto-resolve num_partitions to available CPU cores if set to "auto"
    hw = cfg.setdefault("hardware", {})
    if str(hw.get("num_partitions")).lower() == "auto":
        import os
        try:
            # Respects cpuset on Linux (like SLURM task allocations)
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = os.cpu_count() or 8
        hw["num_partitions"] = cores
        
    return cfg



# Map (execution, workload) -> concrete run function (lazy-imported to avoid
# dragging pyspark into sessions that don't need it).
def _dispatch(execution: str, workload: str):
    execution = execution.upper()
    workload = workload.lower()
    if execution == "A":
        from sail_vs_spark.configs import config_a_spark_row as m
    elif execution == "B":
        from sail_vs_spark.configs import config_b_spark_pandas as m
    elif execution == "C":
        from sail_vs_spark.configs import config_c_sail_arrow as m
    elif execution == "D":
        from sail_vs_spark.configs import config_d_sail_udtf as m
    else:
        raise ValueError(f"unknown execution {execution!r}")
    fn_name = f"run_{workload}"
    if not hasattr(m, fn_name):
        raise ValueError(
            f"workload {workload!r} not implemented for config {execution}"
        )
    return getattr(m, fn_name)


def _make_session(execution: str, cfg: dict) -> Any:
    if execution in ("A", "B"):
        from sail_vs_spark.engines.spark_session import build_spark_session
        return build_spark_session(cfg)
    if execution in ("C", "D"):
        from sail_vs_spark.engines.sail_session import build_sail_session
        return build_sail_session(cfg)
    raise ValueError(f"unknown execution {execution!r}")


def run_one(
    spark: Any,
    cfg: dict,
    workload: str,
    execution: str,
    *,
    results_dir: Path,
    run_id: str | None = None,
) -> dict:
    """Run one (workload, execution) cell. Returns the run's manifest dict."""
    from sail_vs_spark.profiling.metrics_collector import MetricsCollector
    from sail_vs_spark.runner.manifest import make_manifest, save_manifest

    run_id = run_id or f"{workload}_{execution}_{uuid.uuid4().hex[:6]}"
    results_dir.mkdir(parents=True, exist_ok=True)

    import glob, os
    os.makedirs("/tmp/sail_traces", exist_ok=True)
    for f in glob.glob("/tmp/sail_traces/*.jsonl"):
        try: os.remove(f)
        except Exception: pass

    # Resolve the prompts parquet path
    parquet_path = cfg["dataset"]["out_dir"] + "/prompts.parquet"
    if not Path(parquet_path).exists():
        raise FileNotFoundError(
            f"prompts parquet not found: {parquet_path}. Run scripts/prep_dataset.py first."
        )

    output_parquet = str(results_dir / f"{run_id}_output.parquet")
    stats_json = str(results_dir / f"{run_id}_stats.json")
    manifest_json = str(results_dir / f"{run_id}_manifest.json")

    run_fn = _dispatch(execution, workload)

    n_rows = None  # Initialize for finally block
    col = MetricsCollector(
        run_id, sample_interval_sec=cfg["runner"].get("sample_interval_sec", 0.5)
    )
    col.start()
    t0 = time.perf_counter()
    try:
        if workload == "w0":
            depth = int(cfg["workloads"]["w0_chained"].get("depth", 1))
            n_rows = run_fn(spark, parquet_path, depth,
                            output_parquet if execution in ("A", "B", "C", "D") else None)
        else:
            n_rows = run_fn(spark, parquet_path, cfg, output_parquet)
    finally:
        wall = time.perf_counter() - t0
        col.stop()
        from sail_vs_spark.runner.manifest import get_setup_description
        col.save(stats_json, extra={
            "workload": workload, "execution": execution,
            "setup_description": get_setup_description(execution),
            "depth": int(cfg["workloads"]["w0_chained"].get("depth", 1))
                     if workload == "w0" else None,
            "wall_clock_sec": round(wall, 3),
            "output_rows": int(n_rows) if isinstance(n_rows, int) else None,
        })


    manifest = make_manifest(
        run_id=run_id, workload_code=workload, execution_config=execution,
        depth=int(cfg["workloads"]["w0_chained"].get("depth", 1))
               if workload == "w0" else None,
        cfg=cfg,
        output_parquet=output_parquet if Path(output_parquet).exists()
                       else None,
        wall_clock_sec=round(wall, 3),
        output_rows=int(n_rows) if isinstance(n_rows, int) else None,
        boundary_json=None,
        stats_json=stats_json,
    )
    save_manifest(manifest, manifest_json)
    
    # Collect detailed traces if they exist
    import glob
    import os
    trace_events = []
    for f in glob.glob("/tmp/sail_traces/*.jsonl"):
        try:
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        trace_events.append(json.loads(line))
            os.remove(f) # Clean up
        except Exception:
            pass
    
    if trace_events:
        trace_out = str(results_dir / f"{run_id}_trace.json")
        with open(trace_out, "w") as fh:
            json.dump({"traceEvents": trace_events, "displayTimeUnit": "ms"}, fh, indent=2)
        print(f"[cli] saved trace with {len(trace_events)} events to {trace_out}")
        
    print(f"[cli] done — {run_id}  wall={wall:.2f}s  rows={n_rows}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run one (or more) Sail-vs-Spark benchmark cells."
    )
    p.add_argument("--config", required=True,
                   help="Path to a YAML config (config/laptop.yaml or config/gpu_h200.yaml)")
    p.add_argument("--workload", required=True, choices=["w0", "w1", "w2", "w3", "w4"])
    p.add_argument("--execution", required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--depth", type=int, default=None,
                   help="W0 pipeline depth override (1..3 typical).")
    p.add_argument("--device", default=None,
                   help="Override hardware.device (cpu|mps|cuda|auto).")
    p.add_argument("--n-rows", type=int, default=None,
                   help="Override dataset.n_rows (useful for smoke tests).")
    p.add_argument("--results-dir", default=None,
                   help="Override runner.results_dir.")
    p.add_argument("--run-id", default=None,
                   help="Base run ID. With --samples N, IDs are <run-id>_s1 … _sN.")
    p.add_argument("--samples", type=int, default=1,
                   help="Number of back-to-back samples sharing the same session "
                        "(s1=cold/setup, s2+= warm/steady). Default: 1.")

    args = p.parse_args(argv)
    cfg = _load_cfg(args.config)
    cfg = _apply_overrides(cfg, args)

    results_dir = Path(
        args.results_dir or cfg.get("runner", {}).get("results_dir", "results/")
    )

    # Create the session once and share it across all samples so that
    # Spark Python workers (worker.reuse=true) keep model state warm for s2+.
    spark = _make_session(args.execution, cfg)

    n_samples = max(1, args.samples)
    base_id = args.run_id or f"{args.workload}_{args.execution}"
    for i in range(1, n_samples + 1):
        rid = f"{base_id}_s{i}"
        run_one(spark, cfg, args.workload, args.execution,
                results_dir=results_dir, run_id=rid)

    return 0


if __name__ == "__main__":
    sys.exit(main())
