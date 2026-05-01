"""CLI entry point for running benchmark cells."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from sail_vs_spark.execution.registry import SUPPORTED_EXECUTIONS
from sail_vs_spark.runner.core import execute_run
from sail_vs_spark.workloads.registry import REGISTRY


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

def _make_session(execution: str, cfg: dict) -> Any:
    if execution in ("A", "B"):
        from sail_vs_spark.engines.spark_session import build_spark_session
        return build_spark_session(cfg)
    if execution in ("C", "D"):
        from sail_vs_spark.engines.sail_session import build_sail_session
        return build_sail_session(cfg)
    raise ValueError(f"unknown execution {execution!r}")


def run_one(*args, results_dir: Path, run_id: str | None = None) -> dict:
    """Run one cell.

    Supported call shapes:
      ``run_one(spark, cfg, workload, execution, ...)``
      ``run_one(cfg, workload, execution, ...)``  # session created internally
    """
    if len(args) == 4:
        spark, cfg, workload, execution = args
        owns_session = False
    elif len(args) == 3 and isinstance(args[0], dict):
        cfg, workload, execution = args
        spark = _make_session(execution, cfg)
        owns_session = True
    else:
        raise TypeError("run_one expects (spark, cfg, workload, execution) or (cfg, workload, execution)")

    try:
        return execute_run(
            spark,
            cfg,
            workload,
            execution,
            results_dir=results_dir,
            run_id=run_id,
        )
    finally:
        if owns_session:
            try:
                spark.stop()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run one (or more) Sail-vs-Spark benchmark cells."
    )
    p.add_argument("--config", required=True,
                   help="Path to a YAML config (config/laptop.yaml or config/gpu_h200.yaml)")
    p.add_argument("--workload", required=True, choices=sorted(REGISTRY))
    p.add_argument("--execution", required=True, choices=list(SUPPORTED_EXECUTIONS))
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
    try:
        n_samples = max(1, args.samples)
        base_id = args.run_id or f"{args.workload}_{args.execution}"
        for i in range(1, n_samples + 1):
            rid = f"{base_id}_s{i}"
            run_one(
                spark,
                cfg,
                args.workload,
                args.execution,
                results_dir=results_dir,
                run_id=rid,
            )
    finally:
        try:
            spark.stop()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
