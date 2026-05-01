"""Overhead breakdown chart with scaling that keeps tiny Sail bars visible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "overhead_breakdown.png"

    rows = []
    for manifest_path in rdir.glob("*_manifest.json"):
        stats_path = manifest_path.with_name(
            manifest_path.name.replace("_manifest.json", "_stats.json")
        )
        trace_path = manifest_path.with_name(
            manifest_path.name.replace("_manifest.json", "_trace.json")
        )
        if not stats_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(stats_path) as f:
            stats = json.load(f)

        udf_time_sec = 0.0
        if trace_path.exists():
            try:
                with open(trace_path) as f:
                    trace = json.load(f)
                for event in trace.get("traceEvents", []):
                    if event.get("name") in ("UDF_BATCH_EXECUTION", "UDF_ROW_EXECUTION"):
                        udf_time_sec += float(event.get("dur", 0.0)) / 1_000_000.0
            except Exception:
                pass

        wall = float(stats["wall_clock_sec"])
        rows.append(
            {
                "Workload": str(manifest["workload"]).upper(),
                "Config": manifest["execution"],
                "WallTime": wall,
                "UDFTime": udf_time_sec,
                "Overhead": max(0.0, wall - udf_time_sec),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("[plot_overhead] No data found.")
        return 0

    agg = df.groupby(["Workload", "Config"]).mean().reset_index()
    workloads = sorted(agg["Workload"].unique())
    configs = ["A", "B", "C", "D"]

    fig, axes = plt.subplots(1, len(workloads), figsize=(4 * len(workloads), 6), sharey=True)
    if len(workloads) == 1:
        axes = [axes]

    global_max = max(float(agg["WallTime"].max()), 0.1)

    for i, workload in enumerate(workloads):
        ax = axes[i]
        wl_data = agg[agg["Workload"] == workload]
        wl_data = wl_data.set_index("Config").reindex(configs).fillna(0).reset_index()

        x = np.arange(len(configs))
        udf_times = wl_data["UDFTime"].values
        overheads = wl_data["Overhead"].values
        totals = udf_times + overheads

        ax.bar(x, udf_times, label="Pure UDF Compute", color="#2ca02c", edgecolor="white")
        ax.bar(
            x,
            overheads,
            bottom=udf_times,
            label="Framework Overhead\n(IPC/Serialization)",
            color="#d62728",
            edgecolor="white",
        )

        ax.set_yscale("symlog", linthresh=0.02)
        ax.set_ylim(0, global_max * 1.35)
        ax.set_title(f"Workload: {workload}", fontsize=12, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for j, total in enumerate(totals):
            if total > 0:
                label_y = total * (1.08 if total >= 0.05 else 1.5)
                ax.text(j, label_y, f"{total:.2f}s", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("Average Wall Clock Time (seconds, symlog scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
        )
    fig.suptitle("Execution Overhead Breakdown: Sail vs Spark", fontsize=16, fontweight="bold", y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot_overhead_breakdown] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
