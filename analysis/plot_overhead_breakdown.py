"""Overhead Breakdown Stacked Bar Chart.

Plots the ratio of pure UDF Compute vs Framework Overhead for each config.
Reads data directly from the generated aggregate.md or the raw JSON files.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "overhead_breakdown.png"

    # Collect data from manifest, stats, and traces (similar to aggregate_results)
    data = []
    for manifest_path in rdir.glob("*_manifest.json"):
        stats_path = manifest_path.with_name(manifest_path.name.replace("_manifest.json", "_stats.json"))
        trace_path = manifest_path.with_name(manifest_path.name.replace("_manifest.json", "_trace.json"))
        
        if not stats_path.exists():
            continue
            
        with open(manifest_path) as f:
            m = json.load(f)
        with open(stats_path) as f:
            s = json.load(f)
            
        udf_time_sec = 0.0
        if trace_path.exists():
            try:
                with open(trace_path) as f:
                    t = json.load(f)
                    for ev in t.get("traceEvents", []):
                        if ev.get("name") in ("UDF_BATCH_EXECUTION", "UDF_ROW_EXECUTION"):
                            udf_time_sec += ev.get("dur", 0.0) / 1_000_000.0
            except Exception:
                pass
                
        wall = s["wall_clock_sec"]
        data.append({
            "Workload": m["workload"].upper(),
            "Config": m["execution"],
            "WallTime": wall,
            "UDFTime": udf_time_sec,
            "Overhead": max(0.0, wall - udf_time_sec)
        })
        
    df = pd.DataFrame(data)
    if df.empty:
        print("[plot_overhead] No data found.")
        return

    # Average by Workload and Config
    agg = df.groupby(["Workload", "Config"]).mean().reset_index()

    # Pivot for plotting
    workloads = sorted(agg["Workload"].unique())
    configs = ["A", "B", "C", "D"]
    
    # Set up plot
    fig, axes = plt.subplots(1, len(workloads), figsize=(4 * len(workloads), 6), sharey=True)
    if len(workloads) == 1:
        axes = [axes]
        
    for i, wl in enumerate(workloads):
        ax = axes[i]
        wl_data = agg[agg["Workload"] == wl]
        
        # Ensure ordering A, B, C, D
        wl_data = wl_data.set_index("Config").reindex(configs).fillna(0).reset_index()
        
        x = np.arange(len(configs))
        udf_times = wl_data["UDFTime"].values
        overheads = wl_data["Overhead"].values
        
        ax.bar(x, udf_times, label="Pure UDF Compute", color="#2ca02c", edgecolor="white")
        ax.bar(x, overheads, bottom=udf_times, label="Framework Overhead\n(IPC/Serialization)", color="#d62728", edgecolor="white")
        
        ax.set_title(f"Workload: {wl}", fontsize=12, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        
        # Add values on top of bars
        for j, (udf, ov) in enumerate(zip(udf_times, overheads)):
            total = udf + ov
            if total > 0:
                ax.text(j, total + (total * 0.02), f"{total:.2f}s", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("Average Wall Clock Time (seconds)")
    axes[0].legend(loc="upper left")
    
    fig.suptitle("Execution Overhead Breakdown: Sail vs Spark", fontsize=16, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot_overhead_breakdown] Wrote {out}")

if __name__ == "__main__":
    main()
