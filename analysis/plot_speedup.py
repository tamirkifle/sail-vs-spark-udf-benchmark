"""Speedup Bar Chart.

Plots the relative speedup of configs B, C, D against baseline A.
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
    out = Path(args.out) if args.out else rdir / "relative_speedups.png"

    # Collect data
    data = []
    for manifest_path in rdir.glob("*_manifest.json"):
        stats_path = manifest_path.with_name(manifest_path.name.replace("_manifest.json", "_stats.json"))
        if not stats_path.exists():
            continue
            
        with open(manifest_path) as f:
            m = json.load(f)
        with open(stats_path) as f:
            s = json.load(f)
            
        data.append({
            "Workload": m["workload"].upper(),
            "Config": m["execution"],
            "WallTime": s["wall_clock_sec"]
        })
        
    df = pd.DataFrame(data)
    if df.empty:
        print("[plot_speedup] No data found.")
        return

    # Average by Workload and Config (Warm average roughly, simple mean here for plot)
    agg = df.groupby(["Workload", "Config"]).mean().reset_index()

    # Pivot to get speedups
    pivot = agg.pivot(index="Workload", columns="Config", values="WallTime")
    
    # Calculate speedup = Base (A) / Config
    speedups = pd.DataFrame(index=pivot.index)
    for c in ["B", "C", "D"]:
        if c in pivot.columns and "A" in pivot.columns:
            speedups[f"Config {c}"] = pivot["A"] / pivot[c]
            
    if speedups.empty:
        print("[plot_speedup] Insufficient data for speedup calculation.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    speedups.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white", zorder=3)
    
    # Draw baseline at 1.0x (Config A)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Config A Baseline (1.0x)", zorder=2)
    
    ax.set_title("Relative Engine Speedup vs Spark (Row/Pickle)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Speedup Multiplier (Higher is Better)")
    ax.set_xlabel("Workload")
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    
    # Add data labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1fx", padding=3, fontsize=9)
        
    ax.legend(title="Configuration")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot_speedup] Wrote {out}")

if __name__ == "__main__":
    main()
