"""Peak RSS Comparison Bar Chart.

Reduces the 'too many data points' issue by aggregating peak memory
usage per configuration across all workloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def get_label(cfg):
    labels = {
        "A": "Spark (Row/Pickle)",
        "B": "Spark (Pandas/Arrow)",
        "C": "Sail (Zero-Copy)",
        "D": "Sail (SQL-Native)"
    }
    return labels.get(cfg, cfg)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "memory.png"

    rows = []
    for stats_path in rdir.glob("*_stats.json"):
        manifest_path = stats_path.parent / stats_path.name.replace("_stats.json", "_manifest.json")
        if not manifest_path.exists():
            continue
            
        with open(manifest_path) as f:
            m = json.load(f)
        with open(stats_path) as f:
            s = json.load(f)
            
        # Group by Workload and Config, pick peak RSS
        rows.append({
            "Workload": m["workload"].upper(),
            "Config": m["execution"],
            "Label": get_label(m["execution"]),
            "Peak RSS (MB)": s["peak_rss_mb"]
        })

    if not rows:
        print("No memory data found.")
        return 0

    df = pd.DataFrame(rows)
    # Average peak RSS across samples for each workload/config
    df_agg = df.groupby(["Workload", "Label", "Config"]).mean().reset_index()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot grouped bar chart: Workload on X, Peak RSS on Y, Hue is Config
    ax = sns.barplot(
        data=df_agg,
        x="Workload",
        y="Peak RSS (MB)",
        hue="Label",
        palette=["#9E9E9E", "#FF7043", "#42A5F5", "#26A69A"]
    )

    plt.title("Memory Footprint: Peak Process RSS per Configuration", fontsize=14, fontweight='bold')
    plt.ylabel("Peak RSS (MB)", fontweight='bold')
    plt.xlabel("Workload", fontweight='bold')
    plt.legend(title="Execution Path", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"[plot_memory] wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
