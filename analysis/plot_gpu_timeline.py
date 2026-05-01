"""Plot GPU utilisation timeline for representative runs.

Reduces clutter by only plotting Sample 2 (Warm) for each configuration
of a specific representative workload (e.g. W1 or W4).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
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
    p.add_argument("--workload", default="W1", help="Which workload to show in the timeline")
    args = p.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "gpu_timeline.png"

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    any_samples = False
    
    # Consistent palette
    colors = {"A": "#9E9E9E", "B": "#FF7043", "C": "#42A5F5", "D": "#26A69A"}

    # Filter for Sample 2 (Warm) of the selected workload
    pattern = f"{args.workload.lower()}_([A-D])_.*s2_stats.json"
    
    # Fallback if no s2: try s1
    files = sorted(rdir.glob("*_stats.json"))
    
    selected_files = []
    for f in files:
        match = re.search(pattern, f.name)
        if match:
            selected_files.append((f, match.group(1)))

    if not selected_files:
        # If no s2, try s1
        pattern_s1 = f"{args.workload.lower()}_([A-D])_.*s1_stats.json"
        for f in files:
            match = re.search(pattern_s1, f.name)
            if match:
                selected_files.append((f, match.group(1)))

    for stats, config_key in selected_files:
        with open(stats) as fh:
            data = json.load(fh)
        samples = data.get("samples") or []
        ts = [s.get("t_sec", 0) for s in samples]
        ys = [s.get("gpu_util_pct", 0) for s in samples]
        
        # Multiply by 100 if stored as fraction
        if ys and max(ys) <= 1.0:
            ys = [y * 100 for y in ys]

        if not ts:
            continue
            
        label = get_label(config_key)
        ax.plot(ts, ys, label=label, linewidth=2, color=colors.get(config_key, "#333"), alpha=0.9)
        any_samples = True

    ax.set_xlabel("Wall-clock seconds (since run start)", fontweight='bold')
    ax.set_ylabel("GPU SM Utilisation (%)", fontweight='bold')
    ax.set_title(f"GPU Utilization Timeline: {args.workload} (Sawtooth vs. Flatline)", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    if any_samples:
        ax.legend(title="Execution Path", fontsize=9, loc="upper right")
    else:
        ax.text(0.5, 0.5, f"No GPU samples recorded for workload {args.workload}",
                transform=ax.transAxes, ha="center", va="center")
    
    sns.despine()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot_gpu_timeline] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
