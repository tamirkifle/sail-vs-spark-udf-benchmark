"""Speedup bar chart with readable scaling for both slowdowns and large wins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "relative_speedups.png"

    rows = []
    for manifest_path in rdir.glob("*_manifest.json"):
        stats_path = manifest_path.with_name(
            manifest_path.name.replace("_manifest.json", "_stats.json")
        )
        if not stats_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(stats_path) as f:
            stats = json.load(f)

        rows.append(
            {
                "Workload": str(manifest["workload"]).upper(),
                "Config": manifest["execution"],
                "WallTime": float(stats["wall_clock_sec"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("[plot_speedup] No data found.")
        return 0

    agg = df.groupby(["Workload", "Config"]).mean().reset_index()
    pivot = agg.pivot(index="Workload", columns="Config", values="WallTime")

    speedups = pd.DataFrame(index=pivot.index)
    for config in ["B", "C", "D"]:
        if config in pivot.columns and "A" in pivot.columns:
            speedups[f"Config {config}"] = pivot["A"] / pivot[config]

    if speedups.empty:
        print("[plot_speedup] Insufficient data for speedup calculation.")
        return 0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    speedups.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="white", zorder=3)

    ax.set_yscale("log")
    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Config A Baseline (1.0x)",
        zorder=2,
    )
    ax.set_title("Relative Engine Speedup vs Spark (Row/Pickle)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Speedup Multiplier (Higher is Better, log scale)")
    ax.set_xlabel("Workload")
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    ymin = min(float(speedups.min().min()), 0.8)
    ymax = max(float(speedups.max().max()), 1.2)
    ax.set_ylim(max(0.5, ymin * 0.85), ymax * 1.25)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1fx", padding=3, fontsize=9)

    ax.legend(title="Configuration")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot_speedup] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
