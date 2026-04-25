"""Cumulative disk-write bytes per run — the Spark-spills-to-disk evidence.

Reads the raw ``samples`` array from each ``*_stats.json`` and subtracts the
first sample's ``write_bytes`` so each run starts at zero; overlays all runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "disk_io.png"

    fig, ax = plt.subplots(figsize=(11, 5))
    for stats in sorted(rdir.glob("*_stats.json")):
        with open(stats) as fh:
            data = json.load(fh)
        samples = data.get("samples") or []
        if not samples:
            continue
        # Normalise: subtract first sample's write_bytes so each run starts at 0
        first_wb = samples[0].get("write_bytes", 0)
        ts = [s.get("t_sec", 0) for s in samples]
        ys = [max(0, s.get("write_bytes", 0) - first_wb) / 1e6 for s in samples]
        if all(y == 0 for y in ys):
            continue
        label = data.get("config") or stats.stem.replace("_stats", "")
        ax.plot(ts, ys, label=label, linewidth=1.2, alpha=0.85)

    ax.set_xlabel("wall-clock seconds")
    ax.set_ylabel("Cumulative bytes written (MB)")
    ax.set_title("Disk write activity — Spark spills vs Sail stream")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot_disk_io] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
