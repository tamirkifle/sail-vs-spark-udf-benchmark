"""Plot GPU utilisation timeline for every run under --results_dir.

Overlays all runs (one line per run_id) so the Spark sawtooth and Sail flat
patterns are visible side-by-side.
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
    out = Path(args.out) if args.out else rdir / "gpu_timeline.png"

    fig, ax = plt.subplots(figsize=(11, 5))
    any_samples = False
    for stats in sorted(rdir.glob("*_stats.json")):
        with open(stats) as fh:
            data = json.load(fh)
        samples = data.get("samples") or []
        ts = [s.get("t_sec", 0) for s in samples]
        ys = [s.get("gpu_util_pct", 0) for s in samples]
        if not ts or all(y == 0 for y in ys):
            continue
        label = data.get("config") or stats.stem.replace("_stats", "")
        ax.plot(ts, ys, label=label, linewidth=1.2, alpha=0.85)
        any_samples = True

    ax.set_xlabel("wall-clock seconds (since run start)")
    ax.set_ylabel("GPU SM utilisation (%)")
    ax.set_title("GPU utilisation — Spark vs Sail  (sawtooth vs flatline)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    if any_samples:
        ax.legend(fontsize=7, ncol=2, loc="lower right")
    else:
        ax.text(0.5, 0.5, "no GPU samples recorded",
                transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot_gpu_timeline] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
