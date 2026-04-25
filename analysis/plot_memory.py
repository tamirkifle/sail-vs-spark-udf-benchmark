"""System memory (RSS) over time per run — one line per run_id."""

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
    out = Path(args.out) if args.out else rdir / "memory.png"

    fig, ax = plt.subplots(figsize=(11, 5))
    for stats in sorted(rdir.glob("*_stats.json")):
        with open(stats) as fh:
            data = json.load(fh)
        samples = data.get("samples") or []
        ts = [s.get("t_sec", 0) for s in samples]
        ys = [s.get("rss_mb", 0) for s in samples]
        if not ts:
            continue
        label = data.get("config") or stats.stem.replace("_stats", "")
        ax.plot(ts, ys, label=label, linewidth=1.2, alpha=0.85)

    ax.set_xlabel("wall-clock seconds")
    ax.set_ylabel("Process RSS (MB)")
    ax.set_title("Host memory usage over time")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot_memory] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
