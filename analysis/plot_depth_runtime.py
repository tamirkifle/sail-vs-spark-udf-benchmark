"""W0: runtime vs pipeline depth — line plot per execution config.

The slope of each line is the *per-stage IPC cost*. Config A should be steep,
Config B moderate, and Configs C/D close to flat.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "depth_runtime.png"

    # { execution -> [(depth, wall_clock_sec), ...] }
    data: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for man in sorted(rdir.glob("*_manifest.json")):
        m = json.loads(man.read_text())
        if m.get("workload") != "w0":
            continue
        exe = m.get("execution")
        d = m.get("depth")
        w = m.get("wall_clock_sec")
        if exe is None or d is None or w is None:
            continue
        data[exe].append((int(d), float(w)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for exe in sorted(data):
        pts = sorted(data[exe])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=f"Config {exe}", linewidth=1.8)
    ax.set_xlabel("Pipeline depth")
    ax.set_ylabel("Wall clock (seconds)")
    ax.set_title("W0 — runtime vs chained-UDF depth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot_depth_runtime] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
