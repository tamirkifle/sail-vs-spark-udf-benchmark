"""Serialization vs compute pie chart — one pie per (workload, execution).

Reads ``*_boundary.json`` files (when present) and sums:
    serialization_sec = DATA_TRANSFER_IN + DATA_TRANSFER_OUT
    compute_sec       = everything else recorded
    idle_sec          = wall - (serialization + compute)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


_COMPUTE_PHASES = (
    "INFERENCE", "SCORE", "EMBED", "SIMILARITY",
    "TOKENIZE", "DETOKENIZE", "TRIVIAL_COMPUTE",
)
_SERIAL_PHASES = ("DATA_TRANSFER_IN", "DATA_TRANSFER_OUT")


def _split(boundary: dict) -> tuple[float, float, float]:
    phases = boundary.get("phases", {})
    wall = float(boundary.get("total_wall_sec", 0))
    serial = sum(float(phases.get(p, {}).get("total_sec", 0))
                 for p in _SERIAL_PHASES)
    compute = sum(float(phases.get(p, {}).get("total_sec", 0))
                  for p in _COMPUTE_PHASES)
    idle = max(0.0, wall - (serial + compute))
    return serial, compute, idle



def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "serialization_pies.png"

    files = sorted(rdir.glob("*_boundary.json"))
    if not files:
        print(f"[plot_serialization] no *_boundary.json under {rdir} — "
              "nothing to plot. (These are written by instrumented runs; "
              "a minimal benchmark still produces stats/manifest only.)")
        # Write an empty figure with a helpful hint
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No boundary JSON yet.", ha="center", va="center")
        fig.savefig(out, dpi=120)
        return 0

    cols = min(4, len(files))
    rows = math.ceil(len(files) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = [axes] if not hasattr(axes, "flat") else list(axes.flat)

    for ax, f in zip(axes, files):
        b = json.loads(f.read_text())
        serial, compute, idle = _split(b)
        vals = [serial, compute, idle]
        if sum(vals) == 0:
            ax.axis("off"); continue
        ax.pie(vals, labels=["serialization", "compute", "idle"],
               colors=["#d62728", "#2ca02c", "#7f7f7f"],
               autopct=lambda v: f"{v:.0f}%")
        ax.set_title(b.get("config") or f.stem.replace("_boundary", ""),
                     fontsize=9)
    for ax in axes[len(files):]:
        ax.axis("off")

    fig.suptitle("Serialization vs compute — per (workload, execution)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot_serialization] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
