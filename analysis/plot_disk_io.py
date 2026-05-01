"""Disk IO chart with a fallback when per-sample write counters are unavailable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _path_size_bytes(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return sum(int(child.stat().st_size) for child in path.rglob("*") if child.is_file())


def _resolve_output_path(manifest_path: Path, manifest: dict) -> Path | None:
    raw = manifest.get("output_parquet")
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    if candidate.exists():
        return candidate
    alt = manifest_path.parent / candidate.name
    return alt if alt.exists() else None


def _unit_scale(max_mb: float) -> tuple[float, str, str]:
    if max_mb < 0.1:
        return 1000.0, "KB", "%.1f"
    return 1.0, "MB", "%.3f"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "disk_io.png"

    fig, ax = plt.subplots(figsize=(11, 5))
    line_mode = False
    fallback_rows = []

    for stats_path in sorted(rdir.glob("*_stats.json")):
        with open(stats_path) as fh:
            stats = json.load(fh)

        manifest_path = stats_path.with_name(stats_path.name.replace("_stats.json", "_manifest.json"))
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        samples = stats.get("samples") or []

        if samples and "write_bytes" in samples[0]:
            first_wb = samples[0].get("write_bytes", 0)
            ts = [s.get("t_sec", 0) for s in samples]
            ys = [max(0, s.get("write_bytes", 0) - first_wb) / 1e6 for s in samples]
            if any(y > 0 for y in ys):
                label = stats.get("config") or stats_path.stem.replace("_stats", "")
                ax.plot(ts, ys, label=label, linewidth=1.2, alpha=0.85)
                line_mode = True
                continue

        fallback_rows.append(
            {
                "Workload": str(manifest.get("workload", "")).upper(),
                "Config": manifest.get("execution", stats.get("config", "?")),
                "MBWritten": (
                    float(stats.get("mb_written_delta", 0.0) or 0.0)
                    or round(_path_size_bytes(_resolve_output_path(manifest_path, manifest)) / 1e6, 3)
                ),
            }
        )

    if line_mode:
        ymax = 0.0
        for line in ax.lines:
            ys = line.get_ydata()
            if len(ys):
                ymax = max(ymax, float(np.max(ys)))
        scale, unit, fmt = _unit_scale(ymax)
        if scale != 1.0:
            for line in ax.lines:
                line.set_ydata(np.asarray(line.get_ydata()) * scale)
        ax.set_xlabel("wall-clock seconds")
        ax.set_ylabel(f"Cumulative bytes written ({unit})")
        ax.set_title("Disk write activity — Spark spills vs Sail stream")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    else:
        df = pd.DataFrame(fallback_rows)
        if df.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No disk IO metrics found.", ha="center", va="center")
        else:
            pivot = (
                df.groupby(["Workload", "Config"], as_index=False)["MBWritten"]
                .mean()
                .pivot(index="Workload", columns="Config", values="MBWritten")
                .fillna(0.0)
                .reindex(columns=["A", "B", "C", "D"], fill_value=0.0)
            )
            max_mb = float(pivot.to_numpy().max()) if not pivot.empty else 0.0
            scale, unit, fmt = _unit_scale(max_mb)
            plot_values = pivot * scale
            x = np.arange(len(pivot.index))
            width = 0.2
            colors = {"A": "#9E9E9E", "B": "#FF7043", "C": "#42A5F5", "D": "#26A69A"}

            for idx, config in enumerate(pivot.columns):
                bars = ax.bar(
                    x + (idx - 1.5) * width,
                    plot_values[config].values,
                    width=width,
                    label=f"Config {config}",
                    color=colors.get(config, "#555"),
                )
                ax.bar_label(bars, fmt=fmt, padding=2, fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index.tolist())
            ax.set_ylabel(f"{unit} written over run")
            ax.set_title(f"Disk IO fallback — aggregate bytes written per run ({unit})")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot_disk_io] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
