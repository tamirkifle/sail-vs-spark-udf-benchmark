"""Consolidated trace-accounted time breakdown chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_COMPUTE_PHASES = (
    "INFERENCE",
    "SCORE",
    "EMBED",
    "SIMILARITY",
    "TOKENIZE",
    "DETOKENIZE",
    "TRIVIAL_COMPUTE",
    "UDF_BATCH_EXECUTION",
    "UDF_ROW_EXECUTION",
)
_SERIAL_PHASES = ("DATA_TRANSFER_IN", "DATA_TRANSFER_OUT")
_CONFIG_ORDER = ["A", "B", "C", "D"]
_CONFIG_LABELS = {
    "A": "Spark Row",
    "B": "Spark Pandas",
    "C": "Sail Arrow",
    "D": "Sail UDTF",
}


def _split_trace(trace: dict, wall: float) -> tuple[float, float, float]:
    serial = 0.0
    compute = 0.0
    for event in trace.get("traceEvents", []):
        phase = event.get("name")
        dur = float(event.get("dur", 0.0)) / 1_000_000.0
        if phase in _SERIAL_PHASES:
            serial += dur
        elif phase in _COMPUTE_PHASES:
            compute += dur
    untimed = max(0.0, wall - (serial + compute))
    return serial, compute, untimed


def _resolve_trace_path(manifest_path: Path, manifest: dict) -> Path:
    raw_trace = manifest.get("trace_json")
    if raw_trace:
        trace_path = Path(raw_trace)
        if not trace_path.is_absolute():
            trace_path = trace_path if trace_path.exists() else manifest_path.parent / trace_path.name
        return trace_path
    return manifest_path.with_name(manifest_path.name.replace("_manifest.json", "_trace.json"))


def _manifest_paths(rdir: Path) -> list[Path]:
    paths = set(rdir.glob("*_manifest.json"))
    paths.update(rdir.glob("runs/*/manifest.json"))
    return sorted(paths)


def _collect_rows(rdir: Path) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}

    for manifest_path in _manifest_paths(rdir):
        manifest = json.loads(manifest_path.read_text())
        trace_path = _resolve_trace_path(manifest_path, manifest)
        if not trace_path.exists():
            continue
        wall = float(manifest.get("wall_clock_sec", 0.0) or 0.0)
        trace = json.loads(trace_path.read_text())
        split = _split_trace(trace, wall)
        key = (str(manifest.get("workload", "")).upper(), str(manifest.get("execution", "")))
        grouped.setdefault(key, []).append((split[0], split[1], split[2], wall))

    rows: list[dict[str, float | str]] = []
    for (workload, config), samples in grouped.items():
        serial = sum(s[0] for s in samples) / len(samples)
        compute = sum(s[1] for s in samples) / len(samples)
        untimed = sum(s[2] for s in samples) / len(samples)
        wall = sum(s[3] for s in samples) / len(samples)
        total = max(wall, serial + compute + untimed, 1e-12)
        rows.append(
            {
                "Workload": workload,
                "Config": config,
                "Label": f"{workload}  {_CONFIG_LABELS.get(config, config)}",
                "Wall": wall,
                "SerialPct": serial / total * 100.0,
                "ComputePct": compute / total * 100.0,
                "UntimedPct": untimed / total * 100.0,
            }
        )

    workload_order = {"W0": 0, "W1": 1, "W2": 2, "W3": 3, "W4": 4}
    rows.sort(key=lambda row: (workload_order.get(str(row["Workload"]), 99), _CONFIG_ORDER.index(str(row["Config"]))))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rdir = Path(args.results_dir)
    out = Path(args.out) if args.out else rdir / "serialization_pies.png"

    rows = _collect_rows(rdir)
    if not rows:
        print(f"[plot_serialization] no trace timing data under {rdir}")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No serialization timing data available.", ha="center", va="center")
        fig.savefig(out, dpi=120)
        return 0

    plot_rows: list[dict[str, float | str | bool]] = []
    prev_workload: str | None = None
    for row in rows:
        workload = str(row["Workload"])
        if prev_workload is not None and workload != prev_workload:
            plot_rows.append(
                {
                    "Label": "",
                    "Workload": "",
                    "Config": "",
                    "Wall": 0.0,
                    "SerialPct": 0.0,
                    "ComputePct": 0.0,
                    "UntimedPct": 0.0,
                    "Spacer": True,
                }
            )
        item = dict(row)
        item["Spacer"] = False
        plot_rows.append(item)
        prev_workload = workload

    labels = [str(row["Label"]) for row in plot_rows]
    serial = np.array([float(row["SerialPct"]) for row in plot_rows])
    compute = np.array([float(row["ComputePct"]) for row in plot_rows])
    untimed = np.array([float(row["UntimedPct"]) for row in plot_rows])
    walls = [float(row["Wall"]) for row in plot_rows]
    spacers = [bool(row["Spacer"]) for row in plot_rows]
    y = np.arange(len(plot_rows))

    fig_h = max(7.5, 0.42 * len(plot_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    bar_h = [0.22 if spacer else 0.78 for spacer in spacers]
    ax.barh(y, serial, height=bar_h, color="#d62728", label="serialization")
    ax.barh(y, compute, height=bar_h, left=serial, color="#2ca02c", label="traced compute")
    ax.barh(y, untimed, height=bar_h, left=serial + compute, color="#9ca3af", label="untimed/framework")

    for idx, row in enumerate(plot_rows):
        if spacers[idx]:
            ax.axhline(idx, color="#d1d5db", linewidth=1.0, alpha=0.8)
            continue
        if serial[idx] >= 8:
            ax.text(serial[idx] / 2, idx, f"{serial[idx]:.0f}%", ha="center", va="center", fontsize=8, color="white")
        if compute[idx] >= 8:
            ax.text(serial[idx] + compute[idx] / 2, idx, f"{compute[idx]:.0f}%", ha="center", va="center", fontsize=8, color="white")
        if untimed[idx] >= 10:
            ax.text(99.2, idx, f"{untimed[idx]:.0f}%", ha="right", va="center", fontsize=8, color="#111827")
        ax.text(101.5, idx, f"{walls[idx]:.2f}s", va="center", fontsize=8, color="#4b5563")

    group_starts: dict[str, int] = {}
    for idx, row in enumerate(plot_rows):
        workload = str(row["Workload"])
        if workload and workload not in group_starts:
            group_starts[workload] = idx

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("share of wall time (%)")
    ax.set_title("Serialization vs traced compute vs untimed/framework", pad=34)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax.text(
        0.0,
        1.01,
        "W0 depths are consolidated by averaging depth 1/2/3 within each config. Right-edge labels show wall time.",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
    )
    for workload, idx in group_starts.items():
        ax.text(
            -0.16,
            idx,
            workload,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#111827",
    )

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[plot_serialization] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
