"""Aggregate per-run JSON into a single comparison dataframe + markdown table.

Reads every ``*_stats.json`` and ``*_manifest.json`` under ``--results_dir``
and produces:

  * ``aggregate.csv``             — one row per run
  * ``aggregate.md``              — human-readable table grouped by workload
  * ``aggregate.json``            — machine-readable same content
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "run_id", "workload", "execution", "depth",
    "wall_clock_sec", "output_rows",
    "avg_cpu_pct", "peak_rss_mb", "peak_host_ram_gb",
    "avg_gpu_util_pct", "peak_gpu_util_pct",
    "avg_gpu_power_w", "peak_gpu_mem_used_mb",
    "pipeline_continuity",
    "bytes_written_delta", "mb_written_delta", "write_throughput_mb_s",
]


def _load_stats(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        data = json.load(fh)
    row = {k: data.get(k) for k in FIELDS}
    row["run_id"] = data.get("config") or path.stem.replace("_stats", "")
    return row



def aggregate(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stats in sorted(results_dir.glob("*_stats.json")):
        row = _load_stats(stats)
        # Try to enrich with manifest (workload/execution/depth). We use
        # explicit None checks rather than setdefault because the stats
        # file populates these keys with None when not present, and
        # setdefault only fires on *missing* keys, not None values.
        manifest = stats.with_name(
            stats.stem.replace("_stats", "_manifest") + ".json"
        )
        if manifest.exists():
            with open(manifest) as fh:
                m = json.load(fh)
            for k in ("workload", "execution", "depth", "output_rows"):
                if row.get(k) is None:
                    row[k] = m.get(k)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    # Sort so comparison within each workload is easy
    rows = sorted(rows, key=lambda r: (
        str(r.get("workload")), str(r.get("depth") or 0), str(r.get("execution"))
    ))
    lines = ["# Sail vs Spark — benchmark results\n"]
    current = None
    for r in rows:
        wl = r.get("workload") or "?"
        if wl != current:
            current = wl
            lines.append(f"\n## Workload {wl}\n")
            lines.append(
                "| Cfg | Depth | Wall (s) | Rows | GPU util% | "
                "GPU pwr (W) | Peak RSS (MB) | MB written | Continuity |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|")
        lines.append(
            f"| {r.get('execution','?')} | {r.get('depth','')} | "
            f"{r.get('wall_clock_sec',0)} | {r.get('output_rows','')} | "
            f"{r.get('avg_gpu_util_pct',0)} | {r.get('avg_gpu_power_w',0)} | "
            f"{r.get('peak_rss_mb',0)} | {r.get('mb_written_delta',0)} | "
            f"{r.get('pipeline_continuity',0)} |"
        )
    path.write_text("\n".join(lines))



def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()

    rdir = Path(args.results_dir)
    rows = aggregate(rdir)
    if not rows:
        print(f"[aggregate] no *_stats.json found under {rdir}")
        return 1

    write_csv(rows, rdir / "aggregate.csv")
    write_markdown(rows, rdir / "aggregate.md")
    with open(rdir / "aggregate.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    print(f"[aggregate] {len(rows)} runs → "
          f"{rdir/'aggregate.csv'}, {rdir/'aggregate.md'}, {rdir/'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
