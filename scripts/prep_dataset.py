"""Standalone script: download UltraFeedback (or synthesize) → parquet.

Usage
─────
    python scripts/prep_dataset.py --config config/laptop.yaml
    python scripts/prep_dataset.py --config config/gpu_h200.yaml --force-synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the `src/` package importable when running from the repo root
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import yaml  # noqa: E402

from sail_vs_spark.dataset.prep import prepare  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--force-synthetic", action="store_true",
                   help="Skip HF Hub and generate synthetic prompts.")
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    ds = cfg["dataset"]
    out = prepare(
        ds["out_dir"],
        source=ds.get("source", "openbmb/UltraFeedback"),
        split=ds.get("split", "train"),
        n_rows=int(ds.get("n_rows", 100)),
        force_synthetic=args.force_synthetic,
    )
    print(f"OK: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
