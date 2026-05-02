"""Prepare UltraFeedback → parquet with (prompt_id, prompt_text) columns.

Handles three scenarios:
  1. `datasets` available and HF hub reachable — download & slice.
  2. `datasets` available but offline — cache miss => synthetic fallback.
  3. `datasets` not installed — synthetic fallback that still has the exact
     required schema (prompt_id: int, prompt_text: string).

The synthetic fallback keeps unit tests and the W0 scaffolding exercise
working on any machine without network or HF credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from itertools import islice
from typing import Any


def _prompt_column(row_or_columns: Any) -> str:
    columns = row_or_columns.keys() if isinstance(row_or_columns, dict) else row_or_columns
    for col in ("prompt", "instruction", "question"):
        if col in columns:
            return col
    raise ValueError(
        "dataset missing expected column (prompt/instruction/question); "
        f"have {list(columns)}"
    )


def _load_from_hf(source: str, split: str, n_rows: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    # Streaming avoids materializing the full UltraFeedback dataset for small
    # laptop runs. The non-streaming builder needs >1 GiB before slicing.
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(islice(load_dataset(source, split=split, streaming=True), n_rows)):
        col = _prompt_column(row)
        rows.append({"prompt_id": i, "prompt_text": str(row[col])})

    if rows:
        return rows

    ds = load_dataset(source, split=split)
    col = _prompt_column(ds.column_names)
    take = min(n_rows, len(ds))
    return [{"prompt_id": i, "prompt_text": str(ds[i][col])} for i in range(take)]



def _synthetic_rows(n_rows: int) -> list[dict[str, Any]]:
    """Deterministic synthetic prompts — used when HF download is unavailable."""
    templates = [
        "Explain {} in simple terms.",
        "Write a short poem about {}.",
        "List three benefits of {}.",
        "Summarise the history of {}.",
        "What is the difference between {} and its opposite?",
        "Describe a typical day for a {}.",
        "Compare and contrast {} with something similar.",
        "Give a one-sentence definition of {}.",
        "Write a tweet introducing {} to a child.",
        "Translate 'Good morning, {}.' into three languages.",
    ]
    topics = [
        "photosynthesis", "the French Revolution", "gradient descent",
        "Arrow memory format", "the jazz age", "Fermat's last theorem",
        "dark energy", "the Roman aqueducts", "lock-free queues",
        "nitrogen fixation", "Bauhaus design", "the Manhattan Project",
    ]
    rows: list[dict[str, Any]] = []
    for i in range(n_rows):
        tpl = templates[i % len(templates)]
        topic = topics[(i // len(templates)) % len(topics)]
        rows.append({
            "prompt_id": i,
            "prompt_text": tpl.format(topic),
        })
    return rows



def prepare(
    out_dir: str | Path,
    *,
    source: str = "openbmb/UltraFeedback",
    split: str = "train",
    n_rows: int = 100,
    force_synthetic: bool = False,
) -> Path:
    """Build a parquet at ``out_dir/prompts.parquet`` with schema:
        prompt_id: int64, prompt_text: string.

    Returns the path to the written parquet.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "prompts.parquet"

    # Strictly purge the existing parquet folder/file.
    # Spark/Sail output is a directory. If we don't delete it, new files 
    # are added to the old ones, causing the row count to grow (e.g. 1400 rows).
    if parquet_path.exists():
        import shutil
        if parquet_path.is_dir():
            shutil.rmtree(parquet_path)
        else:
            parquet_path.unlink()

    rows: list[dict[str, Any]]
    used_source: str
    if force_synthetic or source == "synthetic":
        rows = _synthetic_rows(n_rows)
        used_source = "synthetic"
    else:
        try:
            rows = _load_from_hf(source, split, n_rows)
            used_source = source
        except Exception as e:
            cause = e.__cause__ or e.__context__
            detail = f"{e.__class__.__name__}: {e}"
            if cause is not None:
                detail += f" | cause={cause.__class__.__name__}: {cause}"
            print(f"[dataset] HF load failed ({detail}); falling back to synthetic data")
            rows = _synthetic_rows(n_rows)
            used_source = "synthetic"

    # Write parquet. We avoid a hard dependency on pyarrow at import time
    # by deferring the import.
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(
        rows, schema=pa.schema([
            ("prompt_id", pa.int64()),
            ("prompt_text", pa.string()),
        ]),
    )
    pq.write_table(table, parquet_path)

    meta = {
        "source": used_source,
        "n_rows": len(rows),
        "split": split,
        "parquet_path": str(parquet_path),
    }
    with open(out_dir / "prompts_meta.json", "w") as fh:
        import json
        json.dump(meta, fh, indent=2)

    print(f"[dataset] wrote {len(rows)} rows → {parquet_path} "
          f"(source={used_source})")
    return parquet_path
