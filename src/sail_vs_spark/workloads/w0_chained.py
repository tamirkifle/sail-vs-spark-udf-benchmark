"""W0 — Chained trivial UDF benchmark.

Pattern
───────
    input:int  → UDF1(x+1) → UDF2(x+1) → UDF3(x+1) → output:int

Variants: depth ∈ {1, 2, 3}.

This is the foundation workload. It quantifies the pure *overhead* of the
Spark/Sail UDF mechanism — no model loading, no GPU, just ``x + 1``. The
Spark-side costs (pickle, socket, Arrow IPC) show up as a near-constant
per-stage tax, so runtime grows ~linearly with depth for Spark and stays
flat for Sail.

Note on schema: W0 operates on ``prompt_id`` (int) only. ``prompt_text`` is
ignored because we want to expose the serialization tax, not string handling.
"""

from __future__ import annotations

from typing import Iterable

from .base import Workload, WorkloadResult


class W0Chained(Workload):
    code = "w0"
    name = "chained_trivial"
    result = WorkloadResult(output_columns=[
        ("prompt_id", "int64"),
        ("value", "int64"),
    ])

    def __init__(self, depth: int = 1) -> None:
        super().__init__()
        if depth < 1 or depth > 8:
            raise ValueError(f"depth must be in [1, 8], got {depth}")
        self.depth = depth

    def init(self, cfg: dict, timer=None) -> None:
        self.bind_timer(timer)
        # No models to load.
        return None

    @staticmethod
    def _stage(x: int) -> int:
        return x + 1

    def apply(self, prompt_id: int, prompt_text: str) -> tuple:
        x = int(prompt_id)
        with self._measure("TRIVIAL_COMPUTE"):
            for _ in range(self.depth):
                x = self._stage(x)
        return (int(prompt_id), int(x))

    def apply_batch(
        self, prompt_ids: Iterable[int], prompt_texts: Iterable[str]
    ) -> dict[str, list]:
        ids = list(prompt_ids)
        # Vectorised: d additions applied to each id
        with self._measure("TRIVIAL_COMPUTE"):
            values = [i + self.depth for i in ids]
        return {"prompt_id": ids, "value": values}
