"""Workload base class — defines the contract every W* implements.

Design rule
───────────
The same ``Workload`` payload is called by all four configs (A/B/C/D). The
payload never owns the execution mechanism (UDF, pandas UDF, mapInArrow,
UDTF) — it just promises:

    workload.init(cfg)             # load models into process-local singletons
    workload.apply(prompt_id, prompt_text) -> tuple    # one-row form
    workload.apply_batch(pids, texts) -> dict[col, list]   # batch form

The 4 config modules wrap these into Spark/Sail UDFs with their respective
serialization surface. This keeps the "only the execution path differs"
invariant intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from sail_vs_spark.profiling.boundary_timer import optional_measure


@dataclass
class WorkloadResult:
    """Describes the output schema of a workload for reflection."""

    output_columns: list[tuple[str, str]]
    # list of (name, arrow_type_string). arrow_type_string is e.g.
    # "int64", "string", "float32".


class Workload:
    code: str = "WX"            # e.g. "w0", "w1"
    name: str = "base"
    result: WorkloadResult = WorkloadResult(output_columns=[])

    def __init__(self) -> None:
        self._timer = None

    def bind_timer(self, timer: Any) -> "Workload":
        self._timer = timer
        return self

    def _measure(self, phase: str):
        return optional_measure(self._timer, phase)

    def init(self, cfg: dict, timer: Any | None = None) -> None:  # pragma: no cover
        """Load models + warm caches. Called once per worker."""
        raise NotImplementedError

    def apply(self, prompt_id: int, prompt_text: str) -> tuple:   # pragma: no cover
        """Single-row form. Returns a tuple matching ``result.output_columns``."""
        raise NotImplementedError

    def apply_batch(
        self, prompt_ids: Iterable[int], prompt_texts: Iterable[str]
    ) -> dict[str, list]:
        """Batched form. Default implementation just loops ``apply``.
        Workloads with true-batch APIs (tokenize-all-then-generate) override
        this for performance.
        """
        ids = list(prompt_ids)
        texts = list(prompt_texts)
        out_cols = {name: [] for name, _ in self.result.output_columns}
        for pid, txt in zip(ids, texts):
            row = self.apply(pid, txt)
            for (name, _), val in zip(self.result.output_columns, row):
                out_cols[name].append(val)
        return out_cols
