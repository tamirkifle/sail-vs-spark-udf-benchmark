"""W2 — Batched LLM inference.

Pattern
───────
    batch(prompts) → generate → one response per prompt

Simpler than W1 — exercises the generator hot path in its batch form. Used
to show pure throughput scaling differences between the four configs when
there is *only one* IPC hop (so any difference comes from IPC efficiency,
not fusion).
"""

from __future__ import annotations

from typing import Iterable

from .base import Workload, WorkloadResult


class W2Batched(Workload):
    code = "w2"
    name = "batched_inference"
    result = WorkloadResult(output_columns=[
        ("prompt_id", "int64"),
        ("response", "string"),
    ])

    def __init__(self) -> None:
        super().__init__()
        self._gen = None
        self._cfg: dict = {}

    def init(self, cfg: dict, timer=None) -> None:
        from ..models.loaders import get_generator
        self.bind_timer(timer)
        self._cfg = cfg
        mcfg = dict(cfg.get("models", {}).get("generator", {}))
        mcfg.setdefault("device",
                        cfg.get("hardware", {}).get("device", "auto"))
        self._gen = get_generator(mcfg, timer=self._timer)

    def _ensure_init(self) -> None:
        if self._gen is None:
            self.init({"models": {"generator": {"prefer_mock": True}},
                       "hardware": {"device": "cpu"}})

    def apply(self, prompt_id: int, prompt_text: str) -> tuple:
        self._ensure_init()
        resp = self._gen.generate([prompt_text], n=1)[0][0]
        return (int(prompt_id), str(resp))

    def apply_batch(
        self, prompt_ids: Iterable[int], prompt_texts: Iterable[str]
    ) -> dict[str, list]:
        self._ensure_init()
        ids = list(prompt_ids)
        texts = list(prompt_texts)
        outs = self._gen.generate(texts, n=1)
        return {
            "prompt_id": ids,
            "response": [o[0] for o in outs],
        }
