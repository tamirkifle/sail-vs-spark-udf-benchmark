"""W1 — Best-of-N LLM pipeline (PRIMARY workload).

Pattern
───────
    prompt → generate N candidates → score each → argmax → (best, reward)

This is the archetypal RL/inference workload: a generator hot path, a scorer
pass, and an aggregation. In Spark each of these stages is typically a
separate UDF, so the serialization cost is paid *three* times per prompt. In
Sail (C/D) the whole pipeline lives in one Python closure — generation +
scoring + argmax never cross an engine boundary, so the per-prompt IPC cost
is paid once.
"""

from __future__ import annotations

from typing import Iterable

from .base import Workload, WorkloadResult


class W1BestOfN(Workload):
    code = "w1"
    name = "best_of_n"
    result = WorkloadResult(output_columns=[
        ("prompt_id", "int64"),
        ("best_response", "string"),
        ("best_reward", "float32"),
        ("n_candidates", "int32"),
    ])

    def __init__(self, n_candidates: int = 4) -> None:
        if n_candidates < 1 or n_candidates > 32:
            raise ValueError(f"n_candidates must be in [1, 32], got {n_candidates}")
        self.n_candidates = n_candidates
        self._gen = None
        self._sc = None
        self._cfg: dict = {}

    def init(self, cfg: dict) -> None:
        from ..models.loaders import get_generator, get_scorer
        self._cfg = cfg
        mcfg_gen = dict(cfg.get("models", {}).get("generator", {}))
        mcfg_sc = dict(cfg.get("models", {}).get("scorer", {}))
        # Inherit device from the top-level hw config
        device = cfg.get("hardware", {}).get("device", "auto")
        mcfg_gen.setdefault("device", device)
        mcfg_sc.setdefault("device", device)
        self._gen = get_generator(mcfg_gen)
        self._sc = get_scorer(mcfg_sc)


    def _ensure_init(self) -> None:
        if self._gen is None or self._sc is None:
            # Defensive: allow apply() to be called without explicit init by
            # loading with sensible defaults (e.g. for unit tests).
            self.init({"models": {"generator": {"prefer_mock": True},
                                  "scorer": {"prefer_mock": True}},
                       "hardware": {"device": "cpu"}})

    def apply(self, prompt_id: int, prompt_text: str) -> tuple:
        self._ensure_init()
        cands = self._gen.generate([prompt_text], n=self.n_candidates)[0]
        rewards = self._sc.score([prompt_text] * len(cands), cands)
        best_idx = max(range(len(cands)), key=lambda i: rewards[i])
        return (int(prompt_id), str(cands[best_idx]),
                float(rewards[best_idx]), int(len(cands)))

    def apply_batch(
        self, prompt_ids: Iterable[int], prompt_texts: Iterable[str]
    ) -> dict[str, list]:
        self._ensure_init()
        ids = list(prompt_ids)
        texts = list(prompt_texts)

        # Generate N candidates per prompt — list[list[str]]
        all_cands = self._gen.generate(texts, n=self.n_candidates)

        # Flatten for a single batched score call: pairs (prompt, cand)
        flat_prompts: list[str] = []
        flat_cands: list[str] = []
        for p, cands in zip(texts, all_cands):
            for c in cands:
                flat_prompts.append(p)
                flat_cands.append(c)
        flat_rewards = self._sc.score(flat_prompts, flat_cands)


        # Re-bucket and argmax per prompt
        best_ids: list[int] = []
        best_responses: list[str] = []
        best_rewards: list[float] = []
        ncands: list[int] = []
        offset = 0
        for pid, cands in zip(ids, all_cands):
            k = len(cands)
            sub_rewards = flat_rewards[offset: offset + k]
            offset += k
            best_local = max(range(k), key=lambda i: sub_rewards[i])
            best_ids.append(int(pid))
            best_responses.append(str(cands[best_local]))
            best_rewards.append(float(sub_rewards[best_local]))
            ncands.append(int(k))

        return {
            "prompt_id": best_ids,
            "best_response": best_responses,
            "best_reward": best_rewards,
            "n_candidates": ncands,
        }
