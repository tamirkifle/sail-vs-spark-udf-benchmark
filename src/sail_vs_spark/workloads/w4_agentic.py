"""W4 — Agentic multi-step loop (SHOWCASE workload for Sail Arrow zero-copy).

Pattern
───────
    for each prompt:
        repeat up to max_iterations:
            generate N candidates → score each → pick best
            if best_reward >= threshold: stop
            else: augment prompt with best response, retry

Why this workload tells the story
──────────────────────────────────
In Spark (configs A/B) every generate+score stage crosses the JVM→socket
boundary. An agentic loop with max_iterations=3 pays that cost 3× per prompt.
In Sail (configs C/D) the entire loop runs inside one Python closure backed by
the zero-copy Arrow stream — the engine boundary is crossed once per batch,
regardless of how many iterations the agent runs.

The serialization tax compounds with iteration count, so this workload makes
Sail's advantage largest and most visible.
"""

from __future__ import annotations

from typing import Iterable

from .base import Workload, WorkloadResult


class W4Agentic(Workload):
    code = "w4"
    name = "agentic_loop"
    result = WorkloadResult(output_columns=[
        ("prompt_id", "int64"),
        ("final_response", "string"),
        ("iterations", "int32"),
        ("best_reward", "float32"),
    ])

    def __init__(
        self,
        max_iterations: int = 3,
        reward_threshold: float = 0.5,
        n_candidates: int = 2,
    ) -> None:
        super().__init__()
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
        self.max_iterations = max_iterations
        self.reward_threshold = reward_threshold
        self.n_candidates = n_candidates
        self._gen = None
        self._sc = None
        self._cfg: dict = {}

    def init(self, cfg: dict, timer=None) -> None:
        from ..models.loaders import get_generator, get_scorer
        self.bind_timer(timer)
        self._cfg = cfg
        mcfg_gen = dict(cfg.get("models", {}).get("generator", {}))
        mcfg_sc = dict(cfg.get("models", {}).get("scorer", {}))
        device = cfg.get("hardware", {}).get("device", "auto")
        mcfg_gen.setdefault("device", device)
        mcfg_sc.setdefault("device", device)
        self._gen = get_generator(mcfg_gen, timer=self._timer)
        self._sc = get_scorer(mcfg_sc, timer=self._timer)

    def _ensure_init(self) -> None:
        if self._gen is None or self._sc is None:
            self.init({"models": {"generator": {"prefer_mock": True},
                                  "scorer": {"prefer_mock": True}},
                       "hardware": {"device": "cpu"}})

    def apply(self, prompt_id: int, prompt_text: str) -> tuple:
        self._ensure_init()
        prompt = prompt_text
        best_resp = ""
        best_reward = -1.0
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            cands = self._gen.generate([prompt], n=self.n_candidates)[0]
            rewards = self._sc.score([prompt_text] * len(cands), cands)
            idx = max(range(len(rewards)), key=lambda i: rewards[i])
            if rewards[idx] > best_reward:
                best_reward = rewards[idx]
                best_resp = cands[idx]
            if best_reward >= self.reward_threshold:
                break
            prompt = f"{prompt_text}\n\nPrevious attempt: {best_resp}\nImprove:"
        return (int(prompt_id), str(best_resp), int(iteration), float(best_reward))

    def apply_batch(
        self, prompt_ids: Iterable[int], prompt_texts: Iterable[str]
    ) -> dict[str, list]:
        self._ensure_init()
        ids = list(prompt_ids)
        texts = list(prompt_texts)
        out: dict[str, list] = {
            "prompt_id": [], "final_response": [], "iterations": [], "best_reward": [],
        }
        for pid, text in zip(ids, texts):
            pid2, resp, iters, reward = self.apply(pid, text)
            out["prompt_id"].append(pid2)
            out["final_response"].append(resp)
            out["iterations"].append(iters)
            out["best_reward"].append(reward)
        return out
