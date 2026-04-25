"""Deterministic mock models for unit testing and laptop-tier smoke runs.

These are pure-Python (no torch dependency) so the scaffolding and W0 runs work
on a bare Python install. They implement the same call surface as the real
loaders in ``loaders.py``:

    gen = MockGenerator(seed=0)
    outs = gen.generate(["hello", "world"], n=4, max_new_tokens=32)

    sc = MockScorer(seed=0)
    rewards = sc.score(["hello"], ["response"])

    emb = MockEmbedder(dim=16, seed=0)
    vecs = emb.encode(["hello", "world"])
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import List, Sequence


def _seeded_rng(tag: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{seed}|{tag}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


class MockGenerator:
    """Deterministic pseudo-LLM. Returns hash-derived fake completions."""

    def __init__(self, seed: int = 0, latency_ms: float = 0.0) -> None:
        self.seed = seed
        self.latency_ms = latency_ms

    def generate(
        self,
        prompts: Sequence[str],
        n: int = 1,
        max_new_tokens: int = 32,
    ) -> List[List[str]]:
        out: list[list[str]] = []
        for p in prompts:
            cands: list[str] = []
            for i in range(n):
                rng = _seeded_rng(f"gen|{p}|{i}", self.seed)
                n_tokens = rng.randint(4, max(4, max_new_tokens))
                cands.append(" ".join(
                    rng.choice(("alpha", "bravo", "charlie", "delta",
                                "echo", "foxtrot", "golf", "hotel"))
                    for _ in range(n_tokens)
                ))
            out.append(cands)
        if self.latency_ms:
            import time
            time.sleep(self.latency_ms / 1000.0 * len(prompts))
        return out



class MockScorer:
    """Deterministic pseudo-reward model.

    Produces a pseudo-reward in [-1, 1] as a hash of (prompt, response).
    Uses a fixed seed so Best-of-N always picks the same candidate for a given
    prompt across runs — required for deterministic quality comparisons.
    """

    def __init__(self, seed: int = 0, latency_ms: float = 0.0) -> None:
        self.seed = seed
        self.latency_ms = latency_ms

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> List[float]:
        if len(prompts) != len(responses):
            raise ValueError(
                f"prompts ({len(prompts)}) and responses ({len(responses)}) "
                "must have the same length"
            )
        out: list[float] = []
        for p, r in zip(prompts, responses):
            rng = _seeded_rng(f"score|{p}|{r}", self.seed)
            # Uniform in [-1, 1]
            out.append(rng.uniform(-1.0, 1.0))
        if self.latency_ms:
            import time
            time.sleep(self.latency_ms / 1000.0 * len(prompts))
        return out



class MockEmbedder:
    """Deterministic pseudo-embedding model.

    Produces unit-length embeddings in R^dim derived from a hash of the text.
    """

    def __init__(self, dim: int = 64, seed: int = 0, latency_ms: float = 0.0) -> None:
        self.dim = dim
        self.seed = seed
        self.latency_ms = latency_ms

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        out: list[list[float]] = []
        for t in texts:
            rng = _seeded_rng(f"embed|{t}", self.seed)
            v = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
            norm = math.sqrt(sum(x * x for x in v))
            if norm == 0:
                norm = 1.0
            out.append([x / norm for x in v])
        if self.latency_ms:
            import time
            time.sleep(self.latency_ms / 1000.0 * len(texts))
        return out

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        # a and b are unit-normed by encode(); clamp for numerical safety
        return max(-1.0, min(1.0, dot))
