"""W3 — Embedding + similarity pipeline (RAG-style).

Pattern
───────
    text → embedding → similarity scoring (cosine vs N reference queries)

Approximates the first two stages of a RAG pipeline. A fixed bank of
``n_queries`` reference queries is embedded once per worker; each prompt is
then embedded and scored against the bank. The output is the best-matching
reference index and cosine score.

This is the smallest workload (few-dim vectors), so the *relative* impact of
serialization is largest — useful for headline numbers.
"""

from __future__ import annotations

from typing import Iterable, List

from .base import Workload, WorkloadResult


_REFERENCE_QUERIES = [
    "What is machine learning?",
    "Describe quantum computing.",
    "Explain climate change.",
    "How does an internal combustion engine work?",
    "Who was Marie Curie?",
    "Write about the Roman Empire.",
    "Summarise the theory of evolution.",
    "What is deep learning?",
    "How do vaccines work?",
    "Explain supply and demand.",
    "What is blockchain?",
    "Describe the water cycle.",
    "Who wrote Hamlet?",
    "How does photosynthesis work?",
    "Explain general relativity.",
    "What is the GDP of a country?",
    "Describe Renaissance art.",
    "How do computers work?",
    "What is the mitochondria?",
    "Explain the process of sleep.",
]



class W3Embedding(Workload):
    code = "w3"
    name = "embedding_rag"
    result = WorkloadResult(output_columns=[
        ("prompt_id", "int64"),
        ("best_query_idx", "int32"),
        ("best_similarity", "float32"),
    ])

    def __init__(self, n_queries: int = 5) -> None:
        self.n_queries = min(max(1, n_queries), len(_REFERENCE_QUERIES))
        self._emb = None
        self._refs: List[List[float]] | None = None

    def init(self, cfg: dict) -> None:
        from ..models.loaders import get_embedder
        mcfg = dict(cfg.get("models", {}).get("embedder", {}))
        mcfg.setdefault("device",
                        cfg.get("hardware", {}).get("device", "auto"))
        self._emb = get_embedder(mcfg)
        self._refs = self._emb.encode(_REFERENCE_QUERIES[: self.n_queries])

    def _ensure_init(self) -> None:
        if self._emb is None or self._refs is None:
            self.init({"models": {"embedder": {"prefer_mock": True, "dim": 32}},
                       "hardware": {"device": "cpu"}})

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # Both are unit-norm (MockEmbedder normalises; real embedder called
        # with normalize_embeddings=True)
        return sum(x * y for x, y in zip(a, b))


    def apply(self, prompt_id: int, prompt_text: str) -> tuple:
        self._ensure_init()
        [v] = self._emb.encode([prompt_text])
        sims = [self._cosine(v, ref) for ref in self._refs]
        best = max(range(len(sims)), key=lambda i: sims[i])
        return (int(prompt_id), int(best), float(sims[best]))

    def apply_batch(
        self, prompt_ids: Iterable[int], prompt_texts: Iterable[str]
    ) -> dict[str, list]:
        self._ensure_init()
        ids = list(prompt_ids)
        texts = list(prompt_texts)
        vecs = self._emb.encode(texts)

        best_idx: list[int] = []
        best_sim: list[float] = []
        for v in vecs:
            sims = [self._cosine(v, ref) for ref in self._refs]
            k = max(range(len(sims)), key=lambda i: sims[i])
            best_idx.append(int(k))
            best_sim.append(float(sims[k]))

        return {
            "prompt_id": ids,
            "best_query_idx": best_idx,
            "best_similarity": best_sim,
        }
