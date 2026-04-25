"""Lazy model loaders — all heavy imports happen *inside* the functions.

Design goals
────────────
1. Modules are importable on machines that don't have ``torch``/``transformers``
   — so the scaffolding can be tested on a bare Python install.
2. The loader returned by ``get_generator(cfg)`` etc. looks identical whether
   it wraps a real HF model or a ``Mock*`` — same ``.generate(...)``,
   ``.score(...)``, ``.encode(...)`` surface.
3. Models are cached *process-local* (once per worker) via module-level
   singletons so that:
      - The first call in a worker pays the MODEL_LOAD cost once.
      - The UDF closure itself stays picklable (no torch objects captured).

The loader functions accept a plain dict (loaded from YAML) so they can be
invoked from anywhere without pulling the full config schema.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Sequence

from .mock import MockEmbedder, MockGenerator, MockScorer

# ── Process-local singletons ────────────────────────────────────────────────
_GENERATOR: Any = None
_SCORER: Any = None
_EMBEDDER: Any = None


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _hf_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False



def _resolve_device(requested: str) -> str:
    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError(f"unknown device {requested!r}")
    if requested != "auto":
        return requested
    if not _torch_available():
        return "cpu"
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _HFGenerator:
    """Thin wrapper around transformers.pipeline('text-generation')."""

    def __init__(self, model_name: str, device: str, max_new_tokens: int,
                 temperature: float) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
        )
        if device == "cuda":
            mdl = mdl.to("cuda")
        elif device == "mps":
            mdl = mdl.to("mps")

        self.pipe = pipeline(
            "text-generation", model=mdl, tokenizer=tok,
            device=mdl.device,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature


    def generate(self, prompts: Sequence[str], n: int = 1,
                 max_new_tokens: int | None = None) -> List[List[str]]:
        mnt = max_new_tokens or self.max_new_tokens
        outputs: list[list[str]] = []
        for p in prompts:
            result = self.pipe(
                p,
                max_new_tokens=mnt,
                temperature=self.temperature,
                do_sample=n > 1,
                num_return_sequences=n,
                pad_token_id=self.pipe.tokenizer.pad_token_id,
            )
            outputs.append([r["generated_text"][len(p):].strip()
                            for r in result])
        return outputs


class _HFScorer:
    """Wraps a HF reward/sequence-classification model."""

    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device
        if device in ("cuda", "mps"):
            self.mdl = self.mdl.to(device)
        self.mdl.eval()
        self._torch = torch

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> List[float]:
        inputs = self.tok(
            list(prompts), list(responses),
            return_tensors="pt", padding=True, truncation=True, max_length=512,
        )
        if self.device in ("cuda", "mps"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.mdl(**inputs).logits.squeeze(-1)
        return logits.float().cpu().tolist()


class _STEmbedder:
    """Wraps sentence-transformers."""

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        import numpy as np
        v = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(v).astype("float32").tolist()

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        # Inputs expected to be unit-norm from normalize_embeddings=True
        return float(sum(x * y for x, y in zip(a, b)))


# ── Public factory functions ────────────────────────────────────────────────
def get_generator(cfg: dict) -> Any:
    """Return (singleton) generator instance, real or mock."""
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)
    device = _resolve_device(cfg.get("device", "auto"))

    if not prefer_mock and _hf_available() and _torch_available():
        try:
            _GENERATOR = _HFGenerator(
                model_name=cfg["name"],
                device=device,
                max_new_tokens=cfg.get("max_new_tokens", 32),
                temperature=cfg.get("temperature", 0.7),
            )
            return _GENERATOR
        except Exception as e:
            if not allow_mock:
                raise
            print(f"[loaders] Generator fell back to mock ({e})")

    _GENERATOR = MockGenerator(seed=cfg.get("seed", 0))
    return _GENERATOR


def get_scorer(cfg: dict) -> Any:
    global _SCORER
    if _SCORER is not None:
        return _SCORER

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)
    device = _resolve_device(cfg.get("device", "auto"))

    if not prefer_mock and _hf_available() and _torch_available():
        try:
            _SCORER = _HFScorer(model_name=cfg["name"], device=device)
            return _SCORER
        except Exception as e:
            if not allow_mock:
                raise
            print(f"[loaders] Scorer fell back to mock ({e})")

    _SCORER = MockScorer(seed=cfg.get("seed", 0))
    return _SCORER


def get_embedder(cfg: dict) -> Any:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)
    device = _resolve_device(cfg.get("device", "auto"))

    if not prefer_mock:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            _EMBEDDER = _STEmbedder(model_name=cfg["name"], device=device)
            return _EMBEDDER
        except Exception as e:
            if not allow_mock:
                raise
            print(f"[loaders] Embedder fell back to mock ({e})")

    _EMBEDDER = MockEmbedder(dim=cfg.get("dim", 64), seed=cfg.get("seed", 0))
    return _EMBEDDER


def reset_singletons() -> None:
    """For tests: clear cached singletons between runs."""
    global _GENERATOR, _SCORER, _EMBEDDER
    _GENERATOR = _SCORER = _EMBEDDER = None
