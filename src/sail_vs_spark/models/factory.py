"""Lazy model factories with process-local singleton caching."""

from __future__ import annotations

import os
from typing import Any

from .adapters import _HFScorer, _STEmbedder, _VLLMGenerator
from .device import hf_available, resolve_device, torch_available
from .mock import MockEmbedder, MockGenerator, MockScorer

_GENERATOR: Any = None
_SCORER: Any = None
_EMBEDDER: Any = None


def get_generator(cfg: dict) -> Any:
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)

    server_url = cfg.get("server_url") or os.environ.get("VLLM_BASE_URL", "")
    if not prefer_mock and server_url:
        try:
            _GENERATOR = _VLLMGenerator(
                server_url=server_url,
                model_id=cfg["name"],
                max_new_tokens=cfg.get("max_new_tokens", 128),
                temperature=cfg.get("temperature", 0.7),
                top_p=cfg.get("top_p", 0.9),
                top_k=cfg.get("top_k", 50),
            )
            print(f"[loaders] Using vLLM generator at {server_url}")
            return _GENERATOR
        except Exception as exc:
            if not allow_mock:
                raise
            print(f"[loaders] vLLM Generator fell back ({exc})")

    if not allow_mock and not prefer_mock:
        raise RuntimeError(
            "real generation requires a vLLM server_url or VLLM_BASE_URL; "
            "set prefer_mock=true for local smoke runs"
        )

    _GENERATOR = MockGenerator(seed=cfg.get("seed", 0))
    return _GENERATOR


def get_scorer(cfg: dict) -> Any:
    global _SCORER
    if _SCORER is not None:
        return _SCORER

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)
    device = resolve_device(cfg.get("device", "auto"))
    if not prefer_mock and hf_available() and torch_available():
        try:
            _SCORER = _HFScorer(
                model_id=cfg["name"],
                device=device,
                score_batch_size=cfg.get("score_batch_size", 8),
                max_length=cfg.get("max_length", 384),
            )
            return _SCORER
        except Exception as exc:
            if not allow_mock:
                raise
            print(f"[loaders] Scorer fell back to mock ({exc})")

    _SCORER = MockScorer(seed=cfg.get("seed", 0))
    return _SCORER


def get_embedder(cfg: dict) -> Any:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)
    device = resolve_device(cfg.get("device", "auto"))
    if not prefer_mock:
        try:
            _EMBEDDER = _STEmbedder(model_id=cfg["name"], device=device)
            return _EMBEDDER
        except Exception as exc:
            if not allow_mock:
                raise
            print(f"[loaders] Embedder fell back to mock ({exc})")

    _EMBEDDER = MockEmbedder(dim=cfg.get("dim", 64), seed=cfg.get("seed", 0))
    return _EMBEDDER


def reset_singletons() -> None:
    global _GENERATOR, _SCORER, _EMBEDDER
    _GENERATOR = _SCORER = _EMBEDDER = None
