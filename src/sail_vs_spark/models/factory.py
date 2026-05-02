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


def _bind_timer(instance: Any, timer: Any) -> Any:
    if timer is not None and hasattr(instance, "bind_timer"):
        instance.bind_timer(timer)
    return instance


def get_generator(cfg: dict, timer: Any | None = None) -> Any:
    global _GENERATOR
    if _GENERATOR is not None:
        return _bind_timer(_GENERATOR, timer)

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)

    server_url = cfg.get("server_url") or os.environ.get("VLLM_BASE_URL", "")
    if not prefer_mock and server_url:
        try:
            if timer is not None:
                with timer.measure("MODEL_LOAD"):
                    _GENERATOR = _VLLMGenerator(
                        server_url=server_url,
                        model_id=cfg["name"],
                        max_new_tokens=cfg.get("max_new_tokens", 128),
                        temperature=cfg.get("temperature", 0.7),
                        top_p=cfg.get("top_p", 0.9),
                        top_k=cfg.get("top_k", 50),
                        timer=timer,
                    )
            else:
                _GENERATOR = _VLLMGenerator(
                    server_url=server_url,
                    model_id=cfg["name"],
                    max_new_tokens=cfg.get("max_new_tokens", 128),
                    temperature=cfg.get("temperature", 0.7),
                    top_p=cfg.get("top_p", 0.9),
                    top_k=cfg.get("top_k", 50),
                    timer=timer,
                )
            print(f"[loaders] Using vLLM generator at {server_url}")
            return _bind_timer(_GENERATOR, timer)
        except Exception as exc:
            if not allow_mock:
                raise
            print(f"[loaders] vLLM Generator fell back ({exc})")

    if not allow_mock and not prefer_mock:
        raise RuntimeError(
            "real generation requires a vLLM server_url or VLLM_BASE_URL; "
            "set prefer_mock=true for local smoke runs"
        )

    _GENERATOR = MockGenerator(seed=cfg.get("seed", 0), timer=timer)
    return _bind_timer(_GENERATOR, timer)


def get_scorer(cfg: dict, timer: Any | None = None) -> Any:
    global _SCORER
    if _SCORER is not None:
        return _bind_timer(_SCORER, timer)

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)
    device = resolve_device(cfg.get("device", "auto"))
    if not prefer_mock and hf_available() and torch_available():
        try:
            if timer is not None:
                with timer.measure("MODEL_LOAD"):
                    _SCORER = _HFScorer(
                        model_id=cfg["name"],
                        device=device,
                        score_batch_size=cfg.get("score_batch_size", 8),
                        max_length=cfg.get("max_length", 384),
                        timer=timer,
                    )
            else:
                _SCORER = _HFScorer(
                    model_id=cfg["name"],
                    device=device,
                    score_batch_size=cfg.get("score_batch_size", 8),
                    max_length=cfg.get("max_length", 384),
                    timer=timer,
                )
            return _bind_timer(_SCORER, timer)
        except Exception as exc:
            if not allow_mock:
                raise
            print(f"[loaders] Scorer fell back to mock ({exc})")

    _SCORER = MockScorer(seed=cfg.get("seed", 0), timer=timer)
    return _bind_timer(_SCORER, timer)


def get_embedder(cfg: dict, timer: Any | None = None) -> Any:
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _bind_timer(_EMBEDDER, timer)

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)
    device = resolve_device(cfg.get("device", "auto"))
    if not prefer_mock:
        try:
            if timer is not None:
                with timer.measure("MODEL_LOAD"):
                    _EMBEDDER = _STEmbedder(model_id=cfg["name"], device=device, timer=timer)
            else:
                _EMBEDDER = _STEmbedder(model_id=cfg["name"], device=device, timer=timer)
            return _bind_timer(_EMBEDDER, timer)
        except Exception as exc:
            if not allow_mock:
                raise
            print(f"[loaders] Embedder fell back to mock ({exc})")

    _EMBEDDER = MockEmbedder(dim=cfg.get("dim", 64), seed=cfg.get("seed", 0), timer=timer)
    return _bind_timer(_EMBEDDER, timer)


def reset_singletons() -> None:
    global _GENERATOR, _SCORER, _EMBEDDER
    _GENERATOR = _SCORER = _EMBEDDER = None
