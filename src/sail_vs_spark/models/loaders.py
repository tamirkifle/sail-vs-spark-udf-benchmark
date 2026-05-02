"""Compatibility facade for lazy model loading."""

from .adapters import _HFGenerator, _STEmbedder, _VLLMGenerator
from .compat import resolve_model_path as _resolve_model_path
from .device import hf_available as _hf_available
from .device import resolve_device as _resolve_device
from .device import torch_available as _torch_available
from .factory import get_embedder, get_generator, get_scorer, reset_singletons

__all__ = [
    "_STEmbedder",
    "_HFGenerator",
    "_VLLMGenerator",
    "_hf_available",
    "_resolve_device",
    "_resolve_model_path",
    "_torch_available",
    "get_embedder",
    "get_generator",
    "get_scorer",
    "reset_singletons",
]
