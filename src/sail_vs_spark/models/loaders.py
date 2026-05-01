"""Compatibility facade for lazy model loading."""

from .adapters import _STEmbedder, _VLLMGenerator
from .compat import _FP8_DTYPE_NAMES, patch_offline_kernel_loading as _patch_offline_kernel_loading
from .compat import patch_snapshot_list_int as _patch_snapshot_list_int
from .compat import resolve_model_path as _resolve_model_path
from .device import hf_available as _hf_available
from .device import resolve_device as _resolve_device
from .device import torch_available as _torch_available
from .factory import get_embedder, get_generator, get_scorer, reset_singletons

__all__ = [
    "_FP8_DTYPE_NAMES",
    "_STEmbedder",
    "_VLLMGenerator",
    "_hf_available",
    "_patch_offline_kernel_loading",
    "_patch_snapshot_list_int",
    "_resolve_device",
    "_resolve_model_path",
    "_torch_available",
    "get_embedder",
    "get_generator",
    "get_scorer",
    "reset_singletons",
]
