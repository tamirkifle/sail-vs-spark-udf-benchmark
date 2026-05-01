"""Model/device capability helpers."""

from __future__ import annotations

import os


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def hf_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_device(requested: str, component: str = "model") -> str:
    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError(f"unknown device {requested!r}")

    if requested != "auto":
        resolved = requested
    else:
        if os.path.exists("/dev/nvidia0"):
            resolved = "cuda"
        else:
            try:
                import torch

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    resolved = "mps"
                else:
                    resolved = "cpu"
            except Exception:
                resolved = "cpu"

    print(f"[loaders] {component} requested device '{requested}', resolved to: '{resolved}'")
    return resolved
