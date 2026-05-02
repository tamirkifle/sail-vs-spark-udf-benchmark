"""Compatibility helpers for local model paths."""

from __future__ import annotations

from pathlib import Path


def model_cache_dir() -> Path:
    """Repo-local model cache used by Hugging Face/Transformers loaders."""
    project_root = Path(__file__).parent.parent.parent.parent
    path = project_root / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_model_path(model_id: str) -> str:
    """Check if model exists in local 'models/' dir, else return id."""
    local_path = model_cache_dir() / model_id.replace("/", "--")
    if local_path.exists() and local_path.is_dir():
        return str(local_path)
    return model_id
