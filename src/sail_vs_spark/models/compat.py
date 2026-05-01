"""Compatibility and offline-loading patches for model runtimes."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

_FP8_DTYPE_NAMES: frozenset[str] = frozenset(
    {
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    }
)

_KERNEL_PATCH_APPLIED = False


def patch_snapshot_list_int(snapshot_dir: Path) -> None:
    """Replace list[int] with List[int] in kernel .py files (PyTorch 2.6 fix)."""
    import re

    for py_file in snapshot_dir.rglob("*.py"):
        try:
            content = py_file.read_text()
            if "list[int]" not in content:
                continue
            first = re.search(r"from typing import ([^\n]+)", content)
            if first:
                if "List" not in first.group(1):
                    content = content.replace(
                        first.group(0),
                        first.group(0).replace("from typing import ", "from typing import List, "),
                        1,
                    )
            else:
                content = "from typing import List\n" + content
            content = content.replace("list[int]", "List[int]")
            py_file.write_text(content)
        except Exception:
            pass


def patch_offline_kernel_loading() -> None:
    global _KERNEL_PATCH_APPLIED

    try:
        import transformers.integrations.finegrained_fp8 as _fp8_stale

        if getattr(_fp8_stale, "_triton_available", None) is False:
            print("[loaders] resetting stale _triton_available=False in finegrained_fp8")
            _fp8_stale._triton_available = None
        if getattr(_fp8_stale, "_deepgemm_available", None) is False:
            _fp8_stale._deepgemm_available = None
    except ImportError:
        pass

    if _KERNEL_PATCH_APPLIED:
        return

    try:
        import huggingface_hub

        _orig_refs = huggingface_hub.HfApi.list_repo_refs

        def _safe_list_repo_refs(self, repo_id: str, *args, **kwargs):
            try:
                return _orig_refs(self, repo_id, *args, **kwargs)
            except Exception:
                hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                refs_dir = Path(hf_home) / "hub" / f"models--{repo_id.replace('/', '--')}" / "refs"
                branches: list = []
                tags: list = []
                if refs_dir.is_dir():
                    for ref_file in refs_dir.iterdir():
                        if not ref_file.is_file():
                            continue
                        commit = ref_file.read_text().strip()
                        ref = SimpleNamespace(name=ref_file.name, target_commit=commit)
                        (branches if ref_file.name == "main" else tags).append(ref)
                if not branches:
                    branches = [SimpleNamespace(name="main", target_commit="main")]
                main_commit = branches[0].target_commit
                for vname in ("v1.0.0", "v1.0", "v1", "1.0.0", "1"):
                    if not any(t.name == vname for t in tags):
                        tags.append(SimpleNamespace(name=vname, target_commit=main_commit))
                return SimpleNamespace(branches=branches, tags=tags)

        huggingface_hub.HfApi.list_repo_refs = _safe_list_repo_refs

        try:
            from transformers.integrations import hub_kernels as _hk

            _orig_get_kernel = _hk.get_kernel

            def _safe_get_kernel(kernel_name, **kwargs):
                try:
                    return _orig_get_kernel(kernel_name, **kwargs)
                except Exception as e:
                    repo_id = kernel_name
                    short_name = repo_id.split("/")[-1]
                    cache_key = f"models--{repo_id.replace('/', '--')}"

                    search_roots: list[Path] = []
                    hf_home = os.environ.get("HF_HOME", "")
                    if hf_home:
                        search_roots.append(Path(hf_home) / "hub")
                    search_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
                    repo_root = Path(__file__).parent.parent.parent.parent
                    search_roots.append(repo_root / ".cache" / "huggingface" / "hub")

                    for hub_dir in search_roots:
                        snapshots_dir = hub_dir / cache_key / "snapshots"
                        if not snapshots_dir.is_dir():
                            continue
                        snapshots = sorted(snapshots_dir.iterdir())
                        if not snapshots:
                            continue
                        snapshot = snapshots[-1]
                        patch_snapshot_list_int(snapshot)
                        for sub in ("build/torch-cuda", "build", ""):
                            variant_path = snapshot / sub if sub else snapshot
                            if variant_path.is_dir():
                                try:
                                    from kernels.utils import _import_from_path

                                    pkg = short_name.replace("-", "_")
                                    return _import_from_path(pkg, variant_path)
                                except Exception:
                                    continue
                    raise RuntimeError(
                        f"[loaders] kernel '{kernel_name}' not in any cache root: "
                        + ", ".join(str(root / cache_key) for root in search_roots)
                    ) from e

            _hk.get_kernel = _safe_get_kernel
        except (ImportError, AttributeError):
            pass

        try:
            import transformers.integrations.finegrained_fp8 as _fp8_mod

            _orig_load_triton = _fp8_mod._load_triton_kernel

            def _retryable_load_triton():
                if getattr(_fp8_mod, "_triton_available", None) is False:
                    _fp8_mod._triton_available = None
                _orig_load_triton()

            _fp8_mod._load_triton_kernel = _retryable_load_triton
        except (ImportError, AttributeError):
            pass

        try:
            from triton.runtime.autotuner import Autotuner as _TritonAutotuner

            if not getattr(_TritonAutotuner, "_nargs_patch_applied", False):
                _orig_triton_bench = _TritonAutotuner._bench

                def _patched_triton_bench(self, *args, config, **meta):
                    if getattr(self, "nargs", None) is None and getattr(self, "arg_names", None) and args:
                        self.nargs = dict(zip(self.arg_names, args))
                    return _orig_triton_bench(self, *args, config=config, **meta)

                _TritonAutotuner._bench = _patched_triton_bench
                _TritonAutotuner._nargs_patch_applied = True
        except (ImportError, AttributeError):
            pass

        _KERNEL_PATCH_APPLIED = True
    except Exception as e:
        print(f"[loaders] offline kernel patch skipped: {e}")


def resolve_model_path(model_id: str) -> str:
    """Check if model exists in local 'models/' dir, else return id."""
    project_root = Path(__file__).parent.parent.parent.parent
    local_path = project_root / "models" / model_id.replace("/", "--")
    if local_path.exists() and local_path.is_dir():
        return str(local_path)
    return model_id
