"""Lazy model loaders — all heavy imports happen *inside* the functions."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Sequence

from .mock import MockEmbedder, MockGenerator, MockScorer


# FP8 dtype string names that are invalid as torch.set_default_dtype() args.
# PyTorch has no Float8_*Storage implementation, so passing any of these to
# from_pretrained(dtype=...) triggers "couldn't find storage object Float8_*".
# For FP8-quantized models, weights are stored as FP8 in safetensors and load
# correctly regardless of this arg; dtype= only controls the COMPUTE dtype
# (non-quantized layers: LayerNorm, embeddings, biases). Use bfloat16.
_FP8_DTYPE_NAMES: frozenset[str] = frozenset({
    "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz",
})


# ── Offline kernel-loader patch ──────────────────────────────────────────────
# On compute nodes with no internet, the `kernels` library (used by
# transformers FP8 MoE integration) makes two network calls even when the
# snapshot is already cached:
#   1. HfApi.list_repo_refs()  — version check, fails with OfflineModeIsEnabled
#   2. kernels version resolver — needs version tags, raises ValueError if missing
# We patch both, and also fix the list[int] PyTorch 2.6 type-annotation bug
# in the snapshot files before they are imported.

_KERNEL_PATCH_APPLIED = False


def _patch_snapshot_list_int(snapshot_dir: Path) -> None:
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


def _patch_offline_kernel_loading() -> None:
    global _KERNEL_PATCH_APPLIED

    # Reset stale failure state — reused Sail Python workers carry
    # _triton_available=False from a prior failed kernel load.
    # _triton_available semantics: None=not attempted, True=loaded, False=failed-won't-retry.
    # We reset False→None before the _KERNEL_PATCH_APPLIED guard so every call
    # can clear the stale flag, even after the one-time patches are applied.
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

        # ── Patch 1: list_repo_refs ──────────────────────────────────────────
        # Falls back to on-disk refs/ files and adds synthetic version tags
        # so the kernels version resolver can find "version 1".
        _orig_refs = huggingface_hub.HfApi.list_repo_refs

        def _safe_list_repo_refs(self, repo_id: str, *args, **kwargs):
            try:
                return _orig_refs(self, repo_id, *args, **kwargs)
            except Exception:
                hf_home = os.environ.get(
                    "HF_HOME", os.path.expanduser("~/.cache/huggingface")
                )
                refs_dir = (
                    Path(hf_home) / "hub"
                    / f"models--{repo_id.replace('/', '--')}"
                    / "refs"
                )
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
                # Synthesise version tags v1/v1.0.0/1 pointing at cached snapshot
                # so kernels/_versions.py resolve_version_spec_as_ref succeeds.
                main_commit = branches[0].target_commit
                for vname in ("v1.0.0", "v1.0", "v1", "1.0.0", "1"):
                    if not any(t.name == vname for t in tags):
                        tags.append(SimpleNamespace(name=vname, target_commit=main_commit))
                return SimpleNamespace(branches=branches, tags=tags)

        huggingface_hub.HfApi.list_repo_refs = _safe_list_repo_refs

        # ── Patch 2: hub_kernels.get_kernel offline fallback ─────────────────
        # lazy_load_kernel (hub_kernels.py:378) calls get_kernel() via module-level
        # name lookup — patching hub_kernels.get_kernel intercepts that call.
        # On failure, the fallback searches the local HF snapshot cache directly.
        try:
            from transformers.integrations import hub_kernels as _hk
            _orig_get_kernel = _hk.get_kernel

            def _safe_get_kernel(kernel_name, **kwargs):
                try:
                    return _orig_get_kernel(kernel_name, **kwargs)
                except Exception as e:
                    # kernel_name is the full repo_id e.g.
                    # "kernels-community/finegrained-fp8"
                    repo_id = kernel_name
                    short_name = repo_id.split("/")[-1]
                    cache_key = f"models--{repo_id.replace('/', '--')}"

                    # Search every plausible HF cache root
                    search_roots: list[Path] = []
                    hf_home = os.environ.get("HF_HOME", "")
                    if hf_home:
                        search_roots.append(Path(hf_home) / "hub")
                    search_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
                    # repo-relative cache (set by prep_download.sh / submit scripts)
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
                        _patch_snapshot_list_int(snapshot)
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
                        + ", ".join(str(r / cache_key) for r in search_roots)
                    ) from e

            _hk.get_kernel = _safe_get_kernel
        except (ImportError, AttributeError):
            pass

        # ── Patch 3: retryable _load_triton_kernel ───────────────────────────
        # _load_triton_kernel sets _triton_available=False at the top (line 81:
        # "mark attempted before any early exit") and raises if lazy_load_kernel
        # fails — leaving the False flag as a permanent "won't retry" block.
        # In Sail, the same Python worker handles multiple partitions: a failed
        # partition 1 permanently blocks partitions 2-N in the same process.
        # This wrapper resets _triton_available=None before each call so every
        # partition gets an independent retry using the now-patched get_kernel.
        # Calling _orig_load_triton() (not a reimplementation) ensures the original
        # function sets all module-level globals (triton_fp8_act_quant etc.) — the
        # previous NoneType error was caused by bypassing this original assignment.
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

        # ── Patch 4: Triton Autotuner nargs fix ──────────────────────────────
        # Autotuner._bench() does {**self.nargs, **current} but self.nargs is
        # None/missing. Root cause: in this Triton version, run() (the entry
        # point for kernel[grid](...) calls) resets or never sets self.nargs
        # before delegating to _bench(). Patching run() is insufficient because
        # run() resets it again before calling _bench(). Patch _bench() directly
        # — that is the only reliable site, right at the point of use.
        # _bench() receives the same positional kernel args as run(), so
        # dict(zip(self.arg_names, args)) reconstructs the correct nargs dict.
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

# ── Path Resolution ─────────────────────────────────────────────────────────

def _resolve_model_path(model_id: str) -> str:
    """Check if model exists in local 'models/' dir, else return id."""
    project_root = Path(__file__).parent.parent.parent.parent
    local_path = project_root / "models" / model_id.replace("/", "--")
    if local_path.exists() and local_path.is_dir():
        return str(local_path)
    return model_id


# ── Process-local singletons ────────────────────────────────────────────────
_GENERATOR: Any = None
_SCORER: Any = None
_EMBEDDER: Any = None


def _torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


def _hf_available() -> bool:
    try:
        import transformers
        return True
    except ImportError:
        return False


def _resolve_device(requested: str, component: str = "model") -> str:
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


class _VLLMGenerator:
    """Calls a running vLLM OpenAI-compatible server via stdlib urllib.

    vLLM handles FP8 quantization internally — no finegrained-fp8,
    causal-conv1d, or deep-gemm kernels required in the UDF process.
    Uses /v1/completions (not /v1/chat/completions) so that a batch of
    prompts can be submitted in a single HTTP request.
    """

    def __init__(
        self,
        server_url: str,
        model_id: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def generate(
        self,
        prompts: Sequence[str],
        n: int = 1,
        max_new_tokens: int | None = None,
    ) -> List[List[str]]:
        import json
        import time
        import urllib.error
        import urllib.request

        prompts_list = list(prompts)
        payload = {
            "model": self.model_id,
            "prompt": prompts_list,
            "n": n,
            "max_tokens": max_new_tokens or self.max_new_tokens,
            # temperature=0 selects greedy decoding — fastest path for n=1.
            "temperature": 0 if n == 1 else self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,  # vLLM extension; ignored by strict OpenAI clients
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.server_url}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.URLError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        choices = data["choices"]  # len = len(prompts_list) * n
        k = len(prompts_list)
        return [
            [choices[i * n + j]["text"].strip() for j in range(n)]
            for i in range(k)
        ]


class _HFGenerator:
    """Wrapper around transformers.pipeline with proper batching support."""

    def __init__(self, model_id: str, device: str, max_new_tokens: int,
                 temperature: float, dtype: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig

        _patch_offline_kernel_loading()
        model_name = _resolve_model_path(model_id)

        if dtype is None:
            resolved_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        elif isinstance(dtype, str) and dtype in _FP8_DTYPE_NAMES:
            # FP8 weights load from safetensors as-is; dtype= here is the
            # compute dtype only. Passing float8_e4m3fn would crash PyTorch
            # inside transformers' local_torch_dtype context manager.
            resolved_dtype = torch.bfloat16
            print(f"[loaders] dtype={dtype!r} → bfloat16 compute dtype "
                  f"(FP8 weights load from safetensors automatically)")
        elif isinstance(dtype, str) and hasattr(torch, dtype):
            resolved_dtype = getattr(torch, dtype)
        else:
            resolved_dtype = dtype
            
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        
        device_map = "auto" if device == "cuda" else (device if device == "mps" else None)
        
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype=resolved_dtype,
            device_map=device_map,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.pipe = pipeline(
            "text-generation", model=mdl, tokenizer=tok, device_map=device_map
        )
        
        self.gen_config = GenerationConfig.from_model_config(mdl.config)
        self.gen_config.max_new_tokens = max_new_tokens
        self.gen_config.temperature = temperature
        self.gen_config.pad_token_id = tok.pad_token_id

    def generate(self, prompts: Sequence[str], n: int = 1,
                 max_new_tokens: int | None = None) -> List[List[str]]:
        prompts_list = list(prompts)

        if n > 1:
            return self._generate_sampling(prompts_list, n, max_new_tokens)

        # n == 1: greedy via pipeline — pipeline applies the chat template for
        # instruct/chat models, which is required for correct input formatting.
        cfg_dict = self.gen_config.to_dict()
        cfg_dict["do_sample"] = False
        cfg_dict["num_return_sequences"] = 1
        if max_new_tokens:
            cfg_dict["max_new_tokens"] = max_new_tokens

        from transformers import GenerationConfig
        call_config = GenerationConfig.from_dict(cfg_dict)

        results = self.pipe(
            (p for p in prompts_list),
            batch_size=len(prompts_list),
            generation_config=call_config,
        )

        if (len(prompts_list) == 1 and isinstance(results, list)
                and results and isinstance(results[0], dict)):
            results = [results]

        outputs: list[list[str]] = []
        for p, result_group in zip(prompts_list, results):
            outputs.append([r["generated_text"][len(p):].strip() for r in result_group])
        return outputs

    def _generate_sampling(self, prompts_list: List[str], n: int,
                           max_new_tokens: int | None) -> List[List[str]]:
        """
        Generate n samples per prompt for Best-of-N via the pipeline API.

        Why pipeline and not model.generate() directly:
        The generator models used in this benchmark are instruct/chat-tuned
        (e.g. Qwen3-MoE-FP8). Chat models have a tokenizer chat-template that
        wraps raw text in role markers before tokenization. Calling
        model.generate() with plain text tokens bypasses this and produces
        malformed input_ids for the model's internal preprocessing, which on
        Qwen3-MoE manifests as a 0-length sequence reshape crash. The
        transformers pipeline applies the chat template automatically when the
        tokenizer has one, so it is the correct API for instruct models.
        Using pipeline for both n=1 and n>1 keeps the code path uniform.

        Two layers of NaN defence are kept:
          Layer 1 — NaNGuard LogitsProcessor forwarded to the pipeline.
          Layer 2 — torch.multinomial patch active for the entire call.
        """
        import torch
        from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

        class _NaNGuard(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

        _orig_multinomial = torch.multinomial

        def _safe_multinomial(input, num_samples, replacement=False, *, generator=None):
            bad = ~torch.isfinite(input) | (input < 0)
            if bad.any():
                input = input.clone().nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0)
                zero_rows = input.sum(-1).eq(0)
                if zero_rows.any():
                    input[zero_rows] = 1.0
            return _orig_multinomial(input, num_samples, replacement=replacement,
                                     generator=generator)

        torch.multinomial = _safe_multinomial

        cfg_dict = self.gen_config.to_dict()
        cfg_dict["do_sample"] = True
        cfg_dict["num_return_sequences"] = n
        if max_new_tokens:
            cfg_dict["max_new_tokens"] = max_new_tokens
        cfg_dict["top_k"] = max(int(cfg_dict.get("top_k") or 0), 50)
        if not cfg_dict.get("top_p"):
            cfg_dict["top_p"] = 0.95
        cfg_dict["num_beams"] = 1  # sampling is incompatible with beam search

        call_config = GenerationConfig.from_dict(cfg_dict)
        nan_guard = LogitsProcessorList([_NaNGuard()])

        outputs: list[list[str]] = []
        try:
            for prompt in prompts_list:
                try:
                    results = self.pipe(
                        prompt,
                        generation_config=call_config,
                        logits_processor=nan_guard,
                    )
                except TypeError:
                    # Older pipeline versions don't accept logits_processor kwarg
                    results = self.pipe(prompt, generation_config=call_config)

                # Normalise: single-prompt pipeline returns list[dict] directly
                if isinstance(results, dict):
                    results = [results]
                texts = []
                for r in results:
                    t = r.get("generated_text", "")
                    # Pipeline returns prompt+continuation by default; strip prefix.
                    if isinstance(t, str) and t.startswith(prompt):
                        t = t[len(prompt):].strip()
                    elif isinstance(t, str):
                        t = t.strip()
                    texts.append(t)
                outputs.append(texts)
        finally:
            torch.multinomial = _orig_multinomial

        return outputs


class _HFScorer:
    """Wraps a HF reward model with batched scoring."""

    def __init__(self, model_id: str, device: str, dtype: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_name = _resolve_model_path(model_id)
        
        if dtype is None:
            resolved_dtype = torch.float16 if device == "cuda" else "auto"
        elif isinstance(dtype, str) and dtype in _FP8_DTYPE_NAMES:
            resolved_dtype = torch.float16
            print(f"[loaders] scorer dtype={dtype!r} → float16 compute dtype")
        elif isinstance(dtype, str) and hasattr(torch, dtype):
            resolved_dtype = getattr(torch, dtype)
        else:
            resolved_dtype = dtype

        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        device_map = "auto" if device == "cuda" else (device if device == "mps" else None)
        
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            dtype=resolved_dtype,
            device_map=device_map,
            local_files_only=True,
            trust_remote_code=True
        )
        self.device = self.mdl.device
        self.mdl.eval()
        self._torch = torch

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> List[float]:
        inputs = self.tok(
            list(prompts), list(responses),
            return_tensors="pt", padding=True, truncation=True, max_length=512,
        )
        inputs = {k: v.to(self.mdl.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.mdl(**inputs).logits.squeeze(-1)
        return logits.float().cpu().tolist()


class _STEmbedder:
    """Wraps sentence-transformers with native batching."""

    def __init__(self, model_id: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer
        model_name = _resolve_model_path(model_id)
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        import numpy as np
        v = self.model.encode(list(texts), normalize_embeddings=True, batch_size=32)
        return np.asarray(v).astype("float32").tolist()

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))


def get_generator(cfg: dict) -> Any:
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR

    prefer_mock = cfg.get("prefer_mock", False)
    allow_mock = cfg.get("allow_mock", True)

    if not prefer_mock:
        # 1. Try vLLM server (preferred on GPU — no kernel deps in UDF process)
        server_url = cfg.get("server_url") or os.environ.get("VLLM_BASE_URL", "")
        if server_url:
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
            except Exception as e:
                if not allow_mock:
                    raise
                print(f"[loaders] vLLM Generator fell back ({e})")

        # 2. Try HF Transformers (laptop / MPS)
        device = _resolve_device(cfg.get("device", "auto"), "Generator")
        if _hf_available() and _torch_available():
            try:
                _GENERATOR = _HFGenerator(
                    model_id=cfg["name"], device=device,
                    max_new_tokens=cfg.get("max_new_tokens", 32),
                    temperature=cfg.get("temperature", 0.7),
                    dtype=cfg.get("dtype"),
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
            _SCORER = _HFScorer(
                model_id=cfg["name"], 
                device=device,
                dtype=cfg.get("dtype"),
            )
            return _SCORER
        except Exception as e:
            if not allow_mock: raise
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
            _EMBEDDER = _STEmbedder(model_id=cfg["name"], device=device)
            return _EMBEDDER
        except Exception as e:
            if not allow_mock: raise
            print(f"[loaders] Embedder fell back to mock ({e})")
    _EMBEDDER = MockEmbedder(dim=cfg.get("dim", 64), seed=cfg.get("seed", 0))
    return _EMBEDDER


def reset_singletons() -> None:
    global _GENERATOR, _SCORER, _EMBEDDER
    _GENERATOR = _SCORER = _EMBEDDER = None
