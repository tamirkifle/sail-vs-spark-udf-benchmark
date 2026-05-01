"""Concrete model adapters."""

from __future__ import annotations

from typing import Any, List, Sequence

from .compat import _FP8_DTYPE_NAMES, patch_offline_kernel_loading, resolve_model_path


class _VLLMGenerator:
    """Calls a running vLLM OpenAI-compatible server via stdlib urllib."""

    _opener = None

    @classmethod
    def _get_opener(cls):
        if cls._opener is None:
            import urllib.request as _ur

            cls._opener = _ur.build_opener(_ur.ProxyHandler({}))
        return cls._opener

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
        import re
        import time
        import urllib.error
        import urllib.request

        think_re = re.compile(r"<think>.*?</think>", re.DOTALL)
        greedy = n == 1
        max_tok = max_new_tokens or self.max_new_tokens
        results: List[List[str]] = []

        for prompt in list(prompts):
            payload: dict[str, Any] = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "n": n,
                "max_tokens": max_tok,
                "temperature": 0 if greedy else self.temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if not greedy:
                payload["top_p"] = self.top_p
                payload["top_k"] = self.top_k
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.server_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            opener = self._get_opener()
            for attempt in range(3):
                try:
                    with opener.open(req, timeout=300) as resp:
                        data = json.loads(resp.read())
                    break
                except urllib.error.HTTPError as e:
                    body_snippet = e.read(512).decode(errors="replace")
                    raise RuntimeError(f"vLLM {e.code} {e.reason}: {body_snippet}") from e
                except urllib.error.URLError:
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)

            candidates = [think_re.sub("", c["message"]["content"]).strip() for c in data["choices"]]
            results.append(candidates)

        return results


class _HFGenerator:
    """Wrapper around transformers.pipeline with proper batching support."""

    def __init__(
        self,
        model_id: str,
        device: str,
        max_new_tokens: int,
        temperature: float,
        dtype: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

        patch_offline_kernel_loading()
        model_name = resolve_model_path(model_id)

        if dtype is None:
            resolved_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        elif isinstance(dtype, str) and dtype in _FP8_DTYPE_NAMES:
            resolved_dtype = torch.bfloat16
            print(
                f"[loaders] dtype={dtype!r} -> bfloat16 compute dtype "
                "(FP8 weights load from safetensors automatically)"
            )
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

        self.pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map=device_map)
        self.gen_config = GenerationConfig.from_model_config(mdl.config)
        self.gen_config.max_new_tokens = max_new_tokens
        self.gen_config.temperature = temperature
        self.gen_config.pad_token_id = tok.pad_token_id

    def generate(
        self, prompts: Sequence[str], n: int = 1, max_new_tokens: int | None = None
    ) -> List[List[str]]:
        prompts_list = list(prompts)

        if n > 1:
            return self._generate_sampling(prompts_list, n, max_new_tokens)

        cfg_dict = self.gen_config.to_dict()
        cfg_dict["do_sample"] = False
        cfg_dict["num_return_sequences"] = 1
        if max_new_tokens:
            cfg_dict["max_new_tokens"] = max_new_tokens

        from transformers import GenerationConfig

        call_config = GenerationConfig.from_dict(cfg_dict)
        results = self.pipe(
            (prompt for prompt in prompts_list),
            batch_size=len(prompts_list),
            generation_config=call_config,
        )
        if len(prompts_list) == 1 and isinstance(results, list) and results and isinstance(results[0], dict):
            results = [results]

        outputs: list[list[str]] = []
        for prompt, result_group in zip(prompts_list, results):
            outputs.append([result["generated_text"][len(prompt) :].strip() for result in result_group])
        return outputs

    def _generate_sampling(
        self, prompts_list: List[str], n: int, max_new_tokens: int | None
    ) -> List[List[str]]:
        import torch
        from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

        class _NaNGuard(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

        orig_multinomial = torch.multinomial

        def _safe_multinomial(input, num_samples, replacement=False, *, generator=None):
            bad = ~torch.isfinite(input) | (input < 0)
            if bad.any():
                input = input.clone().nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0)
                zero_rows = input.sum(-1).eq(0)
                if zero_rows.any():
                    input[zero_rows] = 1.0
            return orig_multinomial(input, num_samples, replacement=replacement, generator=generator)

        torch.multinomial = _safe_multinomial

        cfg_dict = self.gen_config.to_dict()
        cfg_dict["do_sample"] = True
        cfg_dict["num_return_sequences"] = n
        if max_new_tokens:
            cfg_dict["max_new_tokens"] = max_new_tokens
        cfg_dict["top_k"] = max(int(cfg_dict.get("top_k") or 0), 50)
        if not cfg_dict.get("top_p"):
            cfg_dict["top_p"] = 0.95
        cfg_dict["num_beams"] = 1

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
                    results = self.pipe(prompt, generation_config=call_config)

                if isinstance(results, dict):
                    results = [results]
                texts = []
                for result in results:
                    text = result.get("generated_text", "")
                    if isinstance(text, str) and text.startswith(prompt):
                        text = text[len(prompt) :].strip()
                    elif isinstance(text, str):
                        text = text.strip()
                    texts.append(text)
                outputs.append(texts)
        finally:
            torch.multinomial = orig_multinomial

        return outputs


class _HFScorer:
    """Wraps a HF reward model with batched scoring."""

    def __init__(
        self,
        model_id: str,
        device: str,
        *,
        score_batch_size: int = 8,
        max_length: int = 384,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = resolve_model_path(model_id)
        resolved_dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        device_map = "auto" if device == "cuda" else (device if device == "mps" else None)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._torch = torch
        self.score_batch_size = max(1, int(score_batch_size))
        self.max_length = max(1, int(max_length))
        self.mdl.eval()

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> List[float]:
        prompt_list = list(prompts)
        response_list = list(responses)
        if len(prompt_list) != len(response_list):
            raise ValueError(
                f"prompts ({len(prompt_list)}) and responses ({len(response_list)}) "
                "must have the same length"
            )

        scores: list[float] = []
        for start in range(0, len(prompt_list), self.score_batch_size):
            end = start + self.score_batch_size
            inputs = self.tok(
                prompt_list[start:end],
                response_list[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.mdl.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                logits = self.mdl(**inputs).logits.squeeze(-1)
            scores.extend(logits.float().cpu().tolist())
        return scores


class _STEmbedder:
    """Wraps sentence-transformers with native batching."""

    def __init__(self, model_id: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        model_name = resolve_model_path(model_id)
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        import numpy as np

        vectors = self.model.encode(list(texts), normalize_embeddings=True, batch_size=32)
        return np.asarray(vectors).astype("float32").tolist()

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))
