"""Concrete model adapters."""

from __future__ import annotations

from typing import Any, List, Sequence

from .compat import model_cache_dir, resolve_model_path


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
        cache_dir = str(model_cache_dir())
        resolved_dtype = torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        device_map = "auto" if device == "cuda" else (device if device == "mps" else None)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
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
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(model_cache_dir()),
        )

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        import numpy as np

        vectors = self.model.encode(list(texts), normalize_embeddings=True, batch_size=32)
        return np.asarray(vectors).astype("float32").tolist()

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        return float(sum(x * y for x, y in zip(a, b)))
