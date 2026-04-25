"""Workload registry — maps short codes to concrete Workload subclasses.

Tests and the runner use ``make_workload("w1", cfg)`` to instantiate the
right subclass with the right kwargs. This keeps the CLI surface flat.
"""

from __future__ import annotations

from typing import Any

from .base import Workload
from .w0_chained import W0Chained
from .w1_best_of_n import W1BestOfN
from .w2_batched import W2Batched
from .w3_embedding import W3Embedding

# Map short-code -> class.  The depth parameter for W0 is read from cfg.
REGISTRY = {
    "w0": W0Chained,
    "w1": W1BestOfN,
    "w2": W2Batched,
    "w3": W3Embedding,
}


def make_workload(code: str, cfg: dict[str, Any]) -> Workload:
    """Instantiate a workload from the top-level config dict."""
    if code not in REGISTRY:
        raise ValueError(
            f"unknown workload {code!r}. Expected one of {list(REGISTRY)}"
        )
    wcfg_all = cfg.get("workloads", {})
    if code == "w0":
        depth = int(wcfg_all.get("w0_chained", {}).get("depth", 1))
        wl = W0Chained(depth=depth)
    elif code == "w1":
        n = int(wcfg_all.get("w1_best_of_n", {}).get("n_candidates", 4))
        wl = W1BestOfN(n_candidates=n)
    elif code == "w2":
        wl = W2Batched()
    elif code == "w3":
        n = int(wcfg_all.get("w3_embedding", {}).get("n_queries", 5))
        wl = W3Embedding(n_queries=n)
    else:   # pragma: no cover
        raise RuntimeError(f"Unreachable: {code}")

    wl.init(cfg)
    return wl
