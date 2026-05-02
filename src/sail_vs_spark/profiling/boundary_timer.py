"""BoundaryTimer — phase-level timing that survives cloudpickle.

PySpark's Spark Connect cloudpickles every UDF closure even when executing on
Sail (because the closure flows over the client→server gRPC boundary before
Sail ever sees it). ``threading.Lock`` is not picklable, so we drop the lock
in ``__getstate__`` and recreate it in ``__setstate__`` — accumulated timing
data is preserved.

Phases
──────
DATA_TRANSFER_IN    Time to receive data from the engine (deserialize in Spark;
                    pointer-share in Sail — should be ~0 ms)
MODEL_LOAD          One-time model initialisation (amortised; appears once)
TOKENIZE            Convert strings to token tensors
INFERENCE           GPU/CPU compute for token generation (cuda.synchronize
                    guards on either side for accurate GPU timing)
DETOKENIZE          Decode output token tensors back to strings
SCORE               Reward-model forward pass
EMBED               Embedding model forward pass (W3)
SIMILARITY          cosine / dot-product matmul (W3)
TRIVIAL_COMPUTE     W0 ``x + 1`` (and friends)
DATA_TRANSFER_OUT   Time to return results to the engine (serialize in Spark;
                    pointer-share in Sail — should be ~0 ms)
OTHER               catch-all for small helpers

Usage
─────
    timer = BoundaryTimer("config_c")
    with timer.measure("DATA_TRANSFER_IN"):
        texts = batch.column("text").to_pylist()
    with timer.measure("INFERENCE"):
        out = model.generate(...)
    timer.save("results/config_c_boundary.json")
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any



class BoundaryTimer:
    """Thread-safe, pickle-safe phase timer."""

    PHASES: tuple[str, ...] = (
        "DATA_TRANSFER_IN",
        "MODEL_LOAD",
        "TOKENIZE",
        "INFERENCE",
        "DETOKENIZE",
        "SCORE",
        "EMBED",
        "SIMILARITY",
        "TRIVIAL_COMPUTE",
        "DATA_TRANSFER_OUT",
        "UDF_BATCH_EXECUTION",
        "UDF_ROW_EXECUTION",
        "OTHER",
    )

    def __init__(self, config_name: str, enable_tracing: bool = False) -> None:
        self.config_name = config_name
        self.enable_tracing = enable_tracing
        self._lock = threading.Lock()
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._trace_events: list[dict[str, Any]] = []
        self._job_start = time.perf_counter()

    # ── Pickle support ────────────────────────────────────────────────────
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)   # threading.Lock is not picklable
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()  # recreate after unpickling


    # ── Span context manager ──────────────────────────────────────────────
    class _Span:
        __slots__ = ("_parent", "_phase", "_t0", "_wall_t0")

        def __init__(self, parent: "BoundaryTimer", phase: str) -> None:
            self._parent = parent
            self._phase = phase
            self._t0: float | None = None
            self._wall_t0: float | None = None

        def __enter__(self) -> "BoundaryTimer._Span":
            self._t0 = time.perf_counter()
            if self._parent.enable_tracing:
                self._wall_t0 = time.time()
            return self

        def __exit__(self, *_: Any) -> None:
            elapsed = time.perf_counter() - self._t0  # type: ignore[operator]
            with self._parent._lock:
                self._parent._totals[self._phase] += elapsed
                self._parent._counts[self._phase] += 1
                self._parent._samples[self._phase].append(elapsed)
                if self._parent.enable_tracing and self._wall_t0 is not None:
                    import threading, os
                    self._parent._trace_events.append({
                        "name": self._phase,
                        "cat": "benchmark",
                        "ph": "X",
                        "ts": self._wall_t0 * 1_000_000, # microseconds
                        "dur": elapsed * 1_000_000,
                        "pid": os.getpid(),
                        "tid": threading.get_ident()
                    })

    def measure(self, phase: str) -> "BoundaryTimer._Span":
        if phase not in self.PHASES:
            raise ValueError(f"Unknown phase {phase!r}. Expected one of {self.PHASES}")
        return self._Span(self, phase)

    # ── Direct recording (for code paths that don't need a context) ──────
    def record(self, phase: str, elapsed_sec: float) -> None:
        if phase not in self.PHASES:
            raise ValueError(f"Unknown phase {phase!r}")
        with self._lock:
            self._totals[phase] += elapsed_sec
            self._counts[phase] += 1
            self._samples[phase].append(elapsed_sec)
            if self.enable_tracing:
                import threading, os
                wall_now = time.time()
                self._trace_events.append({
                    "name": phase,
                    "cat": "benchmark",
                    "ph": "X",
                    "ts": (wall_now - elapsed_sec) * 1_000_000,
                    "dur": elapsed_sec * 1_000_000,
                    "pid": os.getpid(),
                    "tid": threading.get_ident()
                })

    # ── Merging (for aggregating timers from multiple workers) ────────────
    def merge_from_dict(self, other: dict) -> None:
        """Merge raw phase totals/counts/samples from another timer's state."""
        if not isinstance(other, dict):
            return
        with self._lock:
            for phase, total in (other.get("_totals") or {}).items():
                self._totals[phase] += float(total)
            for phase, count in (other.get("_counts") or {}).items():
                self._counts[phase] += int(count)
            for phase, samples in (other.get("_samples") or {}).items():
                self._samples[phase].extend(float(x) for x in samples)
            if self.enable_tracing and "_trace_events" in other:
                self._trace_events.extend(other["_trace_events"])

    def export_raw(self) -> dict:
        return {
            "_totals": dict(self._totals),
            "_counts": dict(self._counts),
            "_samples": {k: list(v) for k, v in self._samples.items()},
            "_trace_events": list(self._trace_events),
        }

    def save_trace(self, path: str | Path) -> None:
        """Save timeline trace to Chrome Trace Event format (JSON Lines)."""
        if not self.enable_tracing or not self._trace_events:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as fh:
            for event in self._trace_events:
                fh.write(json.dumps(event) + "\n")
        self._trace_events.clear() # clear after saving to avoid double-writing

    # ── Percentile helper ─────────────────────────────────────────────────
    @staticmethod
    def _percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = max(0, int(len(sorted_data) * pct / 100) - 1)
        return sorted_data[idx]


    # ── Report generation ─────────────────────────────────────────────────
    def report(self) -> dict:
        total_wall = time.perf_counter() - self._job_start

        result: dict[str, Any] = {
            "config": self.config_name,
            "total_wall_sec": round(total_wall, 4),
            "phases": {},
        }
        for phase in self.PHASES:
            t = self._totals.get(phase, 0.0)
            n = self._counts.get(phase, 0)
            samples = self._samples.get(phase, [])
            result["phases"][phase] = {
                "total_sec": round(t, 4),
                "call_count": n,
                "avg_ms": round((t / n) * 1000, 3) if n else 0.0,
                "p50_ms": round(self._percentile(samples, 50) * 1000, 3),
                "p95_ms": round(self._percentile(samples, 95) * 1000, 3),
                "pct_of_wall": round((t / total_wall) * 100, 2) if total_wall else 0.0,
            }

        # Derived headline metrics
        serial_sec = (
            self._totals.get("DATA_TRANSFER_IN", 0.0)
            + self._totals.get("DATA_TRANSFER_OUT", 0.0)
        )
        compute_sec = sum(
            self._totals.get(p, 0.0)
            for p in ("INFERENCE", "SCORE", "EMBED", "SIMILARITY",
                      "TOKENIZE", "DETOKENIZE", "TRIVIAL_COMPUTE")
        )
        result["serialization_tax_pct"] = (
            round(serial_sec / total_wall * 100, 2) if total_wall else 0.0
        )
        result["compute_pct"] = (
            round(compute_sec / total_wall * 100, 2) if total_wall else 0.0
        )
        return result


    # ── Persistence + pretty-print ────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        data = self.report()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(data, fh, indent=2)

    def print_summary(self) -> None:
        r = self.report()
        width = 24
        print(f"\n{'─' * 78}")
        print(f"BoundaryTimer: {r['config']}  |  wall={r['total_wall_sec']:.2f}s")
        print(f"{'─' * 78}")
        print(
            f"{'Phase':<{width}} {'Total(s)':>9} {'Calls':>7} "
            f"{'Avg(ms)':>9} {'P95(ms)':>9} {'%Wall':>7}"
        )
        print("─" * 78)
        for phase in self.PHASES:
            d = r["phases"].get(phase, {})
            if not d or d.get("total_sec", 0) == 0:
                continue
            print(
                f"{phase:<{width}} "
                f"{d['total_sec']:>9.3f} "
                f"{d['call_count']:>7d} "
                f"{d['avg_ms']:>9.2f} "
                f"{d['p95_ms']:>9.2f} "
                f"{d['pct_of_wall']:>6.1f}%"
            )
        print("─" * 78)
        print(f"  Serialization tax : {r['serialization_tax_pct']:.2f}%")
        print(f"  Compute           : {r['compute_pct']:.2f}%")
        print()


def optional_measure(timer: BoundaryTimer | None, phase: str):
    if timer is None:
        return nullcontext()
    return timer.measure(phase)
