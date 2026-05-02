"""MetricsCollector — background sampler for system + GPU + disk telemetry.

Samples (every ``interval`` seconds, default 0.5s) :

- wall clock time
- process CPU %
- process RSS (MB)
- host RAM used (GB)
- GPU SM util, memory util, memory used (via ``nvidia-smi``)
- cumulative bytes written / read (via ``psutil.Process.io_counters``)

On machines without ``nvidia-smi`` the GPU samples are skipped silently —
the benchmark still works on a MacBook (no GPU) and produces the other
metrics unchanged.

In parallel, it launches ``nvidia-smi dmon`` in the background with 1-second
resolution to produce a high-res timeline log for the plotting stage.

Usage
─────
    col = MetricsCollector("w1_config_c")
    col.start()
    run_workload()
    stats = col.stop()
    col.save("results/w1_config_c_stats.json")
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


class MetricsCollector:
    """Background thread that samples system + GPU + disk telemetry."""

    DEFAULT_SAMPLE_INTERVAL_SEC = 0.5
    GPU_UTIL_ACTIVE_THRESHOLD = 10    # used for pipeline_continuity calc
    VLLM_METRIC_ALIASES = {
        "vllm_gpu_cache_usage_pct": {
            "vllm:gpu_cache_usage_perc",
            "vllm_gpu_cache_usage_perc",
        },
        "vllm_requests_running": {
            "vllm:num_requests_running",
            "vllm_num_requests_running",
        },
        "vllm_requests_waiting": {
            "vllm:num_requests_waiting",
            "vllm_num_requests_waiting",
        },
        "vllm_prompt_tokens_total": {
            "vllm:prompt_tokens_total",
            "vllm_prompt_tokens_total",
        },
        "vllm_generation_tokens_total": {
            "vllm:generation_tokens_total",
            "vllm_generation_tokens_total",
        },
        "vllm_request_queue_time_seconds_sum": {
            "vllm:request_queue_time_seconds_sum",
            "vllm_request_queue_time_seconds_sum",
        },
        "vllm_request_queue_time_seconds_count": {
            "vllm:request_queue_time_seconds_count",
            "vllm_request_queue_time_seconds_count",
        },
    }

    def __init__(
        self,
        config_name: str,
        sample_interval_sec: float = DEFAULT_SAMPLE_INTERVAL_SEC,
        nvidia_dmon_dir: str = "/tmp",
        nvidia_dmon_path: str | Path | None = None,
    ) -> None:
        self.config_name = config_name
        self.interval = sample_interval_sec
        self.nvidia_dmon_dir = nvidia_dmon_dir
        self.nvidia_dmon_path = Path(nvidia_dmon_path) if nvidia_dmon_path else None

        self._start_time: float | None = None
        self._end_time: float | None = None
        self._samples: list[dict[str, Any]] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._nvidia_proc: subprocess.Popen | None = None
        self._nvidia_log: str | None = None
        self._io_start: Any = None   # psutil io counters at start
        self._io_scope: str = "unavailable"
        self._pid = os.getpid()
        self._vllm_metrics_url = self._build_vllm_metrics_url()

    # ── Public API ─────────────────────────────────────────────────────────
    def start(self) -> None:
        import psutil

        self._start_time = time.perf_counter()
        self._running = True
        proc = psutil.Process(self._pid)
        self._io_start, self._io_scope = self._get_initial_io_counters(psutil, proc)
        self._launch_nvidia_dmon()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()


    def stop(self) -> dict[str, Any]:
        self._running = False
        self._end_time = time.perf_counter()
        if self._thread:
            self._thread.join(timeout=5)
        if self._nvidia_proc:
            try:
                self._nvidia_proc.terminate()
                self._nvidia_proc.wait(timeout=2)
            except Exception:
                pass
        return self.report()

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        data = self.report()
        if extra:
            data.update(extra)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(data, fh, indent=2)

    # ── Internals ──────────────────────────────────────────────────────────
    def _launch_nvidia_dmon(self) -> None:
        if self.nvidia_dmon_path is not None:
            self.nvidia_dmon_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = str(self.nvidia_dmon_path)
        else:
            log_path = os.path.join(
                self.nvidia_dmon_dir, f"nvidia_dmon_{self.config_name}.log"
            )
        self._nvidia_log = log_path
        try:
            # -s pucm = pstate, util, power, cumulative mem; -d = seconds
            self._nvidia_proc = subprocess.Popen(
                [
                    "nvidia-smi", "dmon",
                    "-s", "pucm",
                    "-d", str(max(1, int(self.interval))),
                    "-f", log_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self._nvidia_proc = None   # no GPU on this machine


    def _sample_gpu(self) -> dict[str, float]:
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,"
                    "memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode != 0:
                return {}
            line = r.stdout.strip().splitlines()[0]
            u, mu, mem_used, mem_total, power = [
                float(x.strip()) if x.strip() != "[N/A]" else 0.0
                for x in line.split(",")
            ]
            return {
                "gpu_util_pct": u,
                "gpu_mem_util_pct": mu,
                "gpu_mem_used_mb": mem_used,
                "gpu_mem_total_mb": mem_total,
                "gpu_power_w": power,
            }
        except Exception:
            return {}

    def _build_vllm_metrics_url(self) -> str | None:
        base_url = os.environ.get("VLLM_BASE_URL", "").strip()
        if not base_url:
            return None
        return f"{base_url.rstrip('/')}/metrics"

    @classmethod
    def _parse_prometheus_metrics(cls, text: str) -> dict[str, float]:
        values_by_name: dict[str, list[float]] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            metric_name = parts[0].split("{", 1)[0]
            try:
                value = float(parts[1])
            except ValueError:
                continue
            values_by_name.setdefault(metric_name, []).append(value)

        parsed: dict[str, float] = {}
        for output_key, aliases in cls.VLLM_METRIC_ALIASES.items():
            values = [
                value
                for metric_name in aliases
                for value in values_by_name.get(metric_name, [])
            ]
            if not values:
                continue
            if output_key in {
                "vllm_gpu_cache_usage_pct",
                "vllm_requests_running",
                "vllm_requests_waiting",
            }:
                parsed[output_key] = round(max(values), 6)
            else:
                parsed[output_key] = round(sum(values), 6)
        return parsed

    def _sample_vllm_metrics(self) -> dict[str, float]:
        if not self._vllm_metrics_url:
            return {}
        try:
            with urllib.request.urlopen(self._vllm_metrics_url, timeout=0.5) as response:
                text = response.read().decode("utf-8", errors="replace")
            return self._parse_prometheus_metrics(text)
        except Exception:
            return {}

    def _get_initial_io_counters(self, psutil: Any, proc: Any) -> tuple[Any, str]:
        try:
            return proc.io_counters(), "process"
        except Exception:
            pass
        try:
            return psutil.disk_io_counters(), "system"
        except Exception:
            return None, "unavailable"

    def _sample_io(self, psutil: Any, proc: Any) -> dict[str, int]:
        if self._io_scope == "process":
            io = proc.io_counters()
        elif self._io_scope == "system":
            io = psutil.disk_io_counters()
        else:
            return {}
        return {
            "read_bytes": int(getattr(io, "read_bytes", 0) or 0),
            "write_bytes": int(getattr(io, "write_bytes", 0) or 0),
        }

    def _sample_process_tree(self, proc: Any) -> dict[str, Any]:
        processes = [proc]
        try:
            processes.extend(proc.children(recursive=True))
        except Exception:
            pass

        cpu_total = 0.0
        rss_total = 0
        child_count = max(0, len(processes) - 1)
        for process in processes:
            try:
                cpu_total += float(process.cpu_percent(interval=None) or 0.0)
                rss_total += int(process.memory_info().rss or 0)
            except Exception:
                continue

        return {
            "process_tree_cpu_pct": round(cpu_total, 2),
            "process_tree_rss_mb": round(rss_total / 1e6, 2),
            "child_processes": child_count,
        }


    def _sample_loop(self) -> None:
        import psutil
        proc = psutil.Process(self._pid)

        while self._running:
            t = time.perf_counter() - (self._start_time or 0.0)
            sample: dict[str, Any] = {"t_sec": round(t, 3)}

            try:
                # Note: interval=None to avoid blocking the sampler
                sample["cpu_pct"] = proc.cpu_percent(interval=None)
                mem = proc.memory_info()
                sample["rss_mb"] = round(mem.rss / 1e6, 2)
                vm = psutil.virtual_memory()
                sample["host_ram_used_gb"] = round(vm.used / 1e9, 2)
                sample["host_ram_pct"] = vm.percent
            except Exception:
                pass

            try:
                sample.update(self._sample_process_tree(proc))
            except Exception:
                pass

            try:
                sample.update(self._sample_io(psutil, proc))
            except Exception:
                pass

            sample.update(self._sample_gpu())
            sample.update(self._sample_vllm_metrics())
            self._samples.append(sample)
            time.sleep(self.interval)


    # ── Aggregation ────────────────────────────────────────────────────────
    def report(self) -> dict[str, Any]:
        total_wall = (
            (self._end_time or time.perf_counter()) - (self._start_time or 0.0)
        )

        def avg(key: str) -> float:
            vs = [s.get(key) for s in self._samples if s.get(key) is not None]
            return round(sum(vs) / len(vs), 2) if vs else 0.0

        def peak(key: str) -> float:
            vs = [s.get(key) for s in self._samples if s.get(key) is not None]
            return round(max(vs), 2) if vs else 0.0

        # Pipeline continuity — fraction of samples where GPU util > threshold
        gpu_utils = [
            s.get("gpu_util_pct", 0.0) for s in self._samples
            if "gpu_util_pct" in s
        ]
        gpu_telemetry_available = bool(gpu_utils)
        if gpu_utils:
            gpu_active = sum(1 for u in gpu_utils
                             if u >= self.GPU_UTIL_ACTIVE_THRESHOLD)
            pipeline_continuity = round(gpu_active / len(gpu_utils), 3)
        else:
            pipeline_continuity = None
        vllm_telemetry_available = any(
            any(str(key).startswith("vllm_") for key in sample)
            for sample in self._samples
        )

        def vllm_value(key: str, fn) -> float | None:
            if not vllm_telemetry_available:
                return None
            return fn(key)

        # Cumulative disk IO deltas
        if self._io_start is not None and self._samples:
            last = self._samples[-1]
            read_delta = max(0, last.get("read_bytes", 0)
                             - getattr(self._io_start, "read_bytes", 0))
            write_delta = max(0, last.get("write_bytes", 0)
                              - getattr(self._io_start, "write_bytes", 0))
        else:
            read_delta = write_delta = 0


        return {
            "config": self.config_name,
            "wall_clock_sec": round(total_wall, 3),
            "n_samples": len(self._samples),
            "sample_interval_sec": self.interval,
            # CPU/RAM
            "avg_cpu_pct": avg("cpu_pct"),
            "peak_rss_mb": peak("rss_mb"),
            "avg_rss_mb": avg("rss_mb"),
            "avg_process_tree_cpu_pct": avg("process_tree_cpu_pct"),
            "peak_process_tree_rss_mb": peak("process_tree_rss_mb"),
            "avg_process_tree_rss_mb": avg("process_tree_rss_mb"),
            "sampled_child_processes": int(peak("child_processes")),
            "peak_host_ram_gb": peak("host_ram_used_gb"),
            # GPU
            "gpu_telemetry_available": gpu_telemetry_available,
            "avg_gpu_util_pct": avg("gpu_util_pct"),
            "peak_gpu_util_pct": peak("gpu_util_pct"),
            "avg_gpu_mem_util_pct": avg("gpu_mem_util_pct"),
            "peak_gpu_mem_used_mb": peak("gpu_mem_used_mb"),
            "avg_gpu_power_w": avg("gpu_power_w"),
            "pipeline_continuity_available": gpu_telemetry_available,
            "pipeline_continuity": pipeline_continuity,
            # vLLM
            "vllm_telemetry_available": vllm_telemetry_available,
            "avg_vllm_gpu_cache_usage_pct": vllm_value("vllm_gpu_cache_usage_pct", avg),
            "peak_vllm_gpu_cache_usage_pct": vllm_value("vllm_gpu_cache_usage_pct", peak),
            "peak_vllm_requests_running": vllm_value("vllm_requests_running", peak),
            "peak_vllm_requests_waiting": vllm_value("vllm_requests_waiting", peak),
            # Disk
            "bytes_read_delta": int(read_delta),
            "bytes_written_delta": int(write_delta),
            "mb_written_delta": round(write_delta / 1e6, 2),
            "write_throughput_mb_s": (
                round((write_delta / 1e6) / total_wall, 2)
                if total_wall > 0 else 0.0
            ),
            "disk_counter_scope": self._io_scope,
            # Raw samples for plotting
            "samples": self._samples,
            "nvidia_dmon_log": self._nvidia_log or None,
        }
