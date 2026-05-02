from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def test_run_mode_configs_are_parseable() -> None:
    for path in [
        Path("config/mock.yaml"),
        Path("config/cpu.yaml"),
        Path("config/gpu_h200.yaml"),
        Path("config/gpu_v100_smoke.yaml"),
    ]:
        cfg = yaml.safe_load(path.read_text())
        assert cfg["profile"]
        assert cfg["models"]["generator"]["name"]
        assert cfg["execution"]["configs"]
        assert cfg["runner"]["iterations"] >= 1


def test_benchmark_shell_entrypoints_are_valid() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            "scripts/setup_env.sh",
            "scripts/run_benchmark.sh",
            "scripts/run_all_laptop.sh",
            "scripts/run_all_gpu.sh",
            "scripts/slurm_benchmark_all.sh",
            "scripts/start_vllm_server.sh",
        ],
        check=True,
    )
