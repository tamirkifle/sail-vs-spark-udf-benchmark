from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def test_run_mode_configs_are_parseable() -> None:
    for path in [
        Path("config/mock.yaml"),
        Path("config/cpu.yaml"),
        Path("config/cpu_real.yaml"),
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


def test_cpu_real_mode_is_wired_to_config() -> None:
    runner = Path("scripts/run_benchmark.sh").read_text()
    setup = Path("scripts/setup_env.sh").read_text()

    assert "mock|cpu|cpu_real|gpu" in runner
    assert 'cpu_real) DEFAULT_CONFIG="config/cpu_real.yaml"' in runner
    assert 'START_VLLM="${START_VLLM:-$([[ "$MODE" == "gpu" ]] && echo 1 || echo 0)}"' in runner
    assert "mock|cpu|cpu_real|gpu|dev" in setup


def test_cpu_real_config_uses_real_models_and_transformers_generator() -> None:
    cfg = yaml.safe_load(Path("config/cpu_real.yaml").read_text())

    assert cfg["profile"] == "cpu_real"
    assert "vllm" not in cfg
    assert cfg["models"]["generator"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg["models"]["generator"]["provider"] == "transformers"
    assert "server_url" not in cfg["models"]["generator"]
    assert cfg["models"]["embedder"]["name"] == "sentence-transformers/all-MiniLM-L6-v2"
    for model_cfg in cfg["models"].values():
        assert model_cfg["prefer_mock"] is False
        assert model_cfg["allow_mock"] is False


def test_cpu_real_setup_avoids_vllm_install_path() -> None:
    setup = Path("scripts/setup_env.sh").read_text()
    cpu_real_block = setup.split('if [[ "$MODE" == "cpu_real" ]]; then', 1)[1]
    cpu_real_block = cpu_real_block.split('elif [[ "$MODE" == "cpu" || "$MODE" == "gpu" ]]; then', 1)[0]

    assert '"transformers>=4.51.0"' in cpu_real_block
    assert '"sentence-transformers>=3.0.0"' in cpu_real_block
    assert '"accelerate>=0.26.0"' in cpu_real_block
    assert "vllm" not in cpu_real_block.lower()


def test_cpu_vllm_startup_omits_gpu_only_args() -> None:
    script = Path("scripts/start_vllm_server.sh").read_text()
    cpu_branch = script.split('if [[ "$VLLM_DEVICE" == "cpu" ]]; then', 1)[1]
    cpu_branch = cpu_branch.split('"$PY" -m vllm.entrypoints.openai.api_server', 1)[0]

    assert "--device cpu" in cpu_branch
    assert "--dtype" in cpu_branch
    assert "VLLM_CPU_KVCACHE_SPACE" in cpu_branch
    assert "VLLM_CPU_NUM_OF_RESERVED_CPU" in cpu_branch
    assert "--gpu-memory-utilization" not in cpu_branch
    assert "--quantization" not in cpu_branch
    assert "VLLM_QUANTIZATION" not in cpu_branch
