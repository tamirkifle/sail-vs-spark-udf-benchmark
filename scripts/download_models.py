import argparse
import os
import yaml
from pathlib import Path


def download_models(configs: list[str], models_dir: str | None = None) -> None:
    # We'll try to import these only when needed
    from huggingface_hub import snapshot_download
    
    repo_dir = Path(__file__).parent.parent
    root = Path(models_dir or os.environ.get("MODELS_DIR", repo_dir / "models"))
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(root))
    
    models_to_download = set()
    
    # ... (config parsing logic stays the same) ...
    for cfg_path in configs:
        path = repo_dir / cfg_path
        if not path.exists():
            continue
            
        with open(path) as f:
            cfg = yaml.safe_load(f)
            
        models = cfg.get("models", {})
        for m_type in ["generator", "scorer", "embedder"]:
            m_cfg = models.get(m_type, {})
            if isinstance(m_cfg, dict):
                if m_cfg.get("prefer_mock", False):
                    continue
                if m_type == "generator":
                    # Real generation is served by vLLM. The server script can
                    # point at an already-downloaded local models/<repo> folder.
                    continue
                if "name" in m_cfg:
                    models_to_download.add(m_cfg["name"])
                if "fallback_name" in m_cfg:
                    models_to_download.add(m_cfg["fallback_name"])
            elif isinstance(m_cfg, str):
                models_to_download.add(m_cfg)

    print(f"Downloading {len(models_to_download)} models to {root}...")
    for model_id in models_to_download:
        # Create a clean folder name (e.g., Qwen/Qwen2.5-0.5B -> Qwen--Qwen2.5-0.5B)
        target_dir = root / model_id.replace("/", "--")
        print(f"  - Downloading {model_id} to {target_dir}...")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(target_dir),
                max_workers=1,
                ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*', '*.pt']
            )
        except Exception as e:
            print(f"    Failed to download {model_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=["config/cpu.yaml"],
        help="YAML config to inspect. May be passed more than once.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Local model directory. Defaults to MODELS_DIR or ./models.",
    )
    args = parser.parse_args()
    download_models(args.config, models_dir=args.models_dir)
