import os
import yaml
from pathlib import Path

def download_models():
    # We'll try to import these only when needed
    from huggingface_hub import snapshot_download
    
    configs = ["config/laptop.yaml"]
    repo_dir = Path(__file__).parent.parent
    models_dir = repo_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
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
                if "name" in m_cfg:
                    models_to_download.add(m_cfg["name"])
                if "fallback_name" in m_cfg:
                    models_to_download.add(m_cfg["fallback_name"])
            elif isinstance(m_cfg, str):
                models_to_download.add(m_cfg)

    print(f"Downloading {len(models_to_download)} models to {models_dir}...")
    for model_id in models_to_download:
        # Create a clean folder name (e.g., Qwen/Qwen2.5-0.5B -> Qwen--Qwen2.5-0.5B)
        target_dir = models_dir / model_id.replace("/", "--")
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
    download_models()
