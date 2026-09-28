from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import AppConfig


DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3.5-9B-Base"
DEFAULT_BASE_MODEL_DIRNAME = "Qwen3.5-9B-Base"


def ensure_base_model(cfg: AppConfig) -> Path:
    """Return a local Qwen3.5 multimodal Base checkout, downloading once if needed."""
    target = (cfg.models_dir / DEFAULT_BASE_MODEL_DIRNAME).resolve()
    config_path = target / "config.json"

    if config_path.exists():
        return target

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DEFAULT_BASE_MODEL_ID,
        local_dir=str(target),
    )

    if not config_path.exists():
        raise RuntimeError(
            f"Model download completed without config.json under {target}"
        )

    metadata = {
        "repo_id": DEFAULT_BASE_MODEL_ID,
        "local_path": str(target),
    }
    (target / "oracle-model.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
