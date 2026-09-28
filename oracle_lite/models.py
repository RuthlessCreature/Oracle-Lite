from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .config import AppConfig
from .resources import HostResourcePolicy, wait_for_disk


DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3.5-9B-Base"
DEFAULT_BASE_MODEL_DIRNAME = "Qwen3.5-9B-Base"


def ensure_base_model(cfg: AppConfig, *, monitor: Any | None = None) -> Path:
    """Return local Qwen3.5 Base, waiting for safe disk capacity before download."""
    target = (cfg.models_dir / DEFAULT_BASE_MODEL_DIRNAME).resolve()
    config_path = target / "config.json"

    if config_path.exists():
        return target

    policy = HostResourcePolicy.auto(cfg.output_dir)

    def on_wait(state: dict) -> None:
        if monitor is None:
            return
        monitor.update_phase(
            "waiting_for_disk",
            "Waiting for disk space before model download",
            status="paused",
        )
        monitor.update("system", {
            "disk_free_gb": state.get("free_gb"),
            "disk_reserve_gb": state.get("reserve_gb"),
            "disk_target_gb": state.get("target_gb"),
        })

    wait_for_disk(
        cfg.output_dir,
        policy,
        required_bytes=policy.model_download_budget_bytes,
        on_wait=on_wait,
        poll_seconds=3.0,
        stable_samples=2,
    )
    if monitor is not None:
        monitor.update_phase(
            "model_download",
            "Downloading Qwen3.5-9B-Base",
            status="running",
        )

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
