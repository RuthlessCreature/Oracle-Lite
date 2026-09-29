from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class AdapterBundle:
    adapter_dir: Path
    run_dir: Path
    base_model_path: Path
    run_metadata: dict
    source_kind: str


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_dir_for_adapter(adapter_dir: Path) -> Path:
    if adapter_dir.name == "adapter-final":
        return adapter_dir.parent
    if adapter_dir.name.startswith("checkpoint-"):
        return adapter_dir.parent
    return adapter_dir.parent


def _resolve_base_model(adapter_dir: Path, run_dir: Path, run_metadata: dict) -> Path:
    configured = run_metadata.get("base_model_path")
    if configured:
        path = Path(str(configured)).expanduser().resolve()
        if (path / "config.json").exists():
            return path

    adapter_cfg = _read_json(adapter_dir / "adapter_config.json")
    base = adapter_cfg.get("base_model_name_or_path")
    if base:
        path = Path(str(base)).expanduser()
        if path.exists() and (path.resolve() / "config.json").exists():
            return path.resolve()

    for parent in [run_dir, *run_dir.parents]:
        candidate = parent / "models" / "Qwen3.5-9B-Base"
        if (candidate / "config.json").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve the local base model for adapter "
        f"{adapter_dir}. Expected run.json base_model_path or a local "
        "Qwen3.5-9B-Base model directory."
    )


def discover_latest_adapter(training_output_dir: str | Path) -> AdapterBundle:
    root = Path(training_output_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"training_output_dir does not exist: {root}")

    candidates: list[Path] = []
    if (root / "adapter_config.json").exists():
        candidates.append(root)
    if (root / "adapter-final" / "adapter_config.json").exists():
        candidates.append(root / "adapter-final")

    candidates.extend(
        p.parent for p in root.rglob("adapter-final/adapter_config.json")
    )
    candidates.extend(
        p.parent
        for p in root.rglob("checkpoint-*/adapter_config.json")
    )

    unique: dict[str, Path] = {str(p.resolve()): p.resolve() for p in candidates}
    if not unique:
        raise FileNotFoundError(
            "No LoRA adapter found under training_output_dir. "
            "Expected adapter-final/adapter_config.json or checkpoint-*/adapter_config.json."
        )

    def score(path: Path) -> tuple[float, int]:
        try:
            mtime = (path / "adapter_config.json").stat().st_mtime
        except OSError:
            mtime = 0.0
        final_bonus = 1 if path.name == "adapter-final" else 0
        return (mtime, final_bonus)

    adapter_dir = max(unique.values(), key=score)
    run_dir = _run_dir_for_adapter(adapter_dir)
    run_metadata = _read_json(run_dir / "run.json")
    base_model = _resolve_base_model(adapter_dir, run_dir, run_metadata)
    return AdapterBundle(
        adapter_dir=adapter_dir,
        run_dir=run_dir,
        base_model_path=base_model,
        run_metadata=run_metadata,
        source_kind="final" if adapter_dir.name == "adapter-final" else "checkpoint",
    )
