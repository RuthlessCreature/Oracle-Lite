from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ALLOWED_KEYS = {"training_output_dir"}


@dataclass(slots=True, frozen=True)
class TalkerConfig:
    training_output_dir: Path
    talker_dir: Path

    @property
    def state_dir(self) -> Path:
        return self.talker_dir / "state"

    @property
    def history_db(self) -> Path:
        return self.state_dir / "history.sqlite3"

    @property
    def uploads_dir(self) -> Path:
        return self.state_dir / "uploads"

    @property
    def parsed_dir(self) -> Path:
        return self.state_dir / "parsed"

    def ensure_dirs(self) -> None:
        for path in (self.state_dir, self.uploads_dir, self.parsed_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> TalkerConfig:
    talker_dir = Path(__file__).resolve().parent
    config_path = Path(path).expanduser().resolve() if path else talker_dir / "talker.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Talker config not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            "talker.yaml accepts only training_output_dir. "
            f"Unknown keys: {', '.join(unknown)}"
        )
    value = str(raw.get("training_output_dir") or "").strip()
    if not value:
        raise ValueError("talker.yaml requires training_output_dir")

    training_output = Path(value).expanduser()
    if not training_output.is_absolute():
        training_output = (config_path.parent / training_output).resolve()
    else:
        training_output = training_output.resolve()

    cfg = TalkerConfig(
        training_output_dir=training_output,
        talker_dir=talker_dir,
    )
    cfg.ensure_dirs()
    return cfg
