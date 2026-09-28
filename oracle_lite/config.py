from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXTENSIONS = {
    ".txt", ".md", ".json", ".jsonl", ".csv",
    ".pdf", ".docx", ".pptx",
}


@dataclass(slots=True)
class AppConfig:
    corpus_roots: list[Path] = field(default_factory=lambda: [Path("corpus")])
    state_dir: Path = Path(".oracle")
    parser_version: str = "v1"
    include_extensions: set[str] = field(default_factory=lambda: set(DEFAULT_EXTENSIONS))
    ignore_names: set[str] = field(default_factory=lambda: {".git", ".oracle", "__pycache__"})
    hash_algorithm: str = "sha256"

    @property
    def registry_path(self) -> Path:
        return self.state_dir / "registry.sqlite3"

    @property
    def canonical_dir(self) -> Path:
        return self.state_dir / "canonical"

    @property
    def snapshots_dir(self) -> Path:
        return self.state_dir / "snapshots"

    @property
    def datasets_dir(self) -> Path:
        return self.state_dir / "datasets"

    def ensure_dirs(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)


def _as_path_list(values: list[str] | None) -> list[Path]:
    return [Path(v).expanduser() for v in values] if values else [Path("corpus")]


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        cfg = AppConfig()
        cfg.ensure_dirs()
        return cfg

    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = AppConfig(
        corpus_roots=_as_path_list(raw.get("corpus_roots")),
        state_dir=Path(raw.get("state_dir", ".oracle")).expanduser(),
        parser_version=str(raw.get("parser_version", "v1")),
        include_extensions={str(x).lower() for x in raw.get("include_extensions", DEFAULT_EXTENSIONS)},
        ignore_names=set(raw.get("ignore_names", [".git", ".oracle", "__pycache__"])),
        hash_algorithm=str(raw.get("hash_algorithm", "sha256")),
    )
    cfg.ensure_dirs()
    return cfg
