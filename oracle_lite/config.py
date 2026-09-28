from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXTENSIONS = {
    ".txt", ".md", ".json", ".jsonl", ".csv",
    ".pdf", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
    ".stp", ".step", ".x_t", ".x_b",
}
DEFAULT_IGNORE_NAMES = {".git", ".oracle", "__pycache__"}
ALLOWED_CONFIG_KEYS = {"minimax_api_key", "corpus_dir", "output_dir"}


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(slots=True)
class AppConfig:
    minimax_api_key: str
    corpus_dir: Path
    output_dir: Path

    # Internal defaults: intentionally not user-facing configuration.
    parser_version: str = field(default="v5-cad-corpus-isolation", init=False)
    hash_algorithm: str = field(default="sha256", init=False)
    include_extensions: set[str] = field(default_factory=lambda: set(DEFAULT_EXTENSIONS), init=False)
    ignore_names: set[str] = field(default_factory=lambda: set(DEFAULT_IGNORE_NAMES), init=False)

    @property
    def corpus_roots(self) -> list[Path]:
        return [self.corpus_dir]

    @property
    def corpus_id(self) -> str:
        """Stable namespace derived only from the canonical corpus root path."""
        canonical = str(self.corpus_dir.expanduser().resolve())
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"corpus-{digest}"

    @property
    def corpus_output_dir(self) -> Path:
        return self.output_dir / "_corpora" / self.corpus_id

    @property
    def state_dir(self) -> Path:
        return self.corpus_output_dir / "_state"

    @property
    def registry_path(self) -> Path:
        return self.state_dir / "registry.sqlite3"

    @property
    def canonical_dir(self) -> Path:
        return self.corpus_output_dir / "canonical"

    @property
    def assets_dir(self) -> Path:
        return self.corpus_output_dir / "assets"

    @property
    def snapshots_dir(self) -> Path:
        return self.corpus_output_dir / "snapshots"

    @property
    def datasets_dir(self) -> Path:
        return self.corpus_output_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def training_dir(self) -> Path:
        return self.corpus_output_dir / "training"

    @property
    def logs_dir(self) -> Path:
        return self.corpus_output_dir / "logs"

    def _migrate_legacy_single_corpus_state(self) -> None:
        """Reuse v0.4 registry only when every recorded root is this corpus."""
        legacy_registry = self.output_dir / "_state" / "registry.sqlite3"
        if self.registry_path.exists() or not legacy_registry.exists():
            return
        try:
            conn = sqlite3.connect(legacy_registry)
            try:
                roots = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT DISTINCT root FROM source_files WHERE root IS NOT NULL"
                    )
                }
            finally:
                conn.close()
        except Exception:
            return
        expected = str(self.corpus_dir.resolve())
        if roots != {expected}:
            return

        # Move the entire legacy generated state under this corpus namespace.
        # Models remain shared at output_dir/models and are intentionally untouched.
        for name in ("_state", "canonical", "assets", "snapshots", "datasets", "training", "logs"):
            source = self.output_dir / name
            target = self.corpus_output_dir / name
            if not source.exists() or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))

    def ensure_dirs(self) -> None:
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_single_corpus_state()
        for path in (
            self.output_dir,
            self.corpus_output_dir,
            self.state_dir,
            self.canonical_dir,
            self.assets_dir,
            self.snapshots_dir,
            self.datasets_dir,
            self.models_dir,
            self.training_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path = "oracle.yaml") -> AppConfig:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. Copy configs/oracle.example.yaml to oracle.yaml."
        )

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            "Oracle-Lite config intentionally accepts only "
            "minimax_api_key, corpus_dir and output_dir. "
            f"Unknown keys: {', '.join(unknown)}"
        )

    missing = [key for key in ("minimax_api_key", "corpus_dir", "output_dir") if not raw.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    base_dir = path.parent
    cfg = AppConfig(
        minimax_api_key=str(raw["minimax_api_key"]).strip(),
        corpus_dir=_resolve_path(str(raw["corpus_dir"]), base_dir),
        output_dir=_resolve_path(str(raw["output_dir"]), base_dir),
    )

    if cfg.output_dir == cfg.corpus_dir or cfg.output_dir.is_relative_to(cfg.corpus_dir):
        raise ValueError(
            "output_dir must not be the same as, or inside, corpus_dir; "
            "otherwise generated artifacts can be re-ingested as corpus."
        )

    cfg.ensure_dirs()
    return cfg
