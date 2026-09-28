from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .canonical import CanonicalDocument
from .config import AppConfig
from .db import Registry
from .normalize import normalize_text
from .parsers import parse_file


@dataclass(slots=True)
class IngestStats:
    ready: int = 0
    skipped: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"ready": self.ready, "skipped": self.skipped, "failed": self.failed}


def ingest_corpus(cfg: AppConfig, *, force: bool = False) -> IngestStats:
    registry = Registry(cfg.registry_path)
    stats = IngestStats()

    for row in registry.list_active_unique_content():
        content_hash = row["content_hash"]
        source_path = Path(row["source_path"])
        existing = registry.get_artifact(content_hash, cfg.parser_version)

        if existing is not None and existing["status"] == "ready" and not force:
            canonical_path = existing["canonical_path"]
            if canonical_path and Path(canonical_path).exists():
                stats.skipped += 1
                continue

        try:
            title, text, metadata = parse_file(source_path)
            text = normalize_text(text)
            if not text:
                raise ValueError("Parser produced empty text")

            canonical_path = (
                cfg.canonical_dir
                / content_hash[:2]
                / f"{content_hash}.{cfg.parser_version}.json"
            ).resolve()

            doc = CanonicalDocument(
                content_hash=content_hash,
                source_path=str(source_path.resolve()),
                source_type=source_path.suffix.lower().lstrip("."),
                parser_name=f"oracle-lite:{source_path.suffix.lower()}",
                parser_version=cfg.parser_version,
                text=text,
                title=title,
                metadata=metadata,
            )
            doc.write_json(canonical_path)
            registry.save_artifact(
                content_hash=content_hash,
                parser_version=cfg.parser_version,
                canonical_path=str(canonical_path),
                status="ready",
            )
            stats.ready += 1
        except Exception as exc:  # parser failures are recorded, not swallowed
            registry.save_artifact(
                content_hash=content_hash,
                parser_version=cfg.parser_version,
                canonical_path=None,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            stats.failed += 1

    return stats
