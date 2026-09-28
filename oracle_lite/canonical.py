from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CanonicalSegment:
    text: str
    images: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "images": self.images,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CanonicalDocument:
    content_hash: str
    source_path: str
    source_type: str
    parser_name: str
    parser_version: str
    text: str
    title: str | None = None
    segments: list[CanonicalSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extracted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        # Avoid dataclasses.asdict(): it recursively deep-copies large text and
        # segment structures before JSON serialization.
        return {
            "content_hash": self.content_hash,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "text": self.text,
            "title": self.title,
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": self.metadata,
            "extracted_at": self.extracted_at,
        }

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # json.dump streams encoder chunks directly to disk instead of first
        # constructing one giant JSON string in memory.
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
        tmp.replace(path)

    @classmethod
    def read_json(cls, path: str | Path) -> "CanonicalDocument":
        # json.load avoids an extra whole-file string created by read_text().
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["segments"] = [CanonicalSegment(**item) for item in raw.get("segments", [])]
        return cls(**raw)
