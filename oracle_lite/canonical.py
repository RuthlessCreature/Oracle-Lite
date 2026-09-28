from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


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
    segments_path: str | None = None

    def to_dict(self, *, include_segments: bool = True) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "text": self.text,
            "title": self.title,
            "segments": [segment.to_dict() for segment in self.segments] if include_segments else [],
            "segments_path": self.segments_path,
            "metadata": self.metadata,
            "extracted_at": self.extracted_at,
        }

    @staticmethod
    def _sidecar_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".segments.jsonl")

    def write_json(self, path: str | Path) -> None:
        """Write small document metadata plus streamable segment JSONL sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sidecar = self._sidecar_path(path)
        sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        with sidecar_tmp.open("w", encoding="utf-8") as f:
            for segment in self.segments:
                json.dump(segment.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
                f.write("\n")
        sidecar_tmp.replace(sidecar)

        self.segments_path = str(sidecar.resolve())
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(include_segments=False),
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        tmp.replace(path)

    @classmethod
    def read_header(cls, path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def iter_segments(cls, path: str | Path) -> Iterator[CanonicalSegment]:
        raw = cls.read_header(path)
        sidecar = raw.get("segments_path")
        if sidecar:
            sidecar_path = Path(sidecar)
            if not sidecar_path.exists():
                raise FileNotFoundError(sidecar_path)
            with sidecar_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    yield CanonicalSegment(**json.loads(line))
            return

        # Backward compatibility with V0.2/V0.4 inline canonical artifacts.
        for item in raw.get("segments", []):
            yield CanonicalSegment(**item)

    @classmethod
    def read_json(cls, path: str | Path) -> "CanonicalDocument":
        raw = cls.read_header(path)
        raw["segments"] = list(cls.iter_segments(path))
        return cls(**raw)
