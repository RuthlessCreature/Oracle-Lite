from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CanonicalSegment:
    text: str
    images: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def read_json(cls, path: str | Path) -> "CanonicalDocument":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["segments"] = [CanonicalSegment(**item) for item in raw.get("segments", [])]
        return cls(**raw)
