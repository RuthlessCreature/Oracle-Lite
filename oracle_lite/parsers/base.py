from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParsedSegment:
    text: str
    images: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    title: str
    text: str
    segments: list[ParsedSegment]
    metadata: dict[str, Any] = field(default_factory=dict)
