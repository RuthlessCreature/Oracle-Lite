from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .base import ParsedDocument, ParsedSegment


CHUNK_CHARS = 256_000


def _iter_flatten_json(value: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_flatten_json(item, next_prefix)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            next_prefix = f"{prefix}[{i}]"
            yield from _iter_flatten_json(item, next_prefix)
    else:
        yield f"{prefix}: {value}"


def _segments_from_lines(lines: Iterator[str], *, metadata: dict[str, Any] | None = None) -> list[ParsedSegment]:
    segments: list[ParsedSegment] = []
    chunk: list[str] = []
    chars = 0
    chunk_index = 0
    for line in lines:
        chunk.append(line)
        chars += len(line) + 1
        if chars >= CHUNK_CHARS:
            meta = dict(metadata or {})
            meta["chunk_index"] = chunk_index
            segments.append(ParsedSegment(text="\n".join(chunk), metadata=meta))
            chunk_index += 1
            chunk = []
            chars = 0
    if chunk:
        meta = dict(metadata or {})
        meta["chunk_index"] = chunk_index
        segments.append(ParsedSegment(text="\n".join(chunk), metadata=meta))
    return segments


def parse_json_like(path: Path) -> ParsedDocument:
    ext = path.suffix.lower()
    title = path.stem

    if ext == ".jsonl":
        segments: list[ParsedSegment] = []
        invalid = 0
        records = 0
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                record_segments = _segments_from_lines(
                    _iter_flatten_json(record),
                    metadata={"record_index": records},
                )
                segments.extend(record_segments)
                records += 1
                del record

        return ParsedDocument(
            title=title,
            text="",
            segments=segments,
            metadata={
                "extension": ext,
                "records": records,
                "invalid_lines": invalid,
            },
        )

    # json.load avoids a second whole-file Unicode string created by read_text().
    with path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    root_type = type(raw).__name__
    segments = _segments_from_lines(_iter_flatten_json(raw))
    del raw
    return ParsedDocument(
        title=title,
        text="",
        segments=segments,
        metadata={"extension": ext, "root_type": root_type},
    )
