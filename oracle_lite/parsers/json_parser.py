from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import ParsedDocument, ParsedSegment


def _flatten_json(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_json(item, next_prefix))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            next_prefix = f"{prefix}[{i}]"
            lines.extend(_flatten_json(item, next_prefix))
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def parse_json_like(path: Path) -> ParsedDocument:
    ext = path.suffix.lower()
    title = path.stem

    if ext == ".jsonl":
        records = []
        invalid = 0
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                invalid += 1
        blocks = ["\n".join(_flatten_json(record)) for record in records]
        text = "\n\n--- record ---\n\n".join(blocks)
        return ParsedDocument(
            title=title,
            text=text,
            segments=[ParsedSegment(text=block, metadata={"record_index": i}) for i, block in enumerate(blocks)],
            metadata={"extension": ext, "records": len(records), "invalid_lines": invalid},
        )

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    text = "\n".join(_flatten_json(raw))
    return ParsedDocument(
        title=title,
        text=text,
        segments=[ParsedSegment(text=text)],
        metadata={"extension": ext, "root_type": type(raw).__name__},
    )
