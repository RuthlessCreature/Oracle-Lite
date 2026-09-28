from __future__ import annotations

from pathlib import Path


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path}")


def parse_text_like(path: Path) -> tuple[str, str, dict]:
    text = _read_text(path)
    title = path.stem
    metadata = {
        "extension": path.suffix.lower(),
        "bytes": path.stat().st_size,
    }
    return title, text, metadata
