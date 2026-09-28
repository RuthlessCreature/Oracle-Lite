from __future__ import annotations

from pathlib import Path

from .base import ParsedDocument, ParsedSegment


CHUNK_CHARS = 256_000
ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "latin-1")


def _detect_encoding(path: Path) -> str:
    with path.open("rb") as f:
        sample = f.read(1024 * 1024)
    for encoding in ENCODINGS:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def parse_text_like(path: Path) -> ParsedDocument:
    encoding = _detect_encoding(path)
    segments: list[ParsedSegment] = []
    with path.open("r", encoding=encoding) as f:
        index = 0
        while True:
            chunk = f.read(CHUNK_CHARS)
            if not chunk:
                break
            segments.append(
                ParsedSegment(
                    text=chunk,
                    metadata={"chunk_index": index},
                )
            )
            index += 1

    return ParsedDocument(
        title=path.stem,
        text="",
        segments=segments,
        metadata={
            "extension": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "encoding": encoding,
            "chunks": len(segments),
        },
    )
