from __future__ import annotations

import gc
from pathlib import Path
from queue import Queue

from .canonical import CanonicalDocument, CanonicalSegment
from .memory import apply_linux_address_space_limit
from .normalize import normalize_text
from .parsers import parse_file


def parse_and_write_canonical(
    *,
    source_path: str,
    asset_dir: str,
    canonical_path: str,
    content_hash: str,
    parser_version: str,
    result_queue,
    address_space_limit_bytes: int,
) -> None:
    """Parse exactly one source file inside an isolated worker process."""
    hard_limit_applied = apply_linux_address_space_limit(address_space_limit_bytes)
    try:
        source = Path(source_path)
        parsed = parse_file(source, asset_dir=Path(asset_dir))

        # Segment-aware parsers already preserve the useful text in segments.
        # Keeping an additional full-document concatenation doubles memory and
        # disk for PDF/PPTX/DOCX/JSONL without helping the current dataset path.
        document_text = ""
        if not parsed.segments:
            document_text = normalize_text(parsed.text)

        # Release aggregate parser text as early as possible when segments exist.
        if parsed.segments:
            parsed.text = ""

        segments: list[CanonicalSegment] = []
        visual_segments = 0
        has_text = False

        for segment in parsed.segments:
            segment_text = normalize_text(segment.text)
            images = [
                str(Path(p).resolve())
                for p in segment.images
                if Path(p).exists()
            ]
            if not segment_text and not images:
                continue
            if segment_text:
                has_text = True
            if images:
                visual_segments += 1
            segments.append(
                CanonicalSegment(
                    text=segment_text,
                    images=images,
                    metadata=segment.metadata,
                )
            )

        if not document_text and not has_text and not any(s.images for s in segments):
            raise ValueError("Parser produced neither text nor visual assets")

        doc = CanonicalDocument(
            content_hash=content_hash,
            source_path=str(source.resolve()),
            source_type=source.suffix.lower().lstrip("."),
            parser_name=f"oracle-lite:{source.suffix.lower()}",
            parser_version=parser_version,
            text=document_text,
            title=parsed.title,
            segments=segments,
            metadata=parsed.metadata,
        )
        doc.write_json(canonical_path)

        result_queue.put({
            "ok": True,
            "hard_limit_applied": hard_limit_applied,
            "visual_segments": visual_segments,
            "has_visual": bool(visual_segments),
        })
    except BaseException as exc:
        errno = getattr(exc, "errno", None)
        is_memory = isinstance(exc, MemoryError) or errno == 12
        result_queue.put({
            "ok": False,
            "kind": "memory" if is_memory else "parser",
            "hard_limit_applied": hard_limit_applied,
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        gc.collect()
