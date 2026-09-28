from __future__ import annotations

import gc
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from .memory import apply_linux_address_space_limit
from .normalize import normalize_text


RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
STREAM_CHARS = 256_000


def _write_header(
    *,
    canonical_path: Path,
    source: Path,
    content_hash: str,
    parser_version: str,
    metadata: dict,
    segments_path: Path,
    title: str | None = None,
) -> None:
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = canonical_path.with_suffix(canonical_path.suffix + ".tmp")
    payload = {
        "content_hash": content_hash,
        "source_path": str(source.resolve()),
        "source_type": source.suffix.lower().lstrip("."),
        "parser_name": f"oracle-lite:{source.suffix.lower()}:low-memory",
        "parser_version": parser_version,
        "text": "",
        "title": title or source.stem,
        "segments": [],
        "segments_path": str(segments_path.resolve()),
        "metadata": metadata,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(canonical_path)


def _append_segment(out, *, text: str, images: list[str] | None = None, metadata: dict | None = None) -> bool:
    text = normalize_text(text)
    images = images or []
    if not text and not images:
        return False
    json.dump(
        {"text": text, "images": images, "metadata": metadata or {}},
        out,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    out.write("\n")
    out.flush()
    return True


def _stream_text_source(source: Path, sidecar_tmp: Path) -> tuple[int, int]:
    encodings = ("utf-8", "utf-8-sig", "gb18030", "latin-1")
    with source.open("rb") as f:
        sample = f.read(1024 * 1024)
    encoding = "latin-1"
    for candidate in encodings:
        try:
            sample.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue

    segments = 0
    with source.open("r", encoding=encoding, errors="replace") as inp, sidecar_tmp.open("w", encoding="utf-8") as out:
        index = 0
        while True:
            chunk = inp.read(STREAM_CHARS)
            if not chunk:
                break
            if _append_segment(out, text=chunk, metadata={"kind": "raw_stream", "chunk_index": index}):
                segments += 1
            index += 1
    return segments, 0


def _iter_xml_text(stream):
    for _, elem in ET.iterparse(stream, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "t" and elem.text:
            yield elem.text
        elem.clear()


def _stream_office_source(source: Path, asset_dir: Path, sidecar_tmp: Path) -> tuple[int, int]:
    ext = source.suffix.lower()
    prefix = "word/" if ext == ".docx" else "ppt/"
    xml_names: list[str]
    with zipfile.ZipFile(source) as zf:
        if ext == ".docx":
            xml_names = ["word/document.xml"] if "word/document.xml" in zf.namelist() else []
        else:
            xml_names = sorted(
                n for n in zf.namelist()
                if n.startswith("ppt/slides/slide") and n.endswith(".xml")
            )

        asset_dir.mkdir(parents=True, exist_ok=True)
        media_names = sorted(
            n for n in zf.namelist()
            if n.startswith(prefix + "media/") and Path(n).suffix.lower() in RASTER_EXTENSIONS
        )
        copied_images: list[str] = []
        for idx, name in enumerate(media_names, start=1):
            suffix = Path(name).suffix.lower()
            target = asset_dir / f"media-{idx:05d}{suffix}"
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            copied_images.append(str(target.resolve()))

        segments = 0
        with sidecar_tmp.open("w", encoding="utf-8") as out:
            for xml_index, name in enumerate(xml_names):
                with zf.open(name) as stream:
                    chunk: list[str] = []
                    chars = 0
                    chunk_index = 0
                    for text in _iter_xml_text(stream):
                        chunk.append(text)
                        chars += len(text) + 1
                        if chars >= STREAM_CHARS:
                            images = copied_images if segments == 0 else []
                            if _append_segment(
                                out,
                                text="\n".join(chunk),
                                images=images,
                                metadata={
                                    "kind": "office_low_memory",
                                    "xml_index": xml_index,
                                    "chunk_index": chunk_index,
                                },
                            ):
                                segments += 1
                            chunk = []
                            chars = 0
                            chunk_index += 1
                    if chunk:
                        images = copied_images if segments == 0 else []
                        if _append_segment(
                            out,
                            text="\n".join(chunk),
                            images=images,
                            metadata={
                                "kind": "office_low_memory",
                                "xml_index": xml_index,
                                "chunk_index": chunk_index,
                            },
                        ):
                            segments += 1

            if segments == 0 and copied_images:
                if _append_segment(
                    out,
                    text="",
                    images=copied_images,
                    metadata={"kind": "office_low_memory_unlabeled_images"},
                ):
                    segments += 1

        return segments, len(copied_images)


def _stream_pdf_source(source: Path, asset_dir: Path, sidecar_tmp: Path) -> tuple[int, int]:
    import pymupdf as fitz

    asset_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(source)
    segments = 0
    visual_segments = 0
    try:
        with sidecar_tmp.open("w", encoding="utf-8") as out:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                text = page.get_text("text").strip()
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
                image_path = asset_dir / f"page-{page_index + 1:05d}.png"
                pix.save(str(image_path))
                if _append_segment(
                    out,
                    text=text,
                    images=[str(image_path.resolve())],
                    metadata={
                        "kind": "pdf_page_low_memory",
                        "page": page_index + 1,
                    },
                ):
                    segments += 1
                    visual_segments += 1
                del pix
                del page
                if (page_index + 1) % 4 == 0:
                    gc.collect()
    finally:
        doc.close()
    return segments, visual_segments


def _stream_low_memory_canonical(
    *,
    source: Path,
    asset_dir: Path,
    canonical_path: Path,
    content_hash: str,
    parser_version: str,
) -> dict:
    sidecar = canonical_path.with_suffix(canonical_path.suffix + ".segments.jsonl")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_tmp.parent.mkdir(parents=True, exist_ok=True)

    ext = source.suffix.lower()
    if ext in {".txt", ".md", ".csv", ".json", ".jsonl"}:
        segments, visual_segments = _stream_text_source(source, sidecar_tmp)
        metadata = {"extension": ext, "low_memory_mode": True, "raw_stream": True}
    elif ext == ".pdf":
        segments, visual_segments = _stream_pdf_source(source, asset_dir, sidecar_tmp)
        metadata = {"extension": ext, "low_memory_mode": True}
    elif ext in {".docx", ".pptx"}:
        segments, visual_segments = _stream_office_source(source, asset_dir, sidecar_tmp)
        metadata = {"extension": ext, "low_memory_mode": True}
    else:
        # Native image sources are already constant-memory in the normal parser.
        from .parsers import parse_file
        from .canonical import CanonicalDocument, CanonicalSegment

        parsed = parse_file(source, asset_dir=asset_dir)
        segments_list = []
        visual_segments = 0
        for segment in parsed.segments:
            images = [str(Path(p).resolve()) for p in segment.images if Path(p).exists()]
            text = normalize_text(segment.text)
            if not text and not images:
                continue
            if images:
                visual_segments += 1
            segments_list.append(
                CanonicalSegment(text=text, images=images, metadata=segment.metadata)
            )
        doc = CanonicalDocument(
            content_hash=content_hash,
            source_path=str(source.resolve()),
            source_type=source.suffix.lower().lstrip("."),
            parser_name=f"oracle-lite:{source.suffix.lower()}",
            parser_version=parser_version,
            text="",
            title=parsed.title,
            segments=segments_list,
            metadata={**parsed.metadata, "low_memory_mode": True},
        )
        doc.write_json(canonical_path)
        return {
            "visual_segments": visual_segments,
            "has_visual": bool(visual_segments),
            "low_memory_mode": True,
        }

    if segments == 0:
        raise ValueError("Low-memory parser produced no canonical segments")

    sidecar_tmp.replace(sidecar)
    _write_header(
        canonical_path=canonical_path,
        source=source,
        content_hash=content_hash,
        parser_version=parser_version,
        metadata=metadata,
        segments_path=sidecar,
    )
    return {
        "visual_segments": visual_segments,
        "has_visual": bool(visual_segments),
        "low_memory_mode": True,
    }


def parse_and_write_canonical(
    *,
    source_path: str,
    asset_dir: str,
    canonical_path: str,
    content_hash: str,
    parser_version: str,
    result_queue,
    address_space_limit_bytes: int,
    low_memory: bool = False,
) -> None:
    """Parse exactly one source file inside an isolated, memory-capped worker."""
    hard_limit_applied = apply_linux_address_space_limit(address_space_limit_bytes)
    try:
        source = Path(source_path)
        canonical = Path(canonical_path)
        assets = Path(asset_dir)

        if low_memory:
            result = _stream_low_memory_canonical(
                source=source,
                asset_dir=assets,
                canonical_path=canonical,
                content_hash=content_hash,
                parser_version=parser_version,
            )
            result_queue.put({
                "ok": True,
                "hard_limit_applied": hard_limit_applied,
                **result,
            })
            return

        # Heavy parser imports happen only after the Linux address-space limit is
        # applied, so module import and source parsing share the same hard ceiling.
        from .canonical import CanonicalDocument, CanonicalSegment
        from .parsers import parse_file

        parsed = parse_file(source, asset_dir=assets)
        document_text = ""
        if not parsed.segments:
            document_text = normalize_text(parsed.text)
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
            has_text = has_text or bool(segment_text)
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
        doc.write_json(canonical)

        result_queue.put({
            "ok": True,
            "hard_limit_applied": hard_limit_applied,
            "visual_segments": visual_segments,
            "has_visual": bool(visual_segments),
            "low_memory_mode": False,
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
