from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import fitz

from oracle_lite.resources import scale_for_pixel_budget


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv"}
JSON_EXTENSIONS = {".json", ".jsonl"}
CAD_TEXT_EXTENSIONS = {".stp", ".step", ".x_t"}
SUPPORTED_EXTENSIONS = (
    IMAGE_EXTENSIONS
    | TEXT_EXTENSIONS
    | JSON_EXTENSIONS
    | CAD_TEXT_EXTENSIONS
    | {".x_b", ".pdf", ".docx", ".pptx"}
)

MAX_TEXT_CHARS = 16_000
MAX_RENDERED_IMAGES = 2
MAX_IMAGE_PIXELS = 196_608


def _bounded_text(path: Path, max_chars: int = MAX_TEXT_CHARS) -> str:
    chunks: list[str] = []
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        while total < max_chars:
            chunk = stream.read(min(64_000, max_chars - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    return "".join(chunks)


def _json_summary(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        lines: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for index, line in enumerate(stream):
                if index >= 200 or sum(map(len, lines)) >= MAX_TEXT_CHARS:
                    break
                lines.append(line)
        return "".join(lines)[:MAX_TEXT_CHARS]
    return _bounded_text(path)


def _xml_text(stream) -> str:
    pieces: list[str] = []
    total = 0
    for event, elem in ElementTree.iterparse(stream, events=("end",)):
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag in {"t", "p", "r"} and elem.text:
            value = elem.text.strip()
            if value:
                pieces.append(value)
                total += len(value) + 1
                if total >= MAX_TEXT_CHARS:
                    break
        elem.clear()
    return "\n".join(pieces)[:MAX_TEXT_CHARS]


def _office_summary(path: Path, asset_dir: Path) -> tuple[str, list[str]]:
    ext = path.suffix.lower()
    with zipfile.ZipFile(path) as archive:
        if ext == ".docx":
            xml_names = ["word/document.xml"] if "word/document.xml" in archive.namelist() else []
            media_prefix = "word/media/"
        else:
            xml_names = sorted(
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            )
            media_prefix = "ppt/media/"

        texts: list[str] = []
        total = 0
        for name in xml_names:
            with archive.open(name) as stream:
                text = _xml_text(stream)
            if text:
                remaining = MAX_TEXT_CHARS - total
                texts.append(text[:remaining])
                total += min(len(text), remaining)
            if total >= MAX_TEXT_CHARS:
                break

        asset_dir.mkdir(parents=True, exist_ok=True)
        images: list[str] = []
        for name in sorted(archive.namelist()):
            if len(images) >= MAX_RENDERED_IMAGES:
                break
            if not name.startswith(media_prefix):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            target = asset_dir / f"media-{len(images)+1:02d}{suffix}"
            with archive.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            images.append(str(target.resolve()))
    return "\n".join(texts)[:MAX_TEXT_CHARS], images


def _pdf_summary(path: Path, asset_dir: Path) -> tuple[str, list[str]]:
    doc = fitz.open(path)
    text_parts: list[str] = []
    images: list[str] = []
    total = 0
    asset_dir.mkdir(parents=True, exist_ok=True)
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_text = (page.get_text("text") or "").strip()
            if page_text and total < MAX_TEXT_CHARS:
                remaining = MAX_TEXT_CHARS - total
                value = page_text[:remaining]
                text_parts.append(f"[Page {page_index + 1}]\n{value}")
                total += len(value)

            if len(images) < MAX_RENDERED_IMAGES:
                rect = page.rect
                scale = scale_for_pixel_budget(
                    rect.width,
                    rect.height,
                    MAX_IMAGE_PIXELS,
                )
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    alpha=False,
                )
                target = asset_dir / f"page-{page_index + 1:03d}.png"
                pix.save(str(target))
                images.append(str(target.resolve()))
                del pix
            page = None
    finally:
        doc.close()
    return "\n\n".join(text_parts)[:MAX_TEXT_CHARS], images


def _step_summary(path: Path) -> str:
    schema = []
    entities: dict[str, int] = {}
    schema_seen = False
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            upper = line.upper()
            if "FILE_SCHEMA" in upper:
                schema_seen = True
            if schema_seen and "'" in line and len(schema) < 8:
                schema.extend(re.findall(r"'([^']+)'", line))
                if ";" in line:
                    schema_seen = False
            match = re.match(r"^\s*#\d+\s*=\s*([A-Z0-9_]+)\s*\(", line, re.I)
            if match:
                name = match.group(1).upper()
                entities[name] = entities.get(name, 0) + 1
    top = sorted(entities.items(), key=lambda item: (-item[1], item[0]))[:50]
    lines = [
        "Mechanical CAD attachment (STEP)",
        f"FILE_SCHEMA: {', '.join(schema) if schema else 'unknown'}",
        f"Entity count: {sum(entities.values())}",
    ]
    lines.extend(f"{name}: {count}" for name, count in top)
    return "\n".join(lines)


def _xt_summary(path: Path) -> str:
    version = None
    schema = None
    tokens: dict[str, int] = {}
    matcher = re.compile(r"\b(BODY|ASSEMBLY|WORLD|FACE|EDGE|VERTEX|SHELL|REGION)\b", re.I)
    with path.open("r", encoding="latin-1", errors="replace") as stream:
        for line in stream:
            if version is None and "modeller version" in line.lower():
                version = line.strip()[:200]
            if schema is None:
                match = re.search(r"\bSCH_[0-9_]+\b", line, re.I)
                if match:
                    schema = match.group(0)
            for token in matcher.findall(line):
                key = token.upper()
                tokens[key] = tokens.get(key, 0) + 1
    lines = [
        "Mechanical CAD attachment (Parasolid X_T)",
        f"Version header: {version or 'unknown'}",
        f"Schema: {schema or 'unknown'}",
    ]
    lines.extend(
        f"{name}: {count}"
        for name, count in sorted(tokens.items(), key=lambda item: (-item[1], item[0]))
    )
    return "\n".join(lines)


def parse_attachment(path: str | Path, asset_dir: str | Path) -> tuple[str, list[str]]:
    source = Path(path).resolve()
    asset_dir = Path(asset_dir).resolve()
    ext = source.suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return "", [str(source)]
    if ext in TEXT_EXTENSIONS:
        return _bounded_text(source), []
    if ext in JSON_EXTENSIONS:
        return _json_summary(source), []
    if ext == ".pdf":
        return _pdf_summary(source, asset_dir)
    if ext in {".docx", ".pptx"}:
        return _office_summary(source, asset_dir)
    if ext in {".stp", ".step"}:
        return _step_summary(source), []
    if ext == ".x_t":
        return _xt_summary(source), []
    if ext == ".x_b":
        return (
            "Mechanical CAD attachment (Parasolid X_B binary). "
            "Binary geometry is preserved but not decoded into invented text.",
            [],
        )
    return (
        f"Attachment {source.name} is stored, but this file type is not parsed.",
        [],
    )
