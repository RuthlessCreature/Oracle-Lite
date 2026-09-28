from __future__ import annotations

from pathlib import Path

from docx import Document
from pypdf import PdfReader
from pptx import Presentation


def parse_docx(path: Path) -> tuple[str, str, dict]:
    doc = Document(path)
    blocks: list[str] = []
    headings = 0
    tables = 0

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name or "").lower() if p.style else ""
        if style.startswith("heading"):
            level = 1
            parts = style.split()
            if parts and parts[-1].isdigit():
                level = max(1, min(6, int(parts[-1])))
            blocks.append(f"{'#' * level} {text}")
            headings += 1
        else:
            blocks.append(text)

    for table in doc.tables:
        rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        blocks.append("| " + " | ".join(rows[0]) + " |")
        blocks.append("| " + " | ".join(["---"] * width) + " |")
        for row in rows[1:]:
            blocks.append("| " + " | ".join(row) + " |")
        tables += 1

    return path.stem, "\n\n".join(blocks), {
        "extension": ".docx",
        "headings": headings,
        "tables": tables,
    }


def parse_pdf(path: Path) -> tuple[str, str, dict]:
    reader = PdfReader(str(path))
    pages: list[str] = []
    empty_pages = 0

    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            empty_pages += 1
            continue
        pages.append(f"## Page {idx}\n\n{text}")

    return path.stem, "\n\n".join(pages), {
        "extension": ".pdf",
        "pages": len(reader.pages),
        "empty_pages": empty_pages,
        "needs_ocr": empty_pages == len(reader.pages) and len(reader.pages) > 0,
    }


def parse_pptx(path: Path) -> tuple[str, str, dict]:
    prs = Presentation(path)
    slides: list[str] = []
    table_count = 0

    for slide_idx, slide in enumerate(prs.slides, start=1):
        items: list[tuple[int, int, str]] = []
        for shape in slide.shapes:
            top = int(getattr(shape, "top", 0))
            left = int(getattr(shape, "left", 0))

            if getattr(shape, "has_text_frame", False):
                text = "\n".join(
                    p.text.strip()
                    for p in shape.text_frame.paragraphs
                    if p.text.strip()
                )
                if text:
                    items.append((top, left, text))

            if getattr(shape, "has_table", False):
                rows = []
                for row in shape.table.rows:
                    rows.append([cell.text.strip().replace("\n", " ") for cell in row.cells])
                if rows:
                    width = max(len(r) for r in rows)
                    rows = [r + [""] * (width - len(r)) for r in rows]
                    md = ["| " + " | ".join(rows[0]) + " |",
                          "| " + " | ".join(["---"] * width) + " |"]
                    md.extend("| " + " | ".join(r) + " |" for r in rows[1:])
                    items.append((top, left, "\n".join(md)))
                    table_count += 1

        items.sort(key=lambda x: (x[0], x[1]))
        body = "\n\n".join(text for _, _, text in items)
        if body:
            slides.append(f"## Slide {slide_idx}\n\n{body}")

    return path.stem, "\n\n".join(slides), {
        "extension": ".pptx",
        "slides": len(prs.slides),
        "tables": table_count,
    }
