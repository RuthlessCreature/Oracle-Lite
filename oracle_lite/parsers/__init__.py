from __future__ import annotations

from pathlib import Path

from .json_parser import parse_json_like
from .office import parse_docx, parse_pdf, parse_pptx
from .text import parse_text_like


def parse_file(path: str | Path) -> tuple[str, str, dict]:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".txt", ".md", ".csv"}:
        return parse_text_like(path)
    if ext in {".json", ".jsonl"}:
        return parse_json_like(path)
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext == ".pptx":
        return parse_pptx(path)

    raise ValueError(f"Unsupported file type: {ext}")
