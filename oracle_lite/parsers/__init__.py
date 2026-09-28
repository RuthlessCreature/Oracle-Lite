from __future__ import annotations

from pathlib import Path

from .image import parse_image
from .json_parser import parse_json_like
from .office import parse_docx, parse_pdf, parse_pptx
from .text import parse_text_like


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_file(path: str | Path, *, asset_dir: str | Path):
    path = Path(path)
    asset_dir = Path(asset_dir)
    ext = path.suffix.lower()

    if ext in {".txt", ".md", ".csv"}:
        return parse_text_like(path)
    if ext in {".json", ".jsonl"}:
        return parse_json_like(path)
    if ext == ".pdf":
        return parse_pdf(path, asset_dir)
    if ext == ".docx":
        return parse_docx(path, asset_dir)
    if ext == ".pptx":
        return parse_pptx(path, asset_dir)
    if ext in IMAGE_EXTENSIONS:
        return parse_image(path, asset_dir)

    raise ValueError(f"Unsupported file type: {ext}")
