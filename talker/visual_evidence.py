"""Source-verified PDF detail views for small query-matching text."""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=32)
def _file_digest(path: str, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _word(text: str) -> str:
    value = text.lower().strip('.,:;()[]')
    return value[:-1] if len(value) > 4 and value.endswith('s') and not value.endswith('ss') else value


def pdf_detail_view(source: Path, content_hash: str, page_number: int,
                    terms: list[str], output_dir: Path) -> Path | None:
    if source.suffix.lower() != '.pdf' or not source.is_file():
        return None
    stat = source.stat()
    # Never attach live changed PDF content to evidence from an older canonical
    # artifact. Rendering must use the exact source bytes represented by the hash.
    if _file_digest(str(source), stat.st_mtime_ns, stat.st_size) != content_hash:
        return None
    import pymupdf
    wanted = {_word(term) for term in terms if len(term) > 2}
    with pymupdf.open(source) as pdf:
        if not 1 <= page_number <= len(pdf):
            return None
        page = pdf[page_number - 1]
        words = [w for w in page.get_text('words') if _word(w[4]) in wanted]
        if not words:
            return None
        anchor = min(words, key=lambda w: w[3] - w[1])
        if (anchor[3] - anchor[1]) / page.rect.height >= 0.0075:
            return None
        width, height = page.rect.width * 0.4, page.rect.height * 0.4
        cx, cy = (anchor[0]+anchor[2])/2, (anchor[1]+anchor[3])/2
        x = min(max(0, cx-width/2), page.rect.width-width)
        y = min(max(0, cy-height/2), page.rect.height-height)
        clip = pymupdf.Rect(x, y, x+width, y+height)
        crop_key = hashlib.sha256(str(tuple(clip)).encode()).hexdigest()[:12]
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f'{content_hash[:16]}-page{page_number}-{crop_key}.png'
        if not target.exists():
            scale = math.sqrt(786432 / (width * height))
            page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip).save(target)
        return target
