from __future__ import annotations

import re


_BLANKS = re.compile(r"\n{4,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalize_text(text: str) -> str:
    """Conservative normalization only; never rewrites factual content."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("\n", text)
    text = _BLANKS.sub("\n\n\n", text)
    return text.strip()
