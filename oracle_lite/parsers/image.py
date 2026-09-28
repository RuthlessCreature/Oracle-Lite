from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from .base import ParsedDocument, ParsedSegment


def parse_image(path: Path, asset_dir: Path) -> ParsedDocument:
    """Preserve a native image without inventing a text label from its filename."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    target = asset_dir / f"source{suffix}"
    shutil.copy2(path, target)

    with Image.open(target) as img:
        width, height = img.size
        mode = img.mode

    return ParsedDocument(
        title=path.stem,
        text="",
        segments=[
            ParsedSegment(
                text="",
                images=[str(target.resolve())],
                metadata={"kind": "native_image", "label_status": "unlabeled"},
            )
        ],
        metadata={
            "extension": suffix,
            "width": width,
            "height": height,
            "mode": mode,
            "images": 1,
        },
    )
