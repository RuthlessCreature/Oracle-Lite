from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable


HashProgress = Callable[[int, int], None]


def hash_file(
    path: str | Path,
    algorithm: str = "sha256",
    chunk_size: int = 1024 * 1024,
    progress: HashProgress | None = None,
) -> str:
    file_path = Path(path)
    total = file_path.stat().st_size
    processed = 0
    h = hashlib.new(algorithm)
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            processed += len(chunk)
            if progress is not None:
                progress(processed, total)
    return h.hexdigest()
