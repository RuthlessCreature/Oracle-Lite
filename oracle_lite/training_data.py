from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info


class JSONLStreamingDataset(IterableDataset):
    """Bounded JSONL streaming without Arrow schemas, network or cache writes."""

    def __init__(self, shard_paths, *, mode=None):
        super().__init__()
        self.shard_paths = [Path(path) for path in shard_paths]
        self.mode = mode
        if not self.shard_paths:
            raise ValueError("Training dataset requires at least one shard")

    def __iter__(self):
        worker = get_worker_info()
        paths = self.shard_paths if worker is None else self.shard_paths[worker.id::worker.num_workers]
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number} requires a JSON object")
                    if self.mode is not None and row.get("mode", "text") != self.mode:
                        continue
                    if (row.get("text") or "").strip():
                        yield row


def freeze_instruction_shard(source: Path, directory: Path):
    """Validate and freeze external SFT examples without changing the corpus."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.jsonl', delete=False) as frozen:
        path = Path(frozen.name)
        try:
            with Path(source).open('rb') as handle:
                for number, raw in enumerate(handle, start=1):
                    if raw.strip():
                        row = json.loads(raw)
                        if not isinstance(row, dict) or row.get('mode') != 'instruction':
                            raise ValueError(f'{source}:{number} requires mode=instruction')
                        if any(not isinstance(row.get(key), str) or not row[key].strip()
                               for key in ('text', 'question', 'answer', 'context')):
                            raise ValueError(f'{source}:{number} requires nonempty text, question, answer, context')
                        if row['text'].strip() != row['answer'].strip():
                            raise ValueError(f'{source}:{number} requires text=answer')
                        images = row.get('images') or []
                        if not isinstance(images, list) or len(images) > 1:
                            raise ValueError(f'{source}:{number} allows at most one instruction image')
                        if images:
                            from PIL import Image
                            image_path = Path(images[0])
                            if not image_path.is_absolute():
                                image_path = Path(source).parent / image_path
                            data = image_path.read_bytes()
                            # Decode the bytes that will be frozen, not a second
                            # read of a possibly changing live file.
                            from io import BytesIO
                            with Image.open(BytesIO(data)) as image:
                                image.verify()
                            image_hash = hashlib.sha256(data).hexdigest()
                            image_dir = directory / 'images'
                            image_dir.mkdir(exist_ok=True)
                            image_target = image_dir / (image_hash + image_path.suffix.lower())
                            if not image_target.exists():
                                image_target.write_bytes(data)
                            elif hashlib.sha256(image_target.read_bytes()).hexdigest() != image_hash:
                                raise ValueError(f'Frozen instruction image hash mismatch: {image_target}')
                            row['images'] = [str(image_target.resolve())]
                            row['image_sha256'] = image_hash
                            raw = (json.dumps(row, ensure_ascii=False) + '\n').encode()
                        count += 1
                    frozen.write(raw)
                    digest.update(raw)
            if not count:
                raise ValueError('External instruction shard is empty')
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return path, count, digest.hexdigest()
