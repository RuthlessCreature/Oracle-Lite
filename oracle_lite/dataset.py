from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .canonical import CanonicalDocument
from .config import AppConfig
from .db import Registry


@dataclass(slots=True)
class DatasetResult:
    output_dir: Path
    shard_paths: list[Path]
    documents: int
    characters: int


def build_cpt_dataset(
    cfg: AppConfig,
    *,
    snapshot_id: str,
    max_records_per_shard: int = 5000,
) -> DatasetResult:
    registry = Registry(cfg.registry_path)
    snap = registry.get_snapshot(snapshot_id)
    if snap is None:
        raise ValueError(f"Unknown snapshot: {snapshot_id}")

    manifest_path = Path(snap["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    output_dir = cfg.datasets_dir / snapshot_id / "cpt"
    if output_dir.exists() and any(output_dir.glob("part-*.jsonl")):
        raise FileExistsError(
            f"Dataset already built at {output_dir}; remove it explicitly to rebuild."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    shard = None
    shard_idx = -1
    in_shard = 0
    documents = 0
    characters = 0

    try:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            member = json.loads(line)
            doc = CanonicalDocument.read_json(member["canonical_path"])
            if not doc.text.strip():
                continue

            if shard is None or in_shard >= max_records_per_shard:
                if shard is not None:
                    shard.close()
                shard_idx += 1
                shard_path = output_dir / f"part-{shard_idx:05d}.jsonl"
                shard_paths.append(shard_path)
                shard = shard_path.open("w", encoding="utf-8")
                in_shard = 0

            record = {
                "text": doc.text,
                "content_hash": doc.content_hash,
                "source_path": member["source_path"],
                "role": member.get("role", "current"),
            }
            shard.write(json.dumps(record, ensure_ascii=False) + "\n")
            documents += 1
            characters += len(doc.text)
            in_shard += 1
    finally:
        if shard is not None:
            shard.close()

    manifest = {
        "snapshot_id": snapshot_id,
        "documents": documents,
        "characters": characters,
        "shards": [p.name for p in shard_paths],
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return DatasetResult(
        output_dir=output_dir,
        shard_paths=shard_paths,
        documents=documents,
        characters=characters,
    )
