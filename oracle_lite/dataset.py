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
    records: int
    text_records: int
    visual_records: int
    characters: int


def build_domain_dataset(
    cfg: AppConfig,
    *,
    snapshot_id: str,
    max_records_per_shard: int = 2000,
) -> DatasetResult:
    """Build one mixed domain-adaptation dataset.

    Each record is either:
    - text-only: raw source text for continued language modeling; or
    - multimodal: one or more source images plus source-grounded text.

    No synthetic professional answer is generated here.
    """
    registry = Registry(cfg.registry_path)
    snap = registry.get_snapshot(snapshot_id)
    if snap is None:
        raise ValueError(f"Unknown snapshot: {snapshot_id}")

    manifest_path = Path(snap["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    output_dir = cfg.datasets_dir / snapshot_id / "domain"
    if output_dir.exists() and any(output_dir.glob("part-*.jsonl")):
        raise FileExistsError(
            f"Dataset already built at {output_dir}; remove it explicitly to rebuild."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []
    shard = None
    shard_idx = -1
    in_shard = 0
    records = 0
    text_records = 0
    visual_records = 0
    characters = 0

    def write_record(record: dict) -> None:
        nonlocal shard, shard_idx, in_shard, records
        if shard is None or in_shard >= max_records_per_shard:
            if shard is not None:
                shard.close()
            shard_idx += 1
            shard_path = output_dir / f"part-{shard_idx:05d}.jsonl"
            shard_paths.append(shard_path)
            shard = shard_path.open("w", encoding="utf-8")
            in_shard = 0
        shard.write(json.dumps(record, ensure_ascii=False) + "\n")
        in_shard += 1
        records += 1

    try:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            member = json.loads(line)
            doc = CanonicalDocument.read_json(member["canonical_path"])
            role = member.get("role", "current")

            # Segment-level records preserve PDF page / PPT slide visual grounding.
            emitted_segment = False
            for index, segment in enumerate(doc.segments):
                text = segment.text.strip()
                images = [p for p in segment.images if Path(p).exists()]
                if not text and not images:
                    continue

                record = {
                    "mode": "multimodal" if images else "text",
                    "text": text,
                    "images": images,
                    "content_hash": doc.content_hash,
                    "source_path": member["source_path"],
                    "segment_index": index,
                    "segment_metadata": segment.metadata,
                    "role": role,
                }
                write_record(record)
                emitted_segment = True
                characters += len(text)
                if images:
                    visual_records += 1
                else:
                    text_records += 1

            # Backward-safe fallback for parsers that produced only document text.
            if not emitted_segment and doc.text.strip():
                write_record({
                    "mode": "text",
                    "text": doc.text,
                    "images": [],
                    "content_hash": doc.content_hash,
                    "source_path": member["source_path"],
                    "segment_index": 0,
                    "segment_metadata": {"kind": "document_fallback"},
                    "role": role,
                })
                characters += len(doc.text)
                text_records += 1
    finally:
        if shard is not None:
            shard.close()

    manifest = {
        "snapshot_id": snapshot_id,
        "dataset_kind": "multimodal-domain-adaptation-v1",
        "records": records,
        "text_records": text_records,
        "visual_records": visual_records,
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
        records=records,
        text_records=text_records,
        visual_records=visual_records,
        characters=characters,
    )


# Backward compatibility for early v0.1 callers.
build_cpt_dataset = build_domain_dataset
