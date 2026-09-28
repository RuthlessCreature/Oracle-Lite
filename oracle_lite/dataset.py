from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .canonical import CanonicalDocument
from .config import AppConfig
from .db import Registry
from .memory import MemoryPressureGate


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
    memory_gate: MemoryPressureGate | None = None,
) -> DatasetResult:
    """Build mixed text + image/text domain records with bounded memory."""
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
        # Stream snapshot members one line at a time instead of materializing the
        # entire manifest. Each canonical segment is also streamed from its sidecar.
        with manifest_path.open("r", encoding="utf-8") as manifest:
            for line in manifest:
                if memory_gate is not None:
                    memory_gate.wait_until_safe(context="building dataset")
                if not line.strip():
                    continue
                member = json.loads(line)
                canonical_path = Path(member["canonical_path"])
                header = CanonicalDocument.read_header(canonical_path)
                role = member.get("role", "current")
                emitted_segment = False

                for index, segment in enumerate(CanonicalDocument.iter_segments(canonical_path)):
                    if memory_gate is not None:
                        memory_gate.wait_until_safe(context="streaming canonical segment")
                    text = segment.text.strip()
                    images = [p for p in segment.images if Path(p).exists()]
                    if not text and not images:
                        continue

                    write_record({
                        "mode": "multimodal" if images else "text",
                        "text": text,
                        "images": images,
                        "content_hash": header["content_hash"],
                        "source_path": member["source_path"],
                        "segment_index": index,
                        "segment_metadata": segment.metadata,
                        "role": role,
                    })
                    emitted_segment = True
                    characters += len(text)
                    if images:
                        visual_records += 1
                    else:
                        text_records += 1

                document_text = str(header.get("text") or "").strip()
                if not emitted_segment and document_text:
                    write_record({
                        "mode": "text",
                        "text": document_text,
                        "images": [],
                        "content_hash": header["content_hash"],
                        "source_path": member["source_path"],
                        "segment_index": 0,
                        "segment_metadata": {"kind": "document_fallback"},
                        "role": role,
                    })
                    characters += len(document_text)
                    text_records += 1
    finally:
        if shard is not None:
            shard.close()

    manifest = {
        "snapshot_id": snapshot_id,
        "dataset_kind": "multimodal-domain-adaptation-v2-streaming",
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


build_cpt_dataset = build_domain_dataset
