from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .canonical import CanonicalDocument
from .config import AppConfig
from .db import Registry
from .resources import GIB, HostResourcePolicy, wait_for_disk


@dataclass(slots=True)
class DatasetResult:
    output_dir: Path
    shard_paths: list[Path]
    records: int
    text_records: int
    visual_records: int
    characters: int
    qa_records: int = 0


def build_domain_dataset(
    cfg: AppConfig,
    *,
    snapshot_id: str,
    max_records_per_shard: int = 2000,
) -> DatasetResult:
    """Build mixed text + image/text domain records with bounded memory."""
    registry = Registry(cfg.registry_path)
    resource_policy = HostResourcePolicy.auto(cfg.output_dir)
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
    qa_records = 0
    characters = 0
    qa_members: list[dict] = []

    def write_record(record: dict) -> None:
        nonlocal shard, shard_idx, in_shard, records
        # Keep a stable JSONL schema so streaming Arrow/JSON readers can cast
        # the final QA rows after the multimodal rows in the same shard.
        record.setdefault("question", "")
        record.setdefault("answer", "")
        record.setdefault("context", "")
        record.setdefault("content_hash", "")
        record.setdefault("segment_index", 0)
        record.setdefault("segment_metadata", {})
        record.setdefault("role", "current")
        record.setdefault("images", [])
        # Dataset records are bounded, but disk admission is rechecked
        # periodically so long builds cannot consume the OS reserve.
        if records % 25 == 0:
            wait_for_disk(
                cfg.output_dir,
                resource_policy,
                required_bytes=1 * GIB,
                poll_seconds=2.0,
                stable_samples=1,
            )
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
                if not line.strip():
                    continue
                member = json.loads(line)
                # This file is parsed separately as curated instruction data
                # below. Training its flattened JSON representation as raw text
                # would teach the model to continue JSON instead of answering.
                if Path(member["source_path"]).name == "qa_pairs.jsonl":
                    qa_members.append(member)
                    continue
                canonical_path = Path(member["canonical_path"])
                header = CanonicalDocument.read_header(canonical_path)
                role = member.get("role", "current")
                emitted_segment = False

                for index, segment in enumerate(CanonicalDocument.iter_segments(canonical_path)):
                    text = segment.text.strip()
                    images = [p for p in segment.images if Path(p).exists()]
                    if not text and not images:
                        continue

                    if images:
                        # Hard memory bound: one visual asset per training record.
                        # Multi-image pages/slides/documents are expanded into
                        # independent source-grounded examples.
                        for image_index, image_path in enumerate(images):
                            metadata = dict(segment.metadata)
                            metadata["image_index"] = image_index
                            metadata["image_count"] = len(images)
                            write_record({
                                "mode": "multimodal",
                                "text": text,
                                "images": [image_path],
                                "content_hash": header["content_hash"],
                                "source_path": member["source_path"],
                                "segment_index": index,
                                "segment_metadata": metadata,
                                "role": role,
                            })
                            visual_records += 1
                            characters += len(text)
                    else:
                        write_record({
                            "mode": "text",
                            "text": text,
                            "images": [],
                            "content_hash": header["content_hash"],
                            "source_path": member["source_path"],
                            "segment_index": index,
                            "segment_metadata": segment.metadata,
                            "role": role,
                        })
                        text_records += 1
                        characters += len(text)
                    emitted_segment = True

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
            shard = None

    try:
        for qa_member in qa_members:
            qa_path = Path(qa_member.get("qa_snapshot_path") or qa_member["source_path"])
            digest = hashlib.sha256()
            with qa_path.open("rb") as qa_source:
                for chunk in iter(lambda: qa_source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != qa_member["content_hash"]:
                raise ValueError("QA contents no longer match the snapshot; prepare a new snapshot")
            with qa_path.open("r", encoding="utf-8-sig") as qa_file:
                for line_number, line in enumerate(qa_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        pair = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {qa_path}:{line_number}: {exc}"
                        ) from exc
                    if not isinstance(pair, dict):
                        raise ValueError(f"{qa_path}:{line_number} requires a JSON object")
                    if not all(isinstance(pair.get(key), str) for key in ("question", "answer", "context")):
                        raise ValueError(f"{qa_path}:{line_number} requires string question, answer, and context")
                    question = str(pair.get("question") or "").strip()
                    answer = str(pair.get("answer") or "").strip()
                    context = str(pair.get("context") or "").strip()
                    if not question or not answer or not context:
                        raise ValueError(
                            f"{qa_path}:{line_number} requires non-empty question, answer, and context"
                        )
                    source_path = str(pair.get("source_path") or "curated source")
                    write_record({
                        "mode": "instruction",
                        "text": answer,
                        "question": question,
                        "answer": answer,
                        "context": context,
                        "source_path": source_path,
                        "images": [],
                        "content_hash": qa_member["content_hash"],
                        "role": qa_member.get("role", "current"),
                    })
                    qa_records += 1
                    characters += len(context) + len(answer)
    finally:
        if shard is not None:
            shard.close()
            shard = None

    manifest = {
        "snapshot_id": snapshot_id,
        "dataset_kind": "multimodal-domain-adaptation-v2-streaming",
        "records": records,
        "text_records": text_records,
        "visual_records": visual_records,
        "qa_records": qa_records,
        "characters": characters,
        "shards": [p.name for p in shard_paths],
    }
    wait_for_disk(
        cfg.output_dir,
        resource_policy,
        required_bytes=1 * GIB,
        poll_seconds=2.0,
        stable_samples=1,
    )
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
        qa_records=qa_records,
    )


build_cpt_dataset = build_domain_dataset
