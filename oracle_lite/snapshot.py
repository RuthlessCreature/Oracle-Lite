from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig
from .db import Registry


@dataclass(slots=True)
class SnapshotResult:
    snapshot_id: str
    manifest_path: Path
    total: int
    current: int
    replay: int


def _ready_current(registry: Registry, parser_version: str) -> list[dict]:
    with registry.connect() as conn:
        rows = conn.execute(
            """SELECT sf.content_hash,
                      MIN(sf.path) AS source_path,
                      da.canonical_path
               FROM source_files sf
               JOIN derived_artifacts da
                 ON da.content_hash=sf.content_hash
                AND da.parser_version=?
               WHERE sf.status='active'
                 AND da.status='ready'
                 AND da.canonical_path IS NOT NULL
               GROUP BY sf.content_hash, da.canonical_path
               ORDER BY source_path""",
            (parser_version,),
        ).fetchall()
    return [dict(r) for r in rows]


def create_snapshot(
    cfg: AppConfig,
    *,
    name: str,
    mode: str = "full",
    base_snapshot_id: str | None = None,
    history_replay_ratio: float = 0.20,
    seed: int = 42,
) -> SnapshotResult:
    if mode not in {"full", "incremental"}:
        raise ValueError("mode must be 'full' or 'incremental'")
    if not 0.0 <= history_replay_ratio < 1.0:
        raise ValueError("history_replay_ratio must be in [0, 1)")
    if mode == "incremental" and not base_snapshot_id:
        raise ValueError("incremental snapshots require base_snapshot_id")

    registry = Registry(cfg.registry_path)
    current = _ready_current(registry, cfg.parser_version)
    current_by_hash = {m["content_hash"]: m for m in current}

    members: list[dict] = []
    replay_count = 0
    current_count = 0

    if mode == "full":
        for item in current:
            members.append({**item, "role": "current", "weight": 1.0})
        current_count = len(members)
    else:
        base_hashes = registry.snapshot_hashes(base_snapshot_id or "")
        new_hashes = sorted(set(current_by_hash) - base_hashes)
        history_hashes = sorted(set(current_by_hash) & base_hashes)

        for h in new_hashes:
            members.append({**current_by_hash[h], "role": "current", "weight": 1.0})
        current_count = len(new_hashes)

        if new_hashes and history_hashes and history_replay_ratio > 0:
            # ratio is the desired fraction of replay examples in the selected set.
            wanted = math.ceil(
                len(new_hashes) * history_replay_ratio / (1.0 - history_replay_ratio)
            )
            rng = random.Random(seed)
            selected = rng.sample(history_hashes, min(wanted, len(history_hashes)))
            for h in sorted(selected):
                members.append({**current_by_hash[h], "role": "replay", "weight": 1.0})
            replay_count = len(selected)

    if not members:
        raise ValueError("No eligible canonical documents for snapshot")

    fingerprint_src = "\n".join(
        f"{m['content_hash']}|{m['role']}|{m['canonical_path']}" for m in sorted(
            members, key=lambda x: (x["content_hash"], x["role"])
        )
    )
    digest = hashlib.sha256(fingerprint_src.encode("utf-8")).hexdigest()[:10]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_id = f"{ts}-{digest}"

    snapshot_dir = cfg.snapshots_dir / snapshot_id
    if snapshot_dir.exists():
        raise FileExistsError(f"Snapshot already exists: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True)
    manifest_path = (snapshot_dir / "manifest.jsonl").resolve()

    with manifest_path.open("w", encoding="utf-8") as f:
        for member in sorted(members, key=lambda x: (x["role"], x["source_path"])):
            f.write(json.dumps(member, ensure_ascii=False) + "\n")

    config = {
        "parser_version": cfg.parser_version,
        "mode": mode,
        "base_snapshot_id": base_snapshot_id,
        "history_replay_ratio": history_replay_ratio,
        "seed": seed,
    }
    (snapshot_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "snapshot_id": snapshot_id,
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "counts": {
                    "total": len(members),
                    "current": current_count,
                    "replay": replay_count,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    registry.save_snapshot(
        snapshot_id=snapshot_id,
        name=name,
        mode=mode,
        base_snapshot_id=base_snapshot_id,
        manifest_path=str(manifest_path),
        config=config,
        members=members,
    )

    return SnapshotResult(
        snapshot_id=snapshot_id,
        manifest_path=manifest_path,
        total=len(members),
        current=current_count,
        replay=replay_count,
    )
