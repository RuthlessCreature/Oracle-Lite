from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig
from .db import Registry, utcnow
from .resources import HostResourcePolicy, MIB, wait_for_disk


@dataclass(slots=True)
class SnapshotResult:
    snapshot_id: str
    manifest_path: Path
    total: int
    current: int
    replay: int


CURRENT_CTE = """
WITH current AS (
    SELECT sf.content_hash,
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
)
"""


def _stable_score(seed: int, content_hash: str) -> str:
    return hashlib.sha256(f"{seed}:{content_hash}".encode("utf-8")).hexdigest()


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
    policy = HostResourcePolicy.auto(cfg.output_dir)

    with registry.connect() as conn:
        conn.create_function(
            "stable_score",
            1,
            lambda content_hash: _stable_score(seed, str(content_hash)),
        )

        total_active = int(conn.execute(
            CURRENT_CTE + " SELECT COUNT(*) AS n FROM current",
            (cfg.parser_version,),
        ).fetchone()["n"])
        if total_active <= 0:
            raise ValueError("No eligible canonical documents for snapshot")

        replay_count = 0
        current_count = total_active

        if mode == "full":
            current_sql = CURRENT_CTE + """
                SELECT content_hash,source_path,canonical_path,
                       'current' AS role,1.0 AS weight
                  FROM current
                 ORDER BY source_path
            """
            current_params = (cfg.parser_version,)
            replay_sql = None
            replay_params = ()
        else:
            current_sql = CURRENT_CTE + """
                SELECT c.content_hash,c.source_path,c.canonical_path,
                       'current' AS role,1.0 AS weight
                  FROM current c
                 WHERE NOT EXISTS (
                       SELECT 1
                         FROM snapshot_members sm
                        WHERE sm.snapshot_id=?
                          AND sm.content_hash=c.content_hash
                 )
                 ORDER BY c.source_path
            """
            current_params = (cfg.parser_version, base_snapshot_id)
            current_count = int(conn.execute(
                CURRENT_CTE + """
                    SELECT COUNT(*) AS n
                      FROM current c
                     WHERE NOT EXISTS (
                           SELECT 1
                             FROM snapshot_members sm
                            WHERE sm.snapshot_id=?
                              AND sm.content_hash=c.content_hash
                     )
                """,
                (cfg.parser_version, base_snapshot_id),
            ).fetchone()["n"])

            history_count = int(conn.execute(
                CURRENT_CTE + """
                    SELECT COUNT(*) AS n
                      FROM current c
                     WHERE EXISTS (
                           SELECT 1
                             FROM snapshot_members sm
                            WHERE sm.snapshot_id=?
                              AND sm.content_hash=c.content_hash
                     )
                """,
                (cfg.parser_version, base_snapshot_id),
            ).fetchone()["n"])

            if current_count > 0 and history_count > 0 and history_replay_ratio > 0:
                wanted = math.ceil(
                    current_count
                    * history_replay_ratio
                    / (1.0 - history_replay_ratio)
                )
                replay_count = min(wanted, history_count)
            replay_sql = CURRENT_CTE + """
                SELECT c.content_hash,c.source_path,c.canonical_path,
                       'replay' AS role,1.0 AS weight
                  FROM current c
                 WHERE EXISTS (
                       SELECT 1
                         FROM snapshot_members sm
                        WHERE sm.snapshot_id=?
                          AND sm.content_hash=c.content_hash
                 )
                 ORDER BY stable_score(c.content_hash), c.content_hash
                 LIMIT ?
            """
            replay_params = (
                cfg.parser_version,
                base_snapshot_id,
                replay_count,
            )

        total = current_count + replay_count
        if total <= 0:
            raise ValueError("No eligible canonical documents for snapshot")

        def iter_members():
            for row in conn.execute(current_sql, current_params):
                yield dict(row)
            if replay_sql is not None and replay_count:
                for row in conn.execute(replay_sql, replay_params):
                    yield dict(row)

        digest_builder = hashlib.sha256()
        for member in iter_members():
            digest_builder.update(
                (
                    f"{member['content_hash']}|{member['role']}|"
                    f"{member['canonical_path']}\n"
                ).encode("utf-8")
            )
        digest = digest_builder.hexdigest()[:10]

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_id = f"{ts}-{digest}"
        snapshot_dir = cfg.snapshots_dir / snapshot_id
        if snapshot_dir.exists():
            raise FileExistsError(f"Snapshot already exists: {snapshot_dir}")
        snapshot_dir.mkdir(parents=True)
        manifest_path = (snapshot_dir / "manifest.jsonl").resolve()

        with manifest_path.open("w", encoding="utf-8") as out:
            for index, member in enumerate(iter_members(), start=1):
                if index % 5000 == 0:
                    wait_for_disk(
                        cfg.output_dir,
                        policy,
                        required_bytes=512 * MIB,
                        poll_seconds=2.0,
                        stable_samples=1,
                    )
                out.write(json.dumps(member, ensure_ascii=False) + "\n")

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
                        "total": total,
                        "current": current_count,
                        "replay": replay_count,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        conn.execute(
            """INSERT INTO snapshots(
                   snapshot_id,name,created_at,mode,base_snapshot_id,
                   manifest_path,config_json
               ) VALUES(?,?,?,?,?,?,?)""",
            (
                snapshot_id,
                name,
                utcnow(),
                mode,
                base_snapshot_id,
                str(manifest_path),
                json.dumps(config, ensure_ascii=False),
            ),
        )

        batch: list[tuple] = []
        with manifest_path.open("r", encoding="utf-8") as manifest:
            for line in manifest:
                if not line.strip():
                    continue
                member = json.loads(line)
                batch.append((
                    snapshot_id,
                    member["content_hash"],
                    member["source_path"],
                    member["canonical_path"],
                    member["role"],
                    float(member["weight"]),
                ))
                if len(batch) >= 1000:
                    wait_for_disk(
                        cfg.output_dir,
                        policy,
                        required_bytes=512 * MIB,
                        poll_seconds=2.0,
                        stable_samples=1,
                    )
                    conn.executemany(
                        """INSERT INTO snapshot_members(
                               snapshot_id,content_hash,source_path,canonical_path,role,weight
                           ) VALUES(?,?,?,?,?,?)""",
                        batch,
                    )
                    conn.commit()
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    batch.clear()
        if batch:
            wait_for_disk(
                cfg.output_dir,
                policy,
                required_bytes=512 * MIB,
                poll_seconds=2.0,
                stable_samples=1,
            )
            conn.executemany(
                """INSERT INTO snapshot_members(
                       snapshot_id,content_hash,source_path,canonical_path,role,weight
                   ) VALUES(?,?,?,?,?,?)""",
                batch,
            )
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")


    return SnapshotResult(
        snapshot_id=snapshot_id,
        manifest_path=manifest_path,
        total=total,
        current=current_count,
        replay=replay_count,
    )
