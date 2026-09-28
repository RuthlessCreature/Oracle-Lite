from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .db import Registry, utcnow
from .hashutil import hash_file
from .resources import GIB, HostResourcePolicy, wait_for_disk


ScanProgress = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class ScanStats:
    files_seen: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    duplicates: int = 0
    tombstoned: int = 0
    hashed: int = 0
    bytes_hashed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "files_seen": self.files_seen,
            "new": self.new,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "duplicates": self.duplicates,
            "tombstoned": self.tombstoned,
            "hashed": self.hashed,
            "bytes_hashed": self.bytes_hashed,
        }


def _iter_files(root: Path, cfg: AppConfig):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in cfg.ignore_names for part in path.parts):
            continue
        if path.suffix.lower() not in cfg.include_extensions:
            continue
        yield path


def _mark_missing_streaming(
    registry: Registry,
    *,
    root: Path,
    seen_conn: sqlite3.Connection,
) -> int:
    """Tombstone missing paths without materializing all paths in RAM."""
    root_s = str(root.resolve())
    missing = 0
    batch: list[tuple[str, int]] = []
    with registry.connect() as conn:
        cursor = conn.execute(
            "SELECT id,path FROM source_files WHERE root=? AND status='active' ORDER BY id",
            (root_s,),
        )
        for row in cursor:
            exists = seen_conn.execute(
                "SELECT 1 FROM seen_paths WHERE path=? LIMIT 1",
                (row["path"],),
            ).fetchone()
            if exists:
                continue
            batch.append((utcnow(), int(row["id"])))
            missing += 1
            if len(batch) >= 500:
                conn.executemany(
                    "UPDATE source_files SET status='tombstoned',last_seen_at=? WHERE id=?",
                    batch,
                )
                batch.clear()
        if batch:
            conn.executemany(
                "UPDATE source_files SET status='tombstoned',last_seen_at=? WHERE id=?",
                batch,
            )
    return missing


def scan_corpus(
    cfg: AppConfig,
    *,
    verify_all: bool = False,
    progress: ScanProgress | None = None,
) -> ScanStats:
    registry = Registry(cfg.registry_path)
    stats = ScanStats()
    resource_policy = HostResourcePolicy.auto(cfg.output_dir)

    def emit(**extra: Any) -> None:
        if progress is None:
            return
        payload: dict[str, Any] = stats.as_dict()
        payload.update(extra)
        progress(payload)

    for root in cfg.corpus_roots:
        root = root.expanduser().resolve()
        temp_db = cfg.state_dir / f".scan-seen-{uuid.uuid4().hex}.sqlite3"
        seen_conn = sqlite3.connect(temp_db)
        try:
            seen_conn.execute(
                "CREATE TABLE seen_paths(path TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            seen_conn.commit()
            emit(current_root=str(root), current_file=None, hashing=False)

            pending_seen = 0
            for path in _iter_files(root, cfg) or []:
                resolved = str(path.resolve())
                seen_conn.execute(
                    "INSERT OR IGNORE INTO seen_paths(path) VALUES(?)",
                    (resolved,),
                )
                pending_seen += 1
                if pending_seen >= 500:
                    seen_conn.commit()
                    pending_seen = 0
                    wait_for_disk(
                        cfg.output_dir,
                        resource_policy,
                        required_bytes=512 * 1024 * 1024,
                        poll_seconds=2.0,
                        stable_samples=1,
                    )

                stats.files_seen += 1
                st = path.stat()
                previous = registry.get_source(path)

                emit(
                    current_root=str(root),
                    current_file=resolved,
                    current_file_size=st.st_size,
                    current_file_hashed_bytes=0,
                    hashing=False,
                )

                if (
                    previous is not None
                    and not verify_all
                    and previous["status"] == "active"
                    and previous["size"] == st.st_size
                    and previous["mtime_ns"] == st.st_mtime_ns
                ):
                    # Intelligent restart path: metadata match means zero content I/O.
                    content_hash = previous["content_hash"]
                else:
                    def on_hash(processed: int, total: int) -> None:
                        emit(
                            current_root=str(root),
                            current_file=resolved,
                            current_file_size=total,
                            current_file_hashed_bytes=processed,
                            hashing=True,
                        )

                    content_hash = hash_file(
                        path,
                        cfg.hash_algorithm,
                        progress=on_hash,
                    )
                    stats.hashed += 1
                    stats.bytes_hashed += st.st_size

                has_duplicate = registry.has_other_active_path_for_hash(
                    content_hash,
                    resolved,
                )
                state = registry.upsert_source(
                    root=root,
                    path=path,
                    size=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                    content_hash=content_hash,
                )

                if has_duplicate:
                    stats.duplicates += 1

                if state == "new":
                    stats.new += 1
                elif state == "changed":
                    stats.changed += 1
                else:
                    stats.unchanged += 1

                emit(
                    current_root=str(root),
                    current_file=resolved,
                    current_file_size=st.st_size,
                    current_file_hashed_bytes=st.st_size if state != "unchanged" else 0,
                    hashing=False,
                )

            seen_conn.commit()
            stats.tombstoned += _mark_missing_streaming(
                registry,
                root=root,
                seen_conn=seen_conn,
            )
        finally:
            seen_conn.close()
            temp_db.unlink(missing_ok=True)

    emit(current_file=None, hashing=False, complete=True)
    return stats
