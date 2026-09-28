from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_source_hash ON source_files(content_hash);
CREATE INDEX IF NOT EXISTS idx_source_status ON source_files(status);

CREATE TABLE IF NOT EXISTS source_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file_id INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    discovered_at TEXT NOT NULL,
    FOREIGN KEY(source_file_id) REFERENCES source_files(id)
);

CREATE TABLE IF NOT EXISTS content_objects (
    content_hash TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS derived_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    canonical_path TEXT,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(content_hash, parser_version),
    FOREIGN KEY(content_hash) REFERENCES content_objects(content_hash)
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    base_snapshot_id TEXT,
    manifest_path TEXT NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshot_members (
    snapshot_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_path TEXT NOT NULL,
    canonical_path TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'current',
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY(snapshot_id, content_hash, role),
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    config_path TEXT,
    output_dir TEXT,
    started_at TEXT,
    ended_at TEXT,
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);
"""


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        # Keep SQLite memory predictable for very large corpora. Expensive
        # GROUP BY / EXCEPT / ORDER BY work spills to disk instead of RAM.
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA cache_size=-65536")
        conn.execute("PRAGMA mmap_size=0")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_source(self, path: str | Path):
        with self.connect() as conn:
            return conn.execute("SELECT * FROM source_files WHERE path=?", (str(Path(path).resolve()),)).fetchone()

    def upsert_source(self, *, root: str | Path, path: str | Path, size: int, mtime_ns: int, content_hash: str) -> str:
        root_s = str(Path(root).resolve())
        path_s = str(Path(path).resolve())
        now = utcnow()
        with self.connect() as conn:
            existing = conn.execute("SELECT * FROM source_files WHERE path=?", (path_s,)).fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO content_objects(content_hash,size,first_seen_at) VALUES(?,?,?)",
                (content_hash, size, now),
            )
            if existing is None:
                cur = conn.execute(
                    """INSERT INTO source_files(root,path,size,mtime_ns,content_hash,status,first_seen_at,last_seen_at)
                       VALUES(?,?,?,?,?,'active',?,?)""",
                    (root_s, path_s, size, mtime_ns, content_hash, now, now),
                )
                source_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO source_revisions(source_file_id,content_hash,size,mtime_ns,discovered_at) VALUES(?,?,?,?,?)",
                    (source_id, content_hash, size, mtime_ns, now),
                )
                return "new"

            changed = existing["content_hash"] != content_hash
            conn.execute(
                """UPDATE source_files SET root=?,size=?,mtime_ns=?,content_hash=?,status='active',last_seen_at=?
                   WHERE id=?""",
                (root_s, size, mtime_ns, content_hash, now, existing["id"]),
            )
            if changed:
                conn.execute(
                    "INSERT INTO source_revisions(source_file_id,content_hash,size,mtime_ns,discovered_at) VALUES(?,?,?,?,?)",
                    (existing["id"], content_hash, size, mtime_ns, now),
                )
                return "changed"
            return "unchanged"

    def mark_missing_under_root(self, root: str | Path, seen_paths: set[str]) -> int:
        root_s = str(Path(root).resolve())
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id,path FROM source_files WHERE root=? AND status='active'", (root_s,)
            ).fetchall()
            missing = [r["id"] for r in rows if r["path"] not in seen_paths]
            if missing:
                conn.executemany(
                    "UPDATE source_files SET status='tombstoned',last_seen_at=? WHERE id=?",
                    [(utcnow(), file_id) for file_id in missing],
                )
            return len(missing)

    def active_paths_for_hash(self, content_hash: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT path FROM source_files WHERE content_hash=? AND status='active' ORDER BY path",
                (content_hash,),
            ).fetchall()
            return [r["path"] for r in rows]

    def has_other_active_path_for_hash(
        self,
        content_hash: str,
        path: str | Path,
    ) -> bool:
        path_s = str(Path(path).resolve())
        with self.connect() as conn:
            row = conn.execute(
                """SELECT 1
                     FROM source_files
                    WHERE content_hash=?
                      AND status='active'
                      AND path<>?
                    LIMIT 1""",
                (content_hash, path_s),
            ).fetchone()
            return row is not None

    def list_active_unique_content(self):
        # Backward-compatible small-data helper. Prefer iter_active_unique_content()
        # in memory-sensitive pipeline code.
        with self.connect() as conn:
            return conn.execute(
                """SELECT sf.content_hash, MIN(sf.path) AS source_path, MAX(sf.size) AS size
                   FROM source_files sf
                   WHERE sf.status='active'
                   GROUP BY sf.content_hash
                   ORDER BY source_path"""
            ).fetchall()

    def iter_active_unique_content(self):
        with self.connect() as conn:
            cursor = conn.execute(
                """SELECT sf.content_hash, MIN(sf.path) AS source_path, MAX(sf.size) AS size
                   FROM source_files sf
                   WHERE sf.status='active'
                   GROUP BY sf.content_hash
                   ORDER BY source_path"""
            )
            for row in cursor:
                yield row

    def get_artifact(self, content_hash: str, parser_version: str):
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM derived_artifacts WHERE content_hash=? AND parser_version=?",
                (content_hash, parser_version),
            ).fetchone()

    def save_artifact(self, *, content_hash: str, parser_version: str, canonical_path: str | None, status: str, error: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO derived_artifacts(content_hash,parser_version,canonical_path,status,error,created_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(content_hash,parser_version) DO UPDATE SET
                     canonical_path=excluded.canonical_path,
                     status=excluded.status,
                     error=excluded.error,
                     created_at=excluded.created_at""",
                (content_hash, parser_version, canonical_path, status, error, utcnow()),
            )

    def save_snapshot(self, *, snapshot_id: str, name: str, mode: str, base_snapshot_id: str | None,
                      manifest_path: str, config: dict, members: list[dict]) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO snapshots(snapshot_id,name,created_at,mode,base_snapshot_id,manifest_path,config_json)
                   VALUES(?,?,?,?,?,?,?)""",
                (snapshot_id, name, utcnow(), mode, base_snapshot_id, manifest_path, json.dumps(config, ensure_ascii=False)),
            )
            conn.executemany(
                """INSERT INTO snapshot_members(snapshot_id,content_hash,source_path,canonical_path,role,weight)
                   VALUES(?,?,?,?,?,?)""",
                [
                    (snapshot_id, m["content_hash"], m["source_path"], m["canonical_path"],
                     m.get("role", "current"), float(m.get("weight", 1.0)))
                    for m in members
                ],
            )

    def snapshot_hashes(self, snapshot_id: str) -> set[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT content_hash FROM snapshot_members WHERE snapshot_id=?", (snapshot_id,)
            ).fetchall()
            return {r["content_hash"] for r in rows}

    def get_snapshot(self, snapshot_id: str):
        with self.connect() as conn:
            return conn.execute("SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()

    def active_ready_hashes(self, parser_version: str) -> set[str]:
        # Compatibility helper; one-click pipeline uses SQL matching below.
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT sf.content_hash
                   FROM source_files sf
                   JOIN derived_artifacts da
                     ON da.content_hash=sf.content_hash
                    AND da.parser_version=?
                   WHERE sf.status='active'
                     AND da.status='ready'
                     AND da.canonical_path IS NOT NULL""",
                (parser_version,),
            ).fetchall()
            return {row["content_hash"] for row in rows}

    def active_ready_count(self, parser_version: str) -> int:
        with self.connect() as conn:
            return int(conn.execute(
                """SELECT COUNT(DISTINCT sf.content_hash) AS n
                   FROM source_files sf
                   JOIN derived_artifacts da
                     ON da.content_hash=sf.content_hash
                    AND da.parser_version=?
                   WHERE sf.status='active'
                     AND da.status='ready'
                     AND da.canonical_path IS NOT NULL""",
                (parser_version,),
            ).fetchone()["n"])

    def snapshot_matches_active_ready(
        self,
        snapshot_id: str,
        parser_version: str,
    ) -> bool:
        """Compare snapshot/current content entirely in SQLite, without Python sets."""
        with self.connect() as conn:
            active_minus_snapshot = conn.execute(
                """SELECT 1
                     FROM (
                         SELECT DISTINCT sf.content_hash
                           FROM source_files sf
                           JOIN derived_artifacts da
                             ON da.content_hash=sf.content_hash
                            AND da.parser_version=?
                          WHERE sf.status='active'
                            AND da.status='ready'
                            AND da.canonical_path IS NOT NULL
                         EXCEPT
                         SELECT DISTINCT content_hash
                           FROM snapshot_members
                          WHERE snapshot_id=?
                     )
                    LIMIT 1""",
                (parser_version, snapshot_id),
            ).fetchone()
            if active_minus_snapshot is not None:
                return False

            snapshot_minus_active = conn.execute(
                """SELECT 1
                     FROM (
                         SELECT DISTINCT content_hash
                           FROM snapshot_members
                          WHERE snapshot_id=?
                         EXCEPT
                         SELECT DISTINCT sf.content_hash
                           FROM source_files sf
                           JOIN derived_artifacts da
                             ON da.content_hash=sf.content_hash
                            AND da.parser_version=?
                          WHERE sf.status='active'
                            AND da.status='ready'
                            AND da.canonical_path IS NOT NULL
                     )
                    LIMIT 1""",
                (snapshot_id, parser_version),
            ).fetchone()
            return snapshot_minus_active is None

    def latest_snapshot(self, mode: str | None = None):
        with self.connect() as conn:
            if mode is None:
                return conn.execute(
                    "SELECT * FROM snapshots ORDER BY created_at DESC, snapshot_id DESC LIMIT 1"
                ).fetchone()
            return conn.execute(
                """SELECT * FROM snapshots
                   WHERE mode=?
                   ORDER BY created_at DESC, snapshot_id DESC
                   LIMIT 1""",
                (mode,),
            ).fetchone()

    def latest_training_run(self, snapshot_id: str, kind: str | None = None):
        with self.connect() as conn:
            if kind is None:
                return conn.execute(
                    """SELECT * FROM training_runs
                       WHERE snapshot_id=?
                       ORDER BY started_at DESC, run_id DESC
                       LIMIT 1""",
                    (snapshot_id,),
                ).fetchone()
            return conn.execute(
                """SELECT * FROM training_runs
                   WHERE snapshot_id=? AND kind=?
                   ORDER BY started_at DESC, run_id DESC
                   LIMIT 1""",
                (snapshot_id, kind),
            ).fetchone()
