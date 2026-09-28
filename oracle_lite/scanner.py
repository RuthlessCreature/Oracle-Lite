from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .db import Registry
from .hashutil import hash_file


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


def scan_corpus(
    cfg: AppConfig,
    *,
    verify_all: bool = False,
    progress: ScanProgress | None = None,
) -> ScanStats:
    registry = Registry(cfg.registry_path)
    stats = ScanStats()

    def emit(**extra: Any) -> None:
        if progress is None:
            return
        payload: dict[str, Any] = stats.as_dict()
        payload.update(extra)
        progress(payload)

    for root in cfg.corpus_roots:
        root = root.expanduser().resolve()
        seen: set[str] = set()
        emit(current_root=str(root), current_file=None, hashing=False)

        for path in _iter_files(root, cfg) or []:
            resolved = str(path.resolve())
            seen.add(resolved)
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

            other_paths = registry.active_paths_for_hash(content_hash)
            state = registry.upsert_source(
                root=root,
                path=path,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
                content_hash=content_hash,
            )

            if any(p != resolved for p in other_paths):
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

        stats.tombstoned += registry.mark_missing_under_root(root, seen)

    emit(current_file=None, hashing=False, complete=True)
    return stats
