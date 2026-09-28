from __future__ import annotations

import gc
import multiprocessing as mp
import shutil
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable

import psutil

from .config import AppConfig
from .db import Registry
from .ingest_worker import parse_and_write_canonical
from .memory import GIB, MemoryPolicy, process_tree_rss


IngestProgress = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class IngestStats:
    ready: int = 0
    skipped: int = 0
    failed: int = 0
    visual_documents: int = 0
    visual_segments: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "ready": self.ready,
            "skipped": self.skipped,
            "failed": self.failed,
            "visual_documents": self.visual_documents,
            "visual_segments": self.visual_segments,
        }


def ingest_corpus(
    cfg: AppConfig,
    *,
    force: bool = False,
    progress: IngestProgress | None = None,
) -> IngestStats:
    """Parse corpus safely, one isolated source process at a time.

    A pathological PDF/PPT/JSON can no longer consume the entire workstation:
    each source is parsed in a child process whose RSS is monitored. If either
    the worker exceeds its internal budget or system available memory falls
    below the reserved safety margin, only that worker is terminated.
    """
    registry = Registry(cfg.registry_path)
    stats = IngestStats()
    policy = MemoryPolicy.auto()
    ctx = mp.get_context("spawn")

    def emit(**extra: Any) -> None:
        if progress is None:
            return
        payload: dict[str, Any] = stats.as_dict()
        payload.update(policy.as_dict())
        payload.update(extra)
        progress(payload)

    emit(stage="ingest_start")

    for row in registry.list_active_unique_content():
        content_hash = row["content_hash"]
        source_path = Path(row["source_path"])
        existing = registry.get_artifact(content_hash, cfg.parser_version)

        if existing is not None and existing["status"] == "ready" and not force:
            canonical_path = existing["canonical_path"]
            if canonical_path and Path(canonical_path).exists():
                stats.skipped += 1
                emit(
                    current_parse_file=str(source_path),
                    parse_state="cached",
                    worker_rss_gb=0.0,
                    memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                )
                continue

        canonical_path = (
            cfg.canonical_dir
            / content_hash[:2]
            / f"{content_hash}.{cfg.parser_version}.json"
        ).resolve()
        asset_dir = (cfg.assets_dir / content_hash[:2] / content_hash / cfg.parser_version).resolve()

        gc.collect()
        available = int(psutil.virtual_memory().available)
        if available < policy.reserve_system_bytes:
            error = (
                "MemoryPressureError: available system memory "
                f"{available / GIB:.2f} GB is below Oracle-Lite reserve "
                f"{policy.reserve_system_bytes / GIB:.2f} GB before parsing "
                f"{source_path}"
            )
            registry.save_artifact(
                content_hash=content_hash,
                parser_version=cfg.parser_version,
                canonical_path=None,
                status="failed",
                error=error,
            )
            stats.failed += 1
            emit(
                current_parse_file=str(source_path),
                parse_state="memory_blocked",
                worker_rss_gb=0.0,
                memory_available_gb=round(available / GIB, 2),
                error=error,
            )
            continue

        # Remove leftovers from an interrupted/failed previous parse. Canonical
        # artifacts are immutable once marked ready, so this only affects work
        # that never completed successfully.
        tmp_path = canonical_path.with_suffix(canonical_path.suffix + ".tmp")
        tmp_path.unlink(missing_ok=True)
        if asset_dir.exists():
            shutil.rmtree(asset_dir, ignore_errors=True)

        result_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=parse_and_write_canonical,
            kwargs={
                "source_path": str(source_path),
                "asset_dir": str(asset_dir),
                "canonical_path": str(canonical_path),
                "content_hash": content_hash,
                "parser_version": cfg.parser_version,
                "result_queue": result_queue,
            },
            name=f"oracle-ingest-{content_hash[:8]}",
        )

        emit(
            current_parse_file=str(source_path),
            parse_state="starting",
            worker_rss_gb=0.0,
            memory_available_gb=round(available / GIB, 2),
        )
        proc.start()

        memory_error: str | None = None
        while proc.is_alive():
            proc.join(timeout=0.20)
            worker_rss = process_tree_rss(proc.pid or -1)
            available = int(psutil.virtual_memory().available)

            emit(
                current_parse_file=str(source_path),
                parse_state="parsing",
                worker_rss_gb=round(worker_rss / GIB, 2),
                memory_available_gb=round(available / GIB, 2),
            )

            if worker_rss > policy.max_worker_rss_bytes:
                memory_error = (
                    "MemoryPressureError: parser worker exceeded safe RSS cap "
                    f"({worker_rss / GIB:.2f} GB > "
                    f"{policy.max_worker_rss_bytes / GIB:.2f} GB) while parsing "
                    f"{source_path}"
                )
            elif available < policy.reserve_system_bytes:
                memory_error = (
                    "MemoryPressureError: system available memory fell below "
                    f"reserve ({available / GIB:.2f} GB < "
                    f"{policy.reserve_system_bytes / GIB:.2f} GB) while parsing "
                    f"{source_path}"
                )

            if memory_error:
                proc.terminate()
                proc.join(timeout=3.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)
                break

        result: dict[str, Any] | None = None
        if memory_error is None:
            try:
                result = result_queue.get(timeout=2.0)
            except Empty:
                result = None

        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

        if memory_error is not None:
            error = memory_error
        elif result is None:
            error = (
                "ParserWorkerError: parser process exited without a result "
                f"(exitcode={proc.exitcode}) for {source_path}"
            )
        elif not result.get("ok"):
            error = str(result.get("error") or "ParserWorkerError: unknown parser failure")
        else:
            error = None

        if error is not None:
            tmp_path.unlink(missing_ok=True)
            registry.save_artifact(
                content_hash=content_hash,
                parser_version=cfg.parser_version,
                canonical_path=None,
                status="failed",
                error=error,
            )
            stats.failed += 1
            emit(
                current_parse_file=str(source_path),
                parse_state="failed",
                worker_rss_gb=0.0,
                memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                error=error,
            )
            continue

        registry.save_artifact(
            content_hash=content_hash,
            parser_version=cfg.parser_version,
            canonical_path=str(canonical_path),
            status="ready",
        )
        stats.ready += 1
        visual_segments = int(result.get("visual_segments", 0))
        if visual_segments:
            stats.visual_documents += 1
            stats.visual_segments += visual_segments

        emit(
            current_parse_file=str(source_path),
            parse_state="ready",
            worker_rss_gb=0.0,
            memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
        )
        gc.collect()

    emit(
        stage="ingest_complete",
        current_parse_file=None,
        parse_state="complete",
        worker_rss_gb=0.0,
        memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
    )
    return stats
