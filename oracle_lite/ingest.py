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
from .memory import GIB, MemoryPolicy, process_tree_rss, wait_for_safe_memory


IngestProgress = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class IngestStats:
    ready: int = 0
    skipped: int = 0
    failed: int = 0
    memory_pauses: int = 0
    memory_retries: int = 0
    visual_documents: int = 0
    visual_segments: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "ready": self.ready,
            "skipped": self.skipped,
            "failed": self.failed,
            "memory_pauses": self.memory_pauses,
            "memory_retries": self.memory_retries,
            "visual_documents": self.visual_documents,
            "visual_segments": self.visual_segments,
        }


def ingest_corpus(
    cfg: AppConfig,
    *,
    force: bool = False,
    progress: IngestProgress | None = None,
) -> IngestStats:
    """Parse corpus with backpressure instead of host-RAM exhaustion.

    Memory pressure is never recorded as a failed artifact. The current worker is
    released, the pipeline enters WAITING_FOR_MEMORY, and the same file is retried
    after host RAM has recovered. A low-memory parser mode is enabled after the
    first memory-pressure restart.
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

    def wait_memory(source_path: Path, reason: str, attempt: int) -> None:
        stats.memory_pauses += 1

        def on_wait(state: dict) -> None:
            emit(
                current_parse_file=str(source_path),
                parse_state="waiting_for_memory",
                memory_reason=reason,
                retry_attempt=attempt,
                worker_rss_gb=0.0,
                memory_available_gb=state["available_gb"],
            )

        wait_for_safe_memory(policy, on_wait=on_wait, poll_seconds=1.0, stable_samples=3)

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
        asset_dir = (
            cfg.assets_dir
            / content_hash[:2]
            / content_hash
            / cfg.parser_version
        ).resolve()

        attempt = 0
        low_memory = False

        while True:
            attempt += 1
            gc.collect()

            available = int(psutil.virtual_memory().available)
            if available < policy.resume_system_bytes:
                wait_memory(
                    source_path,
                    (
                        f"host RAM is below resume threshold "
                        f"({available / GIB:.2f} GiB available)"
                    ),
                    attempt,
                )

            tmp_path = canonical_path.with_suffix(canonical_path.suffix + ".tmp")
            sidecar_tmp = canonical_path.with_suffix(
                canonical_path.suffix + ".segments.jsonl.tmp"
            )
            tmp_path.unlink(missing_ok=True)
            sidecar_tmp.unlink(missing_ok=True)
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
                    "address_space_limit_bytes": policy.max_worker_address_space_bytes,
                    "low_memory": low_memory,
                },
                name=f"oracle-ingest-{content_hash[:8]}",
            )

            emit(
                current_parse_file=str(source_path),
                parse_state="starting_low_memory" if low_memory else "starting",
                retry_attempt=attempt,
                low_memory_mode=low_memory,
                worker_rss_gb=0.0,
                memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
            )
            proc.start()

            memory_reason: str | None = None
            while proc.is_alive():
                proc.join(timeout=0.20)
                worker_rss = process_tree_rss(proc.pid or -1)
                available = int(psutil.virtual_memory().available)

                emit(
                    current_parse_file=str(source_path),
                    parse_state="parsing_low_memory" if low_memory else "parsing",
                    retry_attempt=attempt,
                    low_memory_mode=low_memory,
                    worker_rss_gb=round(worker_rss / GIB, 2),
                    memory_available_gb=round(available / GIB, 2),
                )

                if worker_rss > policy.max_worker_rss_bytes:
                    memory_reason = (
                        f"parser RSS {worker_rss / GIB:.2f} GiB exceeded "
                        f"{policy.max_worker_rss_bytes / GIB:.2f} GiB cap"
                    )
                elif available < policy.reserve_system_bytes:
                    memory_reason = (
                        f"host available RAM {available / GIB:.2f} GiB fell below "
                        f"{policy.reserve_system_bytes / GIB:.2f} GiB reserve"
                    )

                if memory_reason:
                    proc.terminate()
                    proc.join(timeout=2.0)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=1.0)
                    break

            result: dict[str, Any] | None = None
            if memory_reason is None:
                try:
                    result = result_queue.get(timeout=2.0)
                except Empty:
                    result = None

            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception:
                pass

            worker_reported_memory = bool(
                result
                and not result.get("ok")
                and result.get("kind") == "memory"
            )

            if memory_reason is not None or worker_reported_memory:
                stats.memory_retries += 1
                reason = memory_reason or str(result.get("error") if result else "memory pressure")
                tmp_path.unlink(missing_ok=True)
                sidecar_tmp.unlink(missing_ok=True)
                shutil.rmtree(asset_dir, ignore_errors=True)
                emit(
                    current_parse_file=str(source_path),
                    parse_state="deferred_memory",
                    memory_reason=reason,
                    retry_attempt=attempt,
                    low_memory_mode=low_memory,
                    worker_rss_gb=0.0,
                    memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                )

                low_memory = True
                wait_memory(source_path, reason, attempt)
                continue

            if result is None:
                stats.memory_retries += 1
                reason = f"worker exited without result (exitcode={proc.exitcode})"
                low_memory = True
                wait_memory(source_path, reason, attempt)
                continue

            if not result.get("ok"):
                error = str(result.get("error") or "ParserWorkerError: unknown parser failure")
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
                    retry_attempt=attempt,
                    low_memory_mode=low_memory,
                    worker_rss_gb=0.0,
                    memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                    error=error,
                )
                break

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
                retry_attempt=attempt,
                low_memory_mode=low_memory,
                worker_rss_gb=0.0,
                memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
            )
            gc.collect()
            break

    emit(
        stage="ingest_complete",
        current_parse_file=None,
        parse_state="complete",
        worker_rss_gb=0.0,
        memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
    )
    return stats
