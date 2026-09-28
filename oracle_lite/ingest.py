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
from .memory import GIB, process_tree_rss
from .resources import HostResourcePolicy, wait_for_disk, wait_for_ram


IngestProgress = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class IngestStats:
    ready: int = 0
    skipped: int = 0
    failed: int = 0
    resource_pauses: int = 0
    resource_retries: int = 0
    visual_documents: int = 0
    visual_segments: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "ready": self.ready,
            "skipped": self.skipped,
            "failed": self.failed,
            "resource_pauses": self.resource_pauses,
            "resource_retries": self.resource_retries,
            "visual_documents": self.visual_documents,
            "visual_segments": self.visual_segments,
        }


def ingest_corpus(
    cfg: AppConfig,
    *,
    force: bool = False,
    progress: IngestProgress | None = None,
) -> IngestStats:
    """Parse one source at a time with admission control and no forced worker kill.

    Oracle-Lite waits before starting memory/disk-heavy work. Parser workers have
    a Linux address-space ceiling. If a worker naturally reports MemoryError or
    ENOSPC, the source is not marked failed: the pipeline waits, lowers the parser
    footprint, and retries the same source.
    """
    registry = Registry(cfg.registry_path)
    stats = IngestStats()
    policy = HostResourcePolicy.auto(cfg.output_dir)
    ctx = mp.get_context("spawn")

    def emit(**extra: Any) -> None:
        if progress is None:
            return
        payload: dict[str, Any] = stats.as_dict()
        payload.update(policy.as_dict())
        payload.update(extra)
        progress(payload)

    def wait_resources(source_path: Path, reason: str, attempt: int) -> None:
        stats.resource_pauses += 1

        def on_wait(state: dict) -> None:
            emit(
                current_parse_file=str(source_path),
                parse_state=(
                    "waiting_for_disk"
                    if state.get("resource") == "disk"
                    else "waiting_for_memory"
                ),
                resource_reason=reason,
                retry_attempt=attempt,
                memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                disk_free_gb=round(
                    shutil.disk_usage(cfg.output_dir).free / GIB, 2
                ),
                resource_state=state,
            )

        wait_for_ram(
            policy,
            on_wait=on_wait,
            poll_seconds=1.0,
            stable_samples=3,
        )
        # Reserve a bounded workspace for one source. Internal worker checks gate
        # each page/media write again, so this is only the outer admission gate.
        required_disk = max(2 * GIB, min(20 * GIB, source_path.stat().st_size * 4))
        wait_for_disk(
            cfg.output_dir,
            policy,
            required_bytes=required_disk,
            on_wait=on_wait,
            poll_seconds=2.0,
            stable_samples=2,
        )

    emit(stage="ingest_start")

    for row in registry.iter_active_unique_content():
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
                    disk_free_gb=round(shutil.disk_usage(cfg.output_dir).free / GIB, 2),
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
        memory_level = 0

        while True:
            attempt += 1
            gc.collect()
            wait_resources(source_path, "resource admission before parser start", attempt)

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
                    "address_space_limit_bytes": policy.parser_address_space_bytes,
                    "low_memory": memory_level > 0,
                    "memory_level": memory_level,
                    "output_dir": str(cfg.output_dir),
                    "disk_reserve_bytes": policy.disk_reserve_bytes,
                    "disk_resume_bytes": policy.disk_resume_bytes,
                },
                name=f"oracle-ingest-{content_hash[:8]}",
            )

            emit(
                current_parse_file=str(source_path),
                parse_state="starting_low_memory" if memory_level > 0 else "starting",
                retry_attempt=attempt,
                low_memory_mode=memory_level > 0,
                memory_level=memory_level,
                worker_rss_gb=0.0,
                memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                disk_free_gb=round(shutil.disk_usage(cfg.output_dir).free / GIB, 2),
            )
            proc.start()

            # Important: no terminate(), kill(), SIGKILL or resource-pressure
            # intervention here. The worker either completes or is denied an
            # allocation by its Linux RLIMIT_AS and reports a retryable condition.
            while proc.is_alive():
                proc.join(timeout=0.25)
                emit(
                    current_parse_file=str(source_path),
                    parse_state="parsing_low_memory" if memory_level > 0 else "parsing",
                    retry_attempt=attempt,
                    low_memory_mode=memory_level > 0,
                    memory_level=memory_level,
                    worker_rss_gb=round(process_tree_rss(proc.pid or -1) / GIB, 2),
                    memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                    disk_free_gb=round(shutil.disk_usage(cfg.output_dir).free / GIB, 2),
                )

            try:
                result = result_queue.get(timeout=2.0)
            except Empty:
                result = None
            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception:
                pass

            retryable_resource = bool(
                result
                and not result.get("ok")
                and result.get("kind") in {"memory", "disk"}
            )

            if retryable_resource or result is None:
                stats.resource_retries += 1
                reason = (
                    str(result.get("error"))
                    if result
                    else f"worker exited without result (exitcode={proc.exitcode})"
                )
                tmp_path.unlink(missing_ok=True)
                sidecar_tmp.unlink(missing_ok=True)
                shutil.rmtree(asset_dir, ignore_errors=True)
                memory_level = min(4, memory_level + 1)
                emit(
                    current_parse_file=str(source_path),
                    parse_state="deferred_resource",
                    resource_reason=reason,
                    retry_attempt=attempt,
                    low_memory_mode=True,
                    memory_level=memory_level,
                    worker_rss_gb=0.0,
                    memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                    disk_free_gb=round(shutil.disk_usage(cfg.output_dir).free / GIB, 2),
                )
                wait_resources(source_path, reason, attempt)
                continue

            if not result.get("ok"):
                # Genuine corruption/unsupported-content errors remain real parser
                # failures. Resource pressure never reaches this branch.
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
                    low_memory_mode=memory_level > 0,
                    memory_level=memory_level,
                    worker_rss_gb=0.0,
                    memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                    disk_free_gb=round(shutil.disk_usage(cfg.output_dir).free / GIB, 2),
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
                low_memory_mode=memory_level > 0,
                memory_level=memory_level,
                worker_rss_gb=0.0,
                memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                disk_free_gb=round(shutil.disk_usage(cfg.output_dir).free / GIB, 2),
            )
            gc.collect()
            break

    emit(
        stage="ingest_complete",
        current_parse_file=None,
        parse_state="complete",
        worker_rss_gb=0.0,
        memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
        disk_free_gb=round(shutil.disk_usage(cfg.output_dir).free / GIB, 2),
    )
    return stats
