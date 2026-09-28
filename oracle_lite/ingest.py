from __future__ import annotations

import gc
import multiprocessing as mp
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable

import psutil

from .config import AppConfig
from .db import Registry
from .ingest_worker import parse_and_write_canonical
from .memory import GIB, MemoryPolicy, MemoryPressureGate, process_tree_rss


IngestProgress = Callable[[dict[str, Any]], None]
MAX_MEMORY_RETRIES = 3


@dataclass(slots=True)
class IngestStats:
    ready: int = 0
    skipped: int = 0
    failed: int = 0
    deferred_memory: int = 0
    retried_memory: int = 0
    visual_documents: int = 0
    visual_segments: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "ready": self.ready,
            "skipped": self.skipped,
            "failed": self.failed,
            "deferred_memory": self.deferred_memory,
            "retried_memory": self.retried_memory,
            "visual_documents": self.visual_documents,
            "visual_segments": self.visual_segments,
        }


def _looks_like_memory_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "memoryerror" in lowered
        or "cannot allocate memory" in lowered
        or "errno 12" in lowered
        or "memorypressureerror" in lowered
    )


def ingest_corpus(
    cfg: AppConfig,
    *,
    force: bool = False,
    progress: IngestProgress | None = None,
    memory_gate: MemoryPressureGate | None = None,
) -> IngestStats:
    """Parse one source per isolated worker without treating RAM pressure as failure."""
    registry = Registry(cfg.registry_path)
    stats = IngestStats()
    policy = memory_gate.policy if memory_gate is not None else MemoryPolicy.auto()
    ctx = mp.get_context("spawn")

    def emit(**extra: Any) -> None:
        if progress is None:
            return
        payload: dict[str, Any] = stats.as_dict()
        payload.update(policy.as_dict())
        payload.update(extra)
        progress(payload)

    queue = deque((row, 0) for row in registry.list_active_unique_content())
    emit(stage="ingest_start", queued=len(queue))

    while queue:
        row, attempt = queue.popleft()
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
                    memory_tier=0,
                    worker_rss_gb=0.0,
                    memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                    queued=len(queue),
                )
                continue

        if memory_gate is not None:
            if memory_gate.paused:
                emit(
                    current_parse_file=str(source_path),
                    parse_state="paused_memory",
                    pause_reason=memory_gate.reason,
                    memory_tier=min(attempt, 2),
                    worker_rss_gb=0.0,
                    memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                    queued=len(queue) + 1,
                )
            memory_gate.wait_until_safe(context=f"parsing {source_path.name}")

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

        gc.collect()
        available = int(psutil.virtual_memory().available)
        if available < policy.reserve_system_bytes:
            reason = (
                f"Available RAM {available / GIB:.2f} GiB below reserve "
                f"{policy.reserve_system_bytes / GIB:.2f} GiB before parsing "
                f"{source_path.name}"
            )
            if memory_gate is not None:
                memory_gate.request_pause(reason)
                emit(
                    current_parse_file=str(source_path),
                    parse_state="paused_memory",
                    pause_reason=reason,
                    memory_tier=min(attempt, 2),
                    worker_rss_gb=0.0,
                    memory_available_gb=round(available / GIB, 2),
                    queued=len(queue) + 1,
                )
                memory_gate.wait_until_safe(context=f"parsing {source_path.name}")
                queue.appendleft((row, attempt))
                continue

        tmp_path = canonical_path.with_suffix(canonical_path.suffix + ".tmp")
        sidecar_tmp = canonical_path.with_suffix(
            canonical_path.suffix + ".segments.jsonl.tmp"
        )
        tmp_path.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
        if asset_dir.exists():
            shutil.rmtree(asset_dir, ignore_errors=True)

        memory_tier = min(attempt, 2)
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
                "memory_tier": memory_tier,
            },
            name=f"oracle-ingest-{content_hash[:8]}",
        )

        emit(
            current_parse_file=str(source_path),
            parse_state="starting",
            memory_tier=memory_tier,
            retry=attempt,
            worker_rss_gb=0.0,
            memory_available_gb=round(available / GIB, 2),
            queued=len(queue),
        )
        proc.start()

        pressure_reason: str | None = None
        while proc.is_alive():
            proc.join(timeout=0.20)
            worker_rss = process_tree_rss(proc.pid or -1)
            available = int(psutil.virtual_memory().available)

            emit(
                current_parse_file=str(source_path),
                parse_state="parsing",
                memory_tier=memory_tier,
                retry=attempt,
                worker_rss_gb=round(worker_rss / GIB, 2),
                memory_available_gb=round(available / GIB, 2),
                queued=len(queue),
            )

            if worker_rss > policy.max_worker_rss_bytes:
                pressure_reason = (
                    f"Parser worker reached {worker_rss / GIB:.2f} GiB RSS "
                    f"(cap {policy.max_worker_rss_bytes / GIB:.2f} GiB)"
                )
            elif available < policy.reserve_system_bytes:
                pressure_reason = (
                    f"Available RAM fell to {available / GIB:.2f} GiB "
                    f"(reserve {policy.reserve_system_bytes / GIB:.2f} GiB)"
                )
            elif memory_gate is not None and memory_gate.paused:
                pressure_reason = memory_gate.reason or "Host memory pressure"

            if pressure_reason:
                proc.terminate()
                proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)
                break

        result: dict[str, Any] | None = None
        if pressure_reason is None:
            try:
                result = result_queue.get(timeout=2.0)
            except Empty:
                result = None

        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

        if result is not None and not result.get("ok"):
            worker_error = str(result.get("error") or "ParserWorkerError")
            if _looks_like_memory_error(worker_error):
                pressure_reason = worker_error

        if pressure_reason is not None:
            stats.retried_memory += 1
            tmp_path.unlink(missing_ok=True)
            sidecar_tmp.unlink(missing_ok=True)
            if asset_dir.exists():
                shutil.rmtree(asset_dir, ignore_errors=True)

            if memory_gate is not None:
                memory_gate.request_pause(
                    f"{source_path.name}: {pressure_reason}",
                    minimum_seconds=2.0,
                )
                emit(
                    current_parse_file=str(source_path),
                    parse_state="paused_memory",
                    pause_reason=pressure_reason,
                    memory_tier=memory_tier,
                    retry=attempt,
                    worker_rss_gb=0.0,
                    memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                    queued=len(queue) + 1,
                )
                memory_gate.wait_until_safe(context=f"retrying {source_path.name}")

            if attempt + 1 < MAX_MEMORY_RETRIES:
                queue.append((row, attempt + 1))
                continue

            # Memory pressure is a deferral, never a red parser failure.
            registry.save_artifact(
                content_hash=content_hash,
                parser_version=cfg.parser_version,
                canonical_path=None,
                status="deferred_memory",
                error=pressure_reason,
            )
            stats.deferred_memory += 1
            emit(
                current_parse_file=str(source_path),
                parse_state="deferred_memory",
                pause_reason=pressure_reason,
                memory_tier=memory_tier,
                retry=attempt,
                worker_rss_gb=0.0,
                memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                queued=len(queue),
            )
            continue

        if result is None:
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
                error=error,
                memory_tier=memory_tier,
                worker_rss_gb=0.0,
                memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
                queued=len(queue),
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
            memory_tier=memory_tier,
            retry=attempt,
            worker_rss_gb=0.0,
            memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
            queued=len(queue),
        )
        gc.collect()

    emit(
        stage="ingest_complete",
        current_parse_file=None,
        parse_state="complete",
        worker_rss_gb=0.0,
        memory_available_gb=round(psutil.virtual_memory().available / GIB, 2),
        queued=0,
    )
    return stats
