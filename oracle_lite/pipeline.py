from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .dataset import build_domain_dataset
from .db import Registry
from .ingest import ingest_corpus
from .scanner import scan_corpus
from .snapshot import create_snapshot
from .training import run_domain_training


@dataclass(slots=True)
class OneClickResult:
    status: str
    snapshot_id: str
    snapshot_reused: bool
    dataset_reused: bool
    run_id: str | None
    scan: dict[str, int]
    ingest: dict[str, int]
    records: int
    text_records: int
    visual_records: int
    characters: int

    def to_dict(self) -> dict:
        return asdict(self)


def _snapshot_matches_current(
    registry: Registry,
    snapshot_row,
    *,
    parser_version: str,
    active_hashes: set[str],
) -> bool:
    if snapshot_row is None:
        return False

    try:
        snapshot_cfg = json.loads(snapshot_row["config_json"])
    except (TypeError, json.JSONDecodeError):
        return False

    if snapshot_cfg.get("parser_version") != parser_version:
        return False

    return registry.snapshot_hashes(snapshot_row["snapshot_id"]) == active_hashes


def _dataset_meta(cfg: AppConfig, snapshot_id: str) -> tuple[Path, Path]:
    output_dir = cfg.datasets_dir / snapshot_id / "domain"
    return output_dir, output_dir / "dataset.json"


def _ensure_dataset(cfg: AppConfig, snapshot_id: str) -> tuple[dict, bool]:
    output_dir, meta_path = _dataset_meta(cfg, snapshot_id)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta, True

    # A previous interrupted build may have left partial shards. This directory is
    # generated output only, so rebuild it atomically from the immutable snapshot.
    if output_dir.exists():
        shutil.rmtree(output_dir)

    result = build_domain_dataset(cfg, snapshot_id=snapshot_id)
    meta = json.loads((result.output_dir / "dataset.json").read_text(encoding="utf-8"))
    return meta, False


def _is_fully_trained(cfg: AppConfig, registry: Registry, snapshot_id: str) -> bool:
    run = registry.latest_training_run(snapshot_id, kind="multimodal-domain")
    if run is None or run["status"] != "completed" or not run["output_dir"]:
        return False

    output_dir = Path(run["output_dir"])
    run_json = output_dir / "run.json"
    adapter_dir = output_dir / "adapter-final"
    if not run_json.exists() or not adapter_dir.exists():
        return False

    try:
        metadata = json.loads(run_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    # Smoke tests use max_steps >= 0 and must never mark the snapshot as fully done.
    max_steps = metadata.get("preset_values", {}).get("max_steps", -1)
    try:
        return int(max_steps) < 0
    except (TypeError, ValueError):
        return False


def run_one_click(
    cfg: AppConfig,
    *,
    max_steps: int | None = None,
    verify_all: bool = False,
    trainer: Callable[..., str] | None = None,
    monitor: Any | None = None,
) -> OneClickResult:
    """Scan -> ingest -> freeze current corpus -> build dataset -> train/resume.

    Normal one-click runs always use a full current-corpus snapshot. Hashes make
    parsing incremental, while full snapshots keep the resulting adapter independent
    from fragile adapter-chaining semantics. This is intentionally optimized for
    correctness and reproducibility rather than speed.
    """
    if monitor is not None:
        monitor.update_phase("scan", "Scanning corpus")
    scan_stats = scan_corpus(cfg, verify_all=verify_all)
    if monitor is not None:
        monitor.update("corpus", scan_stats.as_dict())
        monitor.log("INFO", "Corpus scan completed", **scan_stats.as_dict())

    if monitor is not None:
        monitor.update_phase("ingest", "Parsing multimodal corpus")
    ingest_stats = ingest_corpus(cfg)
    if monitor is not None:
        corpus_state = scan_stats.as_dict()
        corpus_state.update({
            "ready": ingest_stats.ready,
            "skipped": ingest_stats.skipped,
            "failed": ingest_stats.failed,
            "visual_documents": ingest_stats.visual_documents,
            "visual_segments": ingest_stats.visual_segments,
        })
        monitor.update("corpus", corpus_state)
        monitor.log("INFO", "Corpus ingest completed", **ingest_stats.as_dict())

    if ingest_stats.failed:
        raise RuntimeError(
            f"{ingest_stats.failed} active corpus file(s) failed parsing. "
            "Oracle-Lite will not silently train on a partial corpus. "
            "Run 'oracle-lite ingest' to inspect/retry after fixing the source files."
        )

    registry = Registry(cfg.registry_path)
    active_hashes = registry.active_ready_hashes(cfg.parser_version)
    if not active_hashes:
        raise ValueError(
            "No trainable canonical corpus is available. Check corpus_dir and parser failures."
        )

    if monitor is not None:
        monitor.update_phase("snapshot", "Freezing current corpus snapshot")
    latest_full = registry.latest_snapshot(mode="full")
    snapshot_reused = _snapshot_matches_current(
        registry,
        latest_full,
        parser_version=cfg.parser_version,
        active_hashes=active_hashes,
    )

    if snapshot_reused:
        snapshot_id = latest_full["snapshot_id"]
    else:
        name = "auto-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap = create_snapshot(cfg, name=name, mode="full")
        snapshot_id = snap.snapshot_id

    if monitor is not None:
        monitor.update("dataset", {"snapshot_id": snapshot_id})
        monitor.log(
            "INFO",
            "Snapshot selected",
            snapshot_id=snapshot_id,
            reused=snapshot_reused,
        )
        monitor.update_phase("dataset", "Building multimodal training dataset")
    dataset_meta, dataset_reused = _ensure_dataset(cfg, snapshot_id)
    if monitor is not None:
        monitor.update("dataset", {
            "snapshot_id": snapshot_id,
            "records": int(dataset_meta.get("records", 0)),
            "text_records": int(dataset_meta.get("text_records", 0)),
            "visual_records": int(dataset_meta.get("visual_records", 0)),
            "characters": int(dataset_meta.get("characters", 0)),
            "reused": dataset_reused,
        })
        monitor.log(
            "INFO",
            "Dataset ready",
            records=int(dataset_meta.get("records", 0)),
            visual_records=int(dataset_meta.get("visual_records", 0)),
            reused=dataset_reused,
        )

    # A normal no-argument run is idempotent once the current corpus is fully
    # trained. Smoke runs are never treated as completion.
    if max_steps is None and _is_fully_trained(cfg, registry, snapshot_id):
        return OneClickResult(
            status="up_to_date",
            snapshot_id=snapshot_id,
            snapshot_reused=snapshot_reused,
            dataset_reused=dataset_reused,
            run_id=None,
            scan=scan_stats.as_dict(),
            ingest=ingest_stats.as_dict(),
            records=int(dataset_meta.get("records", 0)),
            text_records=int(dataset_meta.get("text_records", 0)),
            visual_records=int(dataset_meta.get("visual_records", 0)),
            characters=int(dataset_meta.get("characters", 0)),
        )

    if trainer is None:
        run_id = run_domain_training(
            cfg,
            snapshot_id=snapshot_id,
            max_steps=max_steps,
            monitor=monitor,
        )
    else:
        run_id = trainer(cfg, snapshot_id=snapshot_id, max_steps=max_steps)

    return OneClickResult(
        status="smoke_trained" if max_steps is not None else "trained",
        snapshot_id=snapshot_id,
        snapshot_reused=snapshot_reused,
        dataset_reused=dataset_reused,
        run_id=run_id,
        scan=scan_stats.as_dict(),
        ingest=ingest_stats.as_dict(),
        records=int(dataset_meta.get("records", 0)),
        text_records=int(dataset_meta.get("text_records", 0)),
        visual_records=int(dataset_meta.get("visual_records", 0)),
        characters=int(dataset_meta.get("characters", 0)),
    )
