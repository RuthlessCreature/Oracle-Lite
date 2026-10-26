"""Run the Viewer automatic FAI pipeline for one project with its local model.

This command is an operational entry point to the same registration, candidate,
and model-authored auto-marking path used by the Viewer. It never creates
pending-review annotations or accepts hand-selected CAD locations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from talker.config import load_config
from talker.model import TalkerModel
from oracle_viewer.local_matching import mark_geometry_candidates
from oracle_viewer.marking_pipeline import prepare_marking_candidates
from oracle_viewer.marking_status import marker_coverage, unmarked_diagnostics
from oracle_viewer.store import ViewerStore


def configured_viewer_model(cfg: object) -> TalkerModel:
    """Construct the same configured local model that the Viewer uses."""
    # The interactive Viewer may alias TalkerState.model, whose endpoint is
    # configured through ORACLE_TALKER_OPENAI_BASE_URL. CLI runs must resolve
    # the same local Strata endpoint instead of silently loading a second copy
    # of the base model onto the GPU.
    endpoint = (
        os.environ.get("ORACLE_VIEWER_MODEL_BASE_URL", "").strip()
        or os.environ.get("ORACLE_TALKER_OPENAI_BASE_URL", "").strip()
    )
    if not endpoint:
        # The Viewer aliases TalkerState.model when no separate Viewer model
        # endpoint is configured, including its TALKER-specific environment.
        return TalkerModel(cfg.training_output_dir)
    return TalkerModel(
        cfg.training_output_dir,
        openai_base_url=endpoint,
        openai_model=os.environ.get(
            "ORACLE_VIEWER_MODEL",
            os.environ.get("ORACLE_TALKER_OPENAI_MODEL", "qwen3.8-flash-next-iq2_xs"),
        ),
        backend_label=os.environ.get(
            "ORACLE_VIEWER_MODEL_BACKEND",
            os.environ.get("ORACLE_TALKER_INFERENCE_BACKEND", "strata-native"),
        ),
    )


def finalize_marking_status(store, project_id, generated, decisions, identity):
    """Persist the final CLI batch state in the same Viewer record as UI runs."""
    latest = store.get(project_id)
    coverage = marker_coverage(store.items(latest))
    review_items = {item['fai']: item for item in decisions.get('items', [])}
    for diagnostic in unmarked_diagnostics(store, latest):
        if diagnostic['fai'] in review_items:
            review_items[diagnostic['fai']]['unmarked_diagnostic'] = diagnostic
        else:
            review_items[diagnostic['fai']] = diagnostic
    store.update_local_model_marking(
        project_id,
        status='partial' if coverage['unmarked'] else 'completed',
        fingerprint=None,
        review_input_fingerprint=None,
        total=decisions['counts']['processed'],
        scope_total=coverage['total'],
        adapter=decisions.get('adapter'),
        counts=decisions['counts'],
        model=identity.get('model_alias'),
        inference_backend=identity.get('inference_backend'),
        items=[review_items[number] for number in sorted(review_items)],
        coverage=coverage,
        error=None,
        candidate_generation=generated['items'],
        progress=dict(done=decisions['counts']['processed'], current_fai=None,
                      message='本地模型核对结束；覆盖数量按实际标记统计'),
        message=f"已标记 {coverage['marked']}/{coverage['total']} 项；未标记 {coverage['unmarked']} 项",
    )
    return coverage


def record_pipeline_failure(store, project_id, stage, exc):
    """Expose interrupted runs as retryable failures without discarding marks."""
    store.update_local_model_marking(
        project_id,
        status="failed",
        fingerprint=None,
        error=f"{type(exc).__name__}: {exc}"[:1000],
        message="自动标记未完成；已保存的 FAI 标记保留，可重新运行续做。",
        progress=dict(stage=stage, message="本地模型任务中断；等待重试"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Viewer project ID")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path; defaults to the project automation folder",
    )
    args = parser.parse_args()

    cfg = load_config()
    store = ViewerStore(cfg.state_dir / "viewer")
    project = store.get(args.project)
    model = configured_viewer_model(cfg)
    model.load()
    if not model.loaded:
        raise RuntimeError("Configured local Viewer model did not load")
    identity = model.info()

    stage = "registration_and_candidate_generation"
    try:
        registration_path, generated = prepare_marking_candidates(store, args.project, model)
        stage = "candidate_generation"
        stage = "local_model_marking"
        decisions = mark_geometry_candidates(
            store,
            model,
            cfg.fai_matching_model_root,
            args.project,
            generated_candidates=generated["candidates"],
        )
    except Exception as exc:
        record_pipeline_failure(store, args.project, stage, exc)
        raise
    coverage = finalize_marking_status(store, args.project, generated, decisions, identity)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": args.project,
        "drawing_no": project.get("drawing_no"),
        "revision": project.get("revision"),
        "model_identity": identity,
        "registration_receipt": str(registration_path) if registration_path else None,
        "registration_error": generated.get("registration_error"),
        "candidate_generation": generated["items"],
        "model_decisions": decisions["items"],
        "decision_counts": decisions["counts"],
        "coverage": coverage,
        "all_fai_processed_by_configured_local_model": True,
        "human_review_required_for_auto_marking": False,
        "metrology_accepted": False,
    }
    output = args.report or (
        store._dir(args.project)
        / "automatic-learning"
        / "local-model-marking-last.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "drawing_no": report["drawing_no"],
                "coverage": coverage,
                "decision_counts": decisions["counts"],
                "report": str(output),
                "model": identity.get("model_alias"),
                "backend": identity.get("inference_backend"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
