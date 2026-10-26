"""Shared preparation stage for the all-local-model FAI marking pipeline."""
from __future__ import annotations

from .automatic_candidates import build_candidates
from .registration_pipeline import ensure_registration


def prepare_marking_candidates(store, project_id, model=None, progress=None):
    """Reindex MSOP, try model-selected registration, then route every FAI.

    Registration is useful for XYZ-driven placement, but it is not allowed to
    block the independent local visual model path. A missing, stale, or failed
    registration only disables geometric proposals; candidate generation
    preserves source XYZ and sends each affected item to visual localization.
    """
    refresh = getattr(store, 'refresh_msop_index', None)
    if refresh:
        refresh(project_id)
    reconcile=getattr(store,'require_current_feature_integrity',None)
    if reconcile:reconcile(project_id)
    registration_path = None
    registration_error = None
    try:
        registration_path = ensure_registration(store, project_id, model)
    except Exception as exc:
        registration_error = f'{type(exc).__name__}: {exc}'[:500]
        if progress:
            progress(dict(done=0, stage='visual_fallback',
                          message='坐标系配准未就绪；继续为全部 FAI 生成本地视觉定位任务'))
    classification=None
    if model is not None and getattr(model,'loaded',False):
        from .fai_strategy import strategy_state,classify_project
        classification=strategy_state(store,project_id)
        if progress:
            progress(dict(done=0,stage='method_classification',
                message='复用已确认的方法分类，仅将原文或失败诊断发生变化的FAI交给本地模型重新分类'))
        classification=classify_project(store,project_id,model,
            batch_size=1 if classification.get('status')=='partial' else 5)
    generated = build_candidates(store, project_id, persist=False, progress=progress)
    if classification:
        generated['method_classification']={'status':classification['status'],
            'classified':len(classification.get('items',[])),'total':classification.get('total')}
        strategies={row['fai']:row for row in classification.get('items',[])}
        for candidate in generated.get('candidates',[]):
            if candidate.get('fai') in strategies:
                candidate.setdefault('evidence',{})['fai_strategy']=strategies[candidate['fai']]
                if strategies[candidate['fai']].get('evidence_class','').startswith('complete_xyz_'):
                    candidate['evidence']['source_scope_review_requested']=True

    if registration_error:
        generated['registration_error'] = registration_error
    return registration_path, generated
