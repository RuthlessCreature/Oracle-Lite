"""Report actual marker coverage independently of training/inference receipts."""
from __future__ import annotations

import json

from .candidate_mining import extract_feature_labels, extract_named_points
from .msop_coordinates import _coordinate_rows


def marker_coverage(items):
    marked = [i for i in items if
              (i.get('annotation') or {}).get('origin') == 'local_model_candidate'
              and (i.get('annotation') or {}).get('status') == 'auto_marked'
              and (i.get('annotation') or {}).get('anchors')]
    return dict(total=len(items), marked=len(marked),
                local_model_marked=sum((i.get('annotation') or {}).get('origin') == 'local_model_candidate'
                                       and (i.get('annotation') or {}).get('status') != 'stale'
                                       and bool((i.get('annotation') or {}).get('anchors')) for i in items),
                unmarked=len(items)-len(marked),
                unmarked_numbers=[i['number'] for i in items if i not in marked])


def unmarked_diagnostics(store, project):
    """Source-scoped reasons, never turn an ICP proposal into a CAD annotation."""
    msop = store.active_msop(project) or {}
    pages = msop.get('method_pages', [])
    named = extract_named_points(pages)
    model = project.get('model') or {}
    receipt_path = store._dir(project['id']) / 'automatic-learning' / 'registration.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    matched = (receipt.get('cad_sha256') == model.get('source_sha256')
               and receipt.get('msop_sha256') == msop.get('sha256'))
    frames = {(f.get('program'), f.get('coordinate_system')): f
              for f in receipt.get('frames', [])} if matched else {}
    results = []
    for item in store.items(project):
        annotation = item.get('annotation') or {}
        if (annotation.get('origin') == 'local_model_candidate'
                and annotation.get('status') == 'auto_marked' and annotation.get('anchors')):
            continue
        detail = store.detail(project['id'], item['number'])
        rows = _coordinate_rows(detail['methods'], [],
                                extract_feature_labels(detail['methods'], item.get('features', [])),
                                pages, named)
        evidence = []
        for scope in sorted({(r.get('program'), r.get('coordinate_system')) for r in rows}, key=str):
            frame = frames.get(scope, {})
            fit = frame.get('triangle_refinement') or frame
            best = fit.get('best') or {}
            evidence.append(dict(program=scope[0], coordinate_system=scope[1],
                                 status=fit.get('status', 'no_source_matched_registration'),
                                 max_residual_mm=best.get('max_residual_mm'),
                                 threshold_mm=fit.get('threshold_mm'),
                                 source_to_cad_registered=frame.get('source_to_cad_registered', False)))
        conflicts = [r['name'] for r in rows if r.get('coordinate_conflict') or r.get('scope_ambiguous')]
        reason = ('source_coordinate_conflict' if any(r.get('coordinate_conflict') for r in rows) else
                  'source_scope_ambiguous' if conflicts else
                  'no_explicit_cross_page_geometry_reference' if not rows else
                  'missing_program_or_frame' if any(not r.get('program') or not r.get('coordinate_system') for r in rows) else
                  'registration_not_validated')
        results.append(dict(fai=item['number'], result='no_geometry_candidate', reason=reason,
                            source_point_count=len(rows), source_pages=item.get('pages', []),
                            conflicts=conflicts, frames=evidence))
    return results
