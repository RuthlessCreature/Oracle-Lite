"""Reproducible engineering checks and delivery status, independent of model prose."""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
from collections import Counter
import numpy as np
from . import msop
from .store import now
from .registration import frame_proposals

VERSION = 'engineering-checks-v1'


def method_contract(task):
    fields = ('page', 'program', 'probe', 'measured_system', 'dimension_system', 'settings', 'text')
    methods = [{k: m.get(k) for k in fields} for m in task.get('methods', [])]
    indicators = [dict(i, page=m.get('page')) for m in methods for i in msop.measurement_indicators(m.get('text') or '')]
    missing = [k for k in ('probe', 'program', 'measured_system', 'dimension_system')
               if any(not m.get(k) for m in methods)]
    if not methods: missing.append('method_pages')
    if not task.get('specifications'): missing.append('specification')
    if not indicators: missing.append('measurement_indicator')
    return dict(fai=task['number'], methods=methods, dependencies=task.get('dependencies', []),
                specifications=task.get('specifications', []), indicators=indicators,
                source_points=(task.get('msop_coordinate_map') or {}).get('source_points', []),
                missing_fields=missing, method_equivalence='unverified', metrology_accepted=False)


def audit_task(task, faces, warnings=()):
    from .visibility import closest_surface
    mapping = task.get('msop_coordinate_map') or {}
    checks = []
    for index, anchor in enumerate(task.get('anchors', [])):
        f = faces.get(anchor.get('face_id'))
        hit = None
        if f and f.get('indices'):
            vertices = np.asarray(f['positions'], float).reshape(-1, 3)
            hit = closest_surface(anchor['position'], vertices[np.asarray(f['indices'], int).reshape(-1, 3)])
        checks.append(dict(index=index, face_id=anchor.get('face_id'), position=anchor['position'],
                           distance_mm=hit['distance_mm'] if hit else None,
                           status='on_display_surface' if hit and hit['distance_mm'] <= .1 else 'needs_relocation'))
    reasons = list(warnings)
    if not checks: reasons.append('缺少CAD标记')
    if any(c['status'] == 'needs_relocation' for c in checks): reasons.append('标记偏离关联面或面关联失效')
    if mapping.get('status') != 'mapped_complete': reasons.append('MSOP点位尚未全部对应到CAD标记')
    if not mapping.get('source_frame_registered_to_cad'): reasons.append('MSOP三维坐标系尚未注册')
    conflicts = [p for p in mapping.get('source_points', []) if p.get('coordinate_conflict')]
    if conflicts: reasons.append('原文点位存在坐标、方向或程序归属冲突：' + ', '.join(p['name'] for p in conflicts))
    if task.get('location_status') != 'reviewed': reasons.append('标记未人工复核')
    contract = method_contract(task)
    if contract['missing_fields']: reasons.append('原测量方法字段不完整')
    source_count = len(mapping.get('source_points', []))
    mapped_count = len(mapping.get('mapped_points', []))
    # This evidence grade must never masquerade as a calibrated probability.
    grade = 'conflicting' if warnings or conflicts or any(c['status']=='needs_relocation' for c in checks) else (
        'geometry_supported' if checks and mapping.get('status')=='mapped_complete' else 'incomplete')
    return dict(fai=task['number'], pages=task.get('pages', []), marker_origin=task.get('location_origin'),
                model_confidence=task.get('location_confidence'), evidence_grade=grade,
                confidence_is_calibrated=False, anchors=checks, source_point_count=source_count,
                mapped_point_count=mapped_count, coordinate_status=mapping.get('status'),
                point_scope_conflict_count=len(conflicts),
                common_source_z_groups=mapping.get('common_source_z_groups', []),
                source_point_constraints=mapping.get('source_point_constraints', []),
                issues=list(dict.fromkeys(reasons)), contract=contract, metrology_accepted=False)


REQUIRED_LIBRARY = {
    'camera': ['width_px','height_px','pixel_um','mount','body_mm'],
    'lens': ['magnification','image_circle_mm','working_distance_mm','dof_mm','length_mm','front_diameter_mm','mount'],
    'light': ['body_mm'],
    'controller': ['body_mm','channels','output_voltage_v','output_current_a','interface'],
    'confocal_sensor': ['working_distance_mm','measuring_range_half_mm','body_mm'],
    'surface_sensor': ['body_mm','reference_distance_mm','z_range_mm','x_range_mm','interface'],
}


def audit_library(cat, asset_root=None):
    rows = []
    for item in cat['items']:
        spec = item.get('spec') or {}; assets = []
        for a in item.get('assets', []):
            valid = None
            if asset_root:
                p = Path(asset_root)/Path(a['file']).name
                valid = p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == a.get('sha256')
            assets.append(dict(file=a['file'], kind=a.get('kind'), hash_valid=valid))
        required=REQUIRED_LIBRARY.get(item['kind'], [])
        if item['kind']=='light':
            required=({'ring':['body_outer_mm','body_inner_mm','body_height_mm','light_distance_mm'],
                       'backlight':['body_mm','active_mm']}.get(spec.get('type'),['body_mm']))
        missing = [k for k in required if spec.get(k) is None]
        rows.append(dict(id=item['id'], kind=item['kind'], vendor=item.get('vendor'),
                         missing_layout_fields=missing, unknown_fields=item.get('unknown_fields', []),
                         cad_status=item.get('cad_status'), review_status=item.get('review_status'),
                         source_url=item.get('source_url'), datasheet_url=item.get('datasheet_url'), assets=assets,
                         has_pdf=any(a['kind']=='pdf' for a in assets),
                         procurement_ready=False, electrical_compatibility='unverified'))
    return dict(revision=cat.get('revision'), item_count=len(rows), categories=dict(Counter(r['kind'] for r in rows)),
                rows=rows, invalid_assets=[a['file'] for r in rows for a in r['assets'] if a['hash_valid'] is False])


def coverage_status(tasks, run):
    r = (run or {}).get('result') or {}; selected = set(r.get('selected_fai', []))
    marked = {t['number'] for t in tasks if t.get('marked')}
    routes = {v['number'] for v in r.get('method_routes', [])}
    optical = selected - set(r.get('uncovered_fai', [])) if r.get('heads') else set()
    return dict(source_fai=len(tasks), marked_fai=len(marked), reviewed_fai=sum(t.get('location_status')=='reviewed' for t in tasks),
                selected_fai=len(selected), method_proposed_fai=len(routes & selected),
                optical_acquisition_fai=len(optical), metrology_accepted_fai=0,
                unmarked_numbers=sorted({t['number'] for t in tasks}-marked),
                optical_uncovered_numbers=sorted(selected-optical),
                current_run=bool(run and run.get('status')=='complete' and not run.get('stale')),
                scope_note='光学覆盖仅统计现有标记点的采集布局，不代表完整轮廓可测或计量验收')


def build_report(store, project, tasks, mesh, cat, run=None):
    items = {i['number']: i for i in store.items(project)}
    faces = {f['id']: f for f in (mesh or {}).get('faces', [])}
    audits = [audit_task(t, faces, items[t['number']].get('warnings', [])) for t in tasks]
    model = project.get('model') or {}
    return dict(version=VERSION, created_at=now(), project_id=project['id'], drawing=project.get('drawing_no'),
                revision=project.get('revision'), source_hashes={d['role']:d['sha256'] for d in project['documents']},
                source_documents=[{k:d.get(k) for k in ('id','role','filename','sha256')} for d in project['documents']],
                geometry={k:model.get(k) for k in ('status','source_format','source_sha256','source_decoding_complete','source_topology_valid','source_face_count','meshed_source_face_count','face_count','triangle_count','brep_valid','metrology_ready','invalid_occt_face_count','error','geometry_authority')},
                coverage=coverage_status(tasks,run), fai=audits, frame_proposals=frame_proposals(tasks),
                library=audit_library(cat), decisions_by='configured_local_model',
                automatic_checks_by='reproducible_geometry_and_source_tools',
                status='engineering_checks_complete_not_metrology_accepted')


def csv_report(report):
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(['FAI','Pages','Marker origin','Model confidence','Evidence grade','MSOP points','Mapped points','Coordinate status','Point scope conflicts','Issues'])
    for a in report['fai']:
        w.writerow([a['fai'],','.join(map(str,a['pages'])),a['marker_origin'],a['model_confidence'],a['evidence_grade'],a['source_point_count'],a['mapped_point_count'],a['coordinate_status'],a.get('point_scope_conflict_count',0),'; '.join(a['issues'])])
    return '\ufeff'+out.getvalue()
