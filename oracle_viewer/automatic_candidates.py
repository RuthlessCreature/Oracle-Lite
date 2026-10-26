"""Automatic XYZ display candidates, preserving full source sections and scopes.

An unpaired frame fit is a hypothesis, never a registered measurement frame.
Each item must retain every referenced point inside the existing 0.1 mm
display retrieval bound; its local model must subsequently accept the proposal.
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import numpy as np

from .candidate_mining import (extract_feature_labels, extract_named_points,
                               constructed_support_rows, resolve_dominant_source_page_conflicts)
from .msop_coordinates import _coordinate_rows
from .msop_coordinates import (_explicit_measure_point_names, MEASURE_POINT_LIST,
                               MEASURED_GEOMETRY_LIST)
from .section_geometry import (source_plane_constraints, closest_plane_section,
                               prefer_near_surface_projection,
                               MAX_DISPLAY_SECTION_DEVIATION_MM)
from .surface_index import SurfaceIndex
from .annotations import MAX_ANNOTATION_POINTS, validate_anchors
from .auto_registration import MIN_REGISTRATION_POINTS

MEASURED_POINT_COUNT = re.compile(r'\bMeasure\s+(\d+)\s+(?:(?:vector|surface)\s+)?points?\b',re.I)
MEASURED_FEATURE_MARKER_COUNT = re.compile(
    r'\bMeasure\s+(\d+)\s+(?:(?:vector|vision|surface|top|bottom|upper|lower)\s+)*(?:circles?|lines?|planes?|scans?)\b', re.I)
DECLARED_LOCATION_COUNT = re.compile(
    r'\b(\d+)\s*[x×]\s*[^\n.;]{0,60}\b(?:measurement\s+locations?|measured\s+points?|weld\s+spots?)\b', re.I)


def _required_point_count(methods, named_points):
    """Count distinct measured point groups, not duplicate method/result pages."""
    resolved = _explicit_measure_point_names(methods, named_points)
    if resolved:
        return len(resolved)
    instructions = set()
    total = 0
    for page in methods or []:
        scope = (page.get('program'), page.get('measured_system'))
        text = page.get('text', '')
        for match in MEASURED_POINT_COUNT.finditer(text):
            count = int(match.group(1))
            instruction = text[match.start():].splitlines()[0]
            clause = re.sub(r'\s+', ' ', instruction).strip().upper()
            key = (*scope, count, clause)
            if key in instructions:
                continue
            instructions.add(key)
            total += count
    return total


def _required_marker_count(methods, named_points=None):
    """Count independent measured points, direct features, or declared locations.

    A point group and a directly measured feature group are separate visible
    FAI locations (for example, 18 vector points plus two vector circles).
    Constructed planes/lines are intentionally not counted because their
    supporting measured points already define the displayed locations.
    """
    point_instructions=set();feature_instructions=set();location_instructions=set()
    for page in methods or []:
        scope=(page.get('program'),page.get('measured_system'))
        text=page.get('text','')
        for regex,instructions in ((MEASURED_POINT_COUNT,point_instructions),
                                   (MEASURED_FEATURE_MARKER_COUNT,feature_instructions),
                                   (DECLARED_LOCATION_COUNT,location_instructions)):
            for match in regex.finditer(text):
                count=int(match.group(1))
                clause=re.sub(r'\s+',' ',text[match.start():].splitlines()[0]).strip().upper()
                instructions.add((*scope,count,clause))
    resolved_points=_explicit_measure_point_names(methods,named_points or {})
    point_count=(len(resolved_points) if resolved_points else
                 sum(key[-2] for key in point_instructions))
    # When exact XYZ names are resolved, measured circles/lines/planes are
    # already present in that set and must not be counted a second time.
    feature_count=(0 if resolved_points else
                   sum(key[-2] for key in feature_instructions))
    if point_count or feature_count:
        return point_count+feature_count
    return sum(key[-2] for key in location_instructions)


def _required_marker_groups(methods, named_points=None):
    """Expose the direct MSOP measurement groups as an explicit marker manifest."""
    named_points=named_points or {}
    groups=[];seen=set();has_direct_groups=False
    for page in methods or []:
        text=page.get('text','')
        from .msop_tables import table_measurement_manifest
        table_group=table_measurement_manifest(page)
        if table_group:
            has_direct_groups=True
            signature=('table',tuple(table_group['source_point_names']))
            if signature not in seen:
                seen.add(signature);groups.append(dict(table_group,kind='measured_features' if table_group.get('feature_kind') else 'measured_points'))
        for kind,pattern in (('measured_points',MEASURE_POINT_LIST),
                             ('measured_features',MEASURED_GEOMETRY_LIST)):
            for match in pattern.finditer(text):
                has_direct_groups=True
                clause=re.sub(r'\s+',' ',match.group(0)).strip()
                isolated={**page,'text':clause+'.'}
                resolved=_explicit_measure_point_names([isolated],named_points)
                names=[name for name in named_points if name in (resolved or set())]
                group=dict(kind=kind,declared_count=int(match.group(1)),
                           source_point_names=names,instruction=clause[:240],
                           source_page=page.get('page'))
                signature=(kind,tuple(names),int(match.group(1)),clause.upper())
                if signature not in seen:
                    seen.add(signature);groups.append(group)
    if not has_direct_groups:
        for page in methods or []:
            text=page.get('text','')
            for match in DECLARED_LOCATION_COUNT.finditer(text):
                clause=re.sub(r'\s+',' ',match.group(0)).strip()
                group=dict(kind='declared_locations',declared_count=int(match.group(1)),
                           source_point_names=[],instruction=clause[:240],
                           source_page=page.get('page'))
                signature=(group['kind'],group['declared_count'],clause.upper())
                if signature not in seen:
                    seen.add(signature);groups.append(group)
    return groups


def _source_point_evidence(rows):
    """Keep conflicting XYZ values as alternatives, never as a first-row default."""
    result=[]
    for row in rows:
        source_variants=row.get('source_coordinate_variants') or row.get('occurrences',[])
        variants=[];seen=set()
        for occurrence in source_variants:
            frames=occurrence.get('coordinate_system_candidates') or [occurrence.get('coordinate_system')]
            for frame in frames:
                variant=dict(page=occurrence.get('page'),xyz=occurrence.get('xyz'),
                             program=occurrence.get('program'),coordinate_system=frame,
                             frame_assignment_conflict=occurrence.get('frame_assignment_conflict',False))
                key=(variant['program'],variant['coordinate_system'],tuple(variant.get('xyz') or ()),
                     variant['frame_assignment_conflict'])
                if key not in seen:
                    seen.add(key);variants.append(variant)
        distinct={(v.get('program'),v.get('coordinate_system'),tuple(v.get('xyz') or []))
                  for v in variants if isinstance(v.get('xyz'),list) and len(v['xyz'])==3}
        ambiguous=(row.get('coordinate_conflict') or row.get('scope_ambiguous')
                   or row.get('frame_assignment_conflict') or len(distinct)>1)
        result.append(dict(name=row['name'],xyz=None if ambiguous else row.get('xyz'),
                           pages=row.get('pages',[]),program=row.get('program'),
                           coordinate_system=row.get('coordinate_system'),
                           requires_variant_choice=bool(ambiguous),
                           coordinate_variants=variants if ambiguous else []))
    return result


def _source_point_scopes(points):
    """Include every explicit variant scope before the model resolves a row.

    An unresolved point deliberately has no preferred scope. Omitting its
    variants here made the later model choice unusable: the corresponding
    already-selected registration transform was never provided for review.
    """
    scopes=set()
    for point in points:
        for row in [point,*(point.get('coordinate_variants') or [])]:
            scope=(row.get('program'),row.get('coordinate_system'))
            if all(isinstance(value,str) and value for value in scope):
                scopes.add(scope)
    return scopes


def _registration_frame_hypotheses(receipt, scopes):
    """Expose geometric alternatives for explicit FAI-level model review only."""
    result=[]
    for frame in receipt.get('frames',[]):
        scope=(frame.get('program'),frame.get('coordinate_system'))
        if scope not in scopes:
            continue
        fit=frame.get('triangle_refinement') or {}
        seen=set()
        for index,row in enumerate([fit.get('best'),*(fit.get('alternatives') or [])][:10]):
            if not isinstance(row,dict):
                continue
            try:
                rotation=np.asarray(row.get('rotation'),float)
                translation=np.asarray(row.get('translation_mm'),float)
            except (TypeError,ValueError):
                continue
            if (rotation.shape!=(3,3) or translation.shape!=(3,)
                    or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                    or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5)
                    or not np.isclose(np.linalg.det(rotation),1,atol=1e-5)):
                continue
            identity=tuple(np.round(np.r_[rotation.ravel(),translation],5))
            if identity in seen:
                continue
            seen.add(identity)
            result.append({key:row.get(key) for key in ('rotation','translation_mm',
                'p95_residual_mm','max_residual_mm','inlier_ratio','initialization')} | {
                'program':scope[0],'coordinate_system':scope[1],
                'hypothesis_id':f'FAI_REVIEW_H{index:02d}',
                'candidate_source':'geometric_hypothesis_pending_local_fai_model_review'})
    return result


def _same_source_stale_seed(item, mesh, msop, mesh_faces, invalid_faces,
                            source_rows=None, required_point_count=0):
    """Offer an unchanged-source auto mark only as a fresh model-review seed."""
    old = item.get('annotation') or {}
    anchors = old.get('anchors') or []
    if (old.get('status') != 'stale' or old.get('origin') != 'local_model_candidate'
            or old.get('stale_reason') != 'automatic_registration_changed'
            or old.get('stale_previous_status') != 'auto_marked'
            or old.get('source_sha256') != mesh.get('source_sha256')
            or old.get('document_id') != item.get('document_id')
            or old.get('document_sha256') != msop.get('sha256')
            or not anchors or len(anchors) > MAX_ANNOTATION_POINTS):
        return None
    if any(anchor.get('face_id') not in mesh_faces
           or anchor.get('face_id') in invalid_faces
           or not isinstance(anchor.get('position'), list)
           or len(anchor['position']) != 3
           or not np.isfinite(np.asarray(anchor['position'], float)).all()
           for anchor in anchors):
        return None
    face_ids = {anchor['face_id'] for anchor in anchors}
    if not face_ids or len(face_ids) > 12:
        return None
    try:
        validated_anchors=validate_anchors(mesh,anchors)
    except (TypeError,ValueError,KeyError):
        return None
    seed_evidence=dict(source_sha256=old['source_sha256'],
                       msop_sha256=old['document_sha256'],
                       stale_reason=old['stale_reason'],
                       previous_status=old['stale_previous_status'],
                       previous_confidence=old.get('confidence'),
                       previous_note=old.get('note', '')[:1200],
                       anchor_count=len(anchors),
                       unique_face_count=len(face_ids),
                       source_hashes_verified=True,
                       all_faces_exist_in_current_CAD=True,
                       current_face_anchor_validation=True,
                       metrology_validated=False)
    seed=dict(fai=item['number'], anchors=validated_anchors,
                candidate_kind='same_source_stale_seed',
                origin='same_source_stale_seed',
                note='同一 CAD 与 MSOP 文件下曾由本地模型自动标记；仅作为重新核对的候选位置，不沿用旧结论。',
                seed_evidence=seed_evidence)
    # A same-source seed can bypass a poor visual re-localization only after a
    # fresh, point-by-point audit against the current source rows. The old
    # model decision is never reused: the resulting candidate still requires
    # a new local-model review in mark_geometry_candidates().
    note=old.get('note','')
    evidence=None
    for line in str(note).splitlines()[1:]:
        if not line.lstrip().startswith('{'):
            continue
        try:
            parsed=json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed,dict) and isinstance(parsed.get('point_correspondences'),list):
            evidence=parsed
            break
    if evidence and source_rows:
        correspondences=evidence.get('point_correspondences') or []
        expected={(row.get('name'),row.get('program'),row.get('coordinate_system')):row
                  for row in source_rows}
        actual={(row.get('name'),row.get('program'),row.get('coordinate_system')):row
                for row in correspondences}
        max_section=float(evidence.get('max_surface_anchor_section_deviation_mm',0.08))
        point_rows_match=(len(expected)==len(source_rows)==len(correspondences)
            and set(expected)==set(actual)
            and all(isinstance(actual[key].get('source_xyz'),list)
                    and np.allclose(np.asarray(expected[key].get('xyz'),float),
                                    np.asarray(actual[key]['source_xyz'],float),atol=1e-6,rtol=0)
                    and isinstance(actual[key].get('residual_mm'),(int,float))
                    and actual[key]['residual_mm']<=0.1
                    and isinstance(actual[key].get('plane_residual_mm'),(int,float))
                    and abs(actual[key]['plane_residual_mm'])<=max_section
                    for key in expected)
            and (not required_point_count or len(correspondences)==required_point_count)
            and evidence.get('all_source_points_retained') is True
            and evidence.get('source_sections_preserved') is True)
        corr_positions={tuple(np.round(np.asarray(row.get('position'),float),6))
                        for row in correspondences
                        if isinstance(row.get('position'),list) and len(row['position'])==3}
        anchor_positions={tuple(np.round(np.asarray(anchor['position'],float),6))
                          for anchor in validated_anchors}
        point_rows_match=(point_rows_match and len(corr_positions)==len(correspondences)
                          and corr_positions==anchor_positions)
        if point_rows_match:
            seed['automatic_xyz_hypothesis']=True
            seed['evidence']=evidence
            seed_evidence.update(current_source_points_match=True,
                                 all_point_residuals_within_0_1_mm=True,
                                 all_source_section_residuals_within_gate=True,
                                 explicit_point_count_matches=True,
                                 point_anchor_bijection=True,
                                 freshly_validated=True)
    return seed


def _fit_bounded_tangent_shift(rows, rotation, translation, planes, index, triangles,
                               max_residual_mm=.1, max_shift_mm=.3, search_radius_mm=.6):
    """Refine only the in-plane origin against a complete MSOP section group.

    The global rigid hypothesis supplies orientation and section height. This
    bounded correction uses all point-to-section intersections together and
    preserves the source-Z plane, so it cannot independently drag anchors onto
    different CAD layers.
    """
    rows = list(rows)
    keyed = [(row, planes.get(row.get('point_key') or row['name'])) for row in rows]
    if len(keyed) < 3 or any(plane is None for _, plane in keyed):
        return np.asarray(translation, float), None
    normals = [rotation @ np.asarray(plane['normal_source'], float) for _, plane in keyed]
    reference_normal = normals[0] / np.linalg.norm(normals[0])
    if any(abs(float(reference_normal @ (normal / np.linalg.norm(normal)))) < .999
           for normal in normals[1:]):
        return np.asarray(translation, float), None

    origin = np.asarray(translation, float).copy()
    translation = origin.copy()

    def project(current_translation):
        targets, hits, distances = [], [], []
        for row, plane in keyed:
            target = rotation @ np.asarray(row['xyz'], float) + current_translation
            normal = rotation @ np.asarray(plane['normal_source'], float)
            normal = normal / np.linalg.norm(normal)
            offset = float(plane['offset_source_mm'] + normal @ current_translation)
            nearby = index.nearby_triangles(target, search_radius_mm)
            if not len(nearby):
                return None
            hit = closest_plane_section(target, triangles[nearby], normal, offset)
            if hit is None:
                return None
            targets.append(target)
            hits.append(np.asarray(hit['point'], float))
            distances.append(float(hit['distance_mm']))
        return np.asarray(targets), np.asarray(hits), np.asarray(distances)

    baseline = project(translation)
    if baseline is None:
        return translation, None
    before = baseline[2]
    for _ in range(6):
        current = project(translation)
        if current is None:
            return origin, None
        targets, hits, _ = current
        delta = np.mean(hits - targets, axis=0)
        delta -= reference_normal * float(delta @ reference_normal)
        if np.linalg.norm(delta) < 1e-5:
            break
        translation += delta
    after_result = project(translation)
    if after_result is None:
        return origin, None
    after = after_result[2]
    shift = float(np.linalg.norm(translation - origin))
    improvement = float(np.max(before) - np.max(after))
    if (shift > max_shift_mm or float(np.max(after)) > max_residual_mm
            or improvement < .02 or float(np.median(after)) > float(np.median(before)) + .02):
        return origin, None
    return translation, dict(method='bounded_common_in_plane_shift_to_MSOP_Z_sections',
        shift_mm=shift, max_residual_before_mm=float(np.max(before)),
        max_residual_after_mm=float(np.max(after)),
        median_residual_before_mm=float(np.median(before)),
        median_residual_after_mm=float(np.median(after)),
        source_to_cad_registered=False, metrology_accepted=False)


def _fit_bounded_common_translation(rows, rotation, translation, index, max_residual_mm=.1,
                                    max_shift_mm=.3):
    """Apply one bounded XYZ shift to a complete display point group.

    Every source point keeps its exact relative vector, including equal-Z
    coplanarity. This display-only fit cannot certify frame registration.
    """
    rows = list(rows)
    if len(rows) < 2:
        return np.asarray(translation, float), None
    rotation = np.asarray(rotation, float)
    origin = np.asarray(translation, float).copy()
    current = origin.copy()

    def project(candidate_translation):
        targets, hits, distances = [], [], []
        for row in rows:
            target = rotation @ np.asarray(row['xyz'], float) + candidate_translation
            hit = index.nearest(target)
            if hit is None or not np.isfinite(hit.get('distance_mm', np.inf)):
                return None
            targets.append(target)
            hits.append(np.asarray(hit['point'], float))
            distances.append(float(hit['distance_mm']))
        return np.asarray(targets), np.asarray(hits), np.asarray(distances)

    baseline = project(current)
    if baseline is None:
        return origin, None
    before = baseline[2]
    for _ in range(8):
        measured = project(current)
        if measured is None:
            return origin, None
        targets, hits, _ = measured
        delta = np.median(hits - targets, axis=0)
        if np.linalg.norm(delta) < 1e-5:
            break
        current += delta
    result = project(current)
    if result is None:
        return origin, None
    after = result[2]
    shift = float(np.linalg.norm(current - origin))
    improvement = float(np.max(before) - np.max(after))
    if (shift > max_shift_mm or float(np.max(after)) > max_residual_mm
            or improvement < .02 or float(np.median(after)) > float(np.median(before)) + .02):
        return origin, None
    return current, dict(method='bounded_common_xyz_translation_to_CAD_surface', shift_mm=shift,
        max_residual_before_mm=float(np.max(before)), max_residual_after_mm=float(np.max(after)),
        median_residual_before_mm=float(np.median(before)), median_residual_after_mm=float(np.median(after)),
        preserves_all_inter_point_vectors=True, source_to_cad_registered=False, metrology_accepted=False)


def _fit_bounded_minimax_translation(rows, rotation, translation, index, max_residual_mm=.1,
                                     max_shift_mm=.3):
    """Find one bounded rigid translation minimizing the worst point-to-surface gap.

    Approximate triangle retrieval is used only during optimization. The proposed
    translation is rechecked against exact CAD triangles for every source point;
    no point is dropped and the source group's vectors and planes are unchanged.
    """
    from scipy.optimize import minimize

    rows=list(rows)
    if len(rows)<2:return np.asarray(translation,float),None
    rotation=np.asarray(rotation,float);origin=np.asarray(translation,float).copy()
    source=np.asarray([row['xyz'] for row in rows],float)
    base=source@rotation.T+origin
    before=np.asarray([index.nearest(point)['distance_mm'] for point in base],float)
    if not np.isfinite(before).all():return origin,None

    def distances(delta,exact=False):
        return np.asarray([index.nearest(point+delta,exact=exact)['distance_mm'] for point in base],float)

    def objective(delta):
        delta=np.asarray(delta,float);length=float(np.linalg.norm(delta))
        ds=distances(delta,exact=False)
        # The max term protects every source point; the small median term
        # resolves flat minimax plateaus without sacrificing worst-case fit.
        return float(ds.max()+.02*np.median(ds)+1000*max(0.0,length-max_shift_mm)**2)

    result=minimize(objective,np.zeros(3),method='Powell',
                    bounds=[(-max_shift_mm,max_shift_mm)]*3,
                    options={'maxiter':24,'xtol':.004,'ftol':1e-4})
    current=origin+np.asarray(result.x,float)
    after=distances(current-origin,exact=True)
    shift=float(np.linalg.norm(current-origin))
    improvement=float(np.max(before)-np.max(after))
    if (not np.isfinite(after).all() or shift>max_shift_mm+1e-9
            or float(np.max(after))>max_residual_mm or improvement<.02
            or float(np.median(after))>float(np.median(before))+.02):
        return origin,None
    return current,dict(method='bounded_minimax_common_translation_to_CAD_surface',
        shift_mm=shift,max_residual_before_mm=float(np.max(before)),
        max_residual_after_mm=float(np.max(after)),median_residual_before_mm=float(np.median(before)),
        median_residual_after_mm=float(np.median(after)),preserves_all_inter_point_vectors=True,
        optimizer='approximate_surface_search_then_exact_all_point_validation',
        source_to_cad_registered=False,metrology_accepted=False)


def build_candidates(store, project_id, *, registration_path=None, persist=False, progress=None):
    project=store.get(project_id);msop=store.active_msop(project);mesh=store.mesh(project_id)
    mesh_faces={face['id']:face for face in mesh.get('faces',[])}
    path=Path(registration_path) if registration_path else store._dir(project_id)/'automatic-learning'/'registration.json'
    from .auto_registration import registration_algorithm_sha256
    current_algorithm_sha256=registration_algorithm_sha256()
    registration_status='missing'
    receipt={}
    if path.is_file():
        try:
            candidate_receipt=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):
            registration_status='invalid_receipt'
        else:
            if (candidate_receipt.get('cad_sha256')==mesh['source_sha256']
                    and candidate_receipt.get('msop_sha256')==msop['sha256']
                    and candidate_receipt.get('algorithm_sha256')==current_algorithm_sha256):
                receipt=candidate_receipt
                registration_status='current'
            else:
                registration_status='stale_receipt'
    frames={(f.get('program'),f.get('coordinate_system')):f for f in receipt.get('frames',[])}
    registration_decisions=(receipt.get('local_model_registration') or {}).get('frames',[])

    def selected_model_transform(scope):
        frame=frames.get(scope) or {}
        decision=next((row for row in registration_decisions
                       if row.get('program')==scope[0]
                       and row.get('coordinate_system')==scope[1]
                       and row.get('status')=='selected'),None)
        if not decision:
            return None
        selected=decision.get('selected_transform')
        if isinstance(selected,dict):
            rotation=selected.get('rotation');translation=selected.get('translation_mm')
            if (isinstance(rotation,list) and len(rotation)==3
                    and isinstance(translation,list) and len(translation)==3):
                return {key:selected.get(key) for key in ('rotation','translation_mm','p95_residual_mm',
                    'max_residual_mm','held_out_max_mm','inlier_ratio','initialization',
                    'support_feature_group','support_point_names','support_point_count')} | {
                    'program':scope[0],'coordinate_system':scope[1],
                    'hypothesis_id':decision.get('hypothesis_id'),
                    'model_confidence':decision.get('model_confidence'),
                    'model_evidence':decision.get('evidence')}
        fit=frame.get('triangle_refinement') or {}
        options=[fit.get('best'),*(fit.get('alternatives') or [])]
        index=decision.get('hypothesis_index')
        if (type(index) is not int or not 0<=index<len(options)
                or decision.get('hypothesis_id')!=f'H{index:02d}'):
            return None
        option=options[index]
        if not isinstance(option,dict):
            return None
        return {key:option.get(key) for key in ('rotation','translation_mm','p95_residual_mm',
                'max_residual_mm','held_out_max_mm','inlier_ratio','initialization')} | {
                'program':scope[0],'coordinate_system':scope[1],
                'hypothesis_id':decision.get('hypothesis_id'),
                'model_confidence':decision.get('model_confidence'),
                'model_evidence':decision.get('evidence')}
    triangles=[];face_ids=[];invalid=set()
    for face in mesh['faces']:
        xyz=np.asarray(face['positions'],float).reshape(-1,3)
        tri=xyz[np.asarray(face['indices'],int).reshape(-1,3)]
        triangles.append(tri);face_ids.extend([face['id']]*len(tri))
        if face.get('occt_face_valid') is False:invalid.add(face['id'])
    triangles=np.concatenate(triangles)
    # Invalid source faces are never eligible anchors. Search the complete
    # valid surface set, rather than rejecting an item because an invalid
    # tessellation happened to win a nearest-neighbour tie at a shared edge.
    valid_ids=np.asarray([i for i,f in enumerate(face_ids) if f not in invalid],dtype=int)
    if not len(valid_ids):
        return dict(candidates=[],items=[],reason='no_valid_display_faces')
    valid_triangles=triangles[valid_ids];index=SurfaceIndex(valid_triangles)
    pages=msop['method_pages'];named=extract_named_points(pages)
    report=[];candidates=[]
    # Only a current local-model mark counts as complete. Manual bindings and
    # hand-authored annotations must never suppress automatic FAI processing.
    pending=[i for i in store.items(project) if not (
             (i.get('annotation') or {}).get('origin') == 'local_model_candidate'
             and (i.get('annotation') or {}).get('status') == 'auto_marked')]
    for item_index,item in enumerate(pending):
        annotation=item.get('annotation') or {}
        if annotation.get('origin')=='local_model_candidate' and annotation.get('status')!='stale':continue
        if progress:
            progress(dict(stage='candidate_generation',done=item_index,total=len(pending),
                          current_fai=item['number'],message=f"检查 FAI {item['number']} 的全组点位与有效 CAD 面"))
        detail=store.detail(project_id,item['number'])
        rows=_coordinate_rows(detail['methods'],[],extract_feature_labels(detail['methods'],item.get('features',[])),pages,named)
        rows,coordinate_page_resolutions=resolve_dominant_source_page_conflicts(rows)
        required_point_count=_required_point_count(detail['methods'],named)
        required_marker_count=_required_marker_count(detail['methods'],named)
        required_marker_groups=_required_marker_groups(detail['methods'],named)
        reason=None
        if not rows and required_point_count:reason='incomplete_measurement_point_coordinates'
        elif not rows:reason='no_source_points'
        elif len(rows)>MAX_ANNOTATION_POINTS:reason='point_group_exceeds_current_annotation_limit'
        elif required_point_count and len({row['name'] for row in rows})<required_point_count:
            reason='incomplete_measurement_point_coordinates'
        elif (not required_point_count and required_marker_count
              and len({row['name'] for row in rows})<required_marker_count):
            reason='incomplete_measurement_point_coordinates'
        elif any(r.get('coordinate_conflict') or r.get('scope_ambiguous') for r in rows):reason='source_conflict'
        constraints=source_plane_constraints(rows,detail['methods'])
        planes={key:plane for plane in constraints for key in plane['point_keys']}
        projected=[];frame_evidence={};failures=[];frame_transforms={};item_transforms={}
        def reject(row, failure_reason, **evidence):
            failures.append(dict(name=row['name'],source_xyz=row['xyz'],pages=row.get('pages'),
                program=row.get('program'),coordinate_system=row.get('coordinate_system'),
                reason=failure_reason,**evidence))
        for row in ([] if reason else rows):
            scope=(row.get('program'),row.get('coordinate_system'));frame=frames.get(scope,{})
            fit=frame.get('triangle_refinement') or {}
            model_decisions=(receipt.get('local_model_registration') or {}).get('frames',[])
            decision=next((value for value in model_decisions
                           if value.get('program')==scope[0]
                           and value.get('coordinate_system')==scope[1]
                           and value.get('status')=='selected'),None)
            if decision is None:
                reject(row,'local_model_did_not_select_registration_hypothesis');continue
            best=selected_model_transform(scope)
            if not best:
                reject(row,'invalid_local_model_registration_hypothesis');continue
            if not best or None in scope or len(frame.get('source_points',[]))<MIN_REGISTRATION_POINTS:
                reject(row,'missing_valid_rigid_frame_hypothesis');continue
            support=np.asarray([p['xyz'] for p in frame['source_points']],float)
            if np.linalg.matrix_rank(support-support.mean(0),tol=1e-6)<2:
                reject(row,'insufficient_source_frame_spread');continue
            rotation=np.asarray(best['rotation'],float);translation=np.asarray(best['translation_mm'],float)
            if (rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all()
                    or not np.isfinite(translation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
                    or not np.isclose(np.linalg.det(rotation),1,atol=1e-6)):
                raise ValueError('Invalid rigid frame proposal')
            threshold=float(fit['threshold_mm'])
            if not 0<threshold<=.1:raise ValueError('Display proposal bound must not exceed 0.1 mm')
            if scope not in frame_transforms:
                frame_transforms[scope]=(rotation,translation,fit)
            rotation,base_translation,fit=frame_transforms[scope]
            if scope not in item_transforms:
                scope_rows=[p for p in rows if (p.get('program'),p.get('coordinate_system'))==scope]
                support_rows=constructed_support_rows(scope_rows,detail['methods'],named)
                fit_rows={r.get('point_key') or r['name']:r for r in scope_rows}
                fit_rows.update({r.get('point_key') or r['name']:r for r in support_rows})
                fit_rows=list(fit_rows.values())
                fit_planes={key:plane for plane in source_plane_constraints(fit_rows,detail['methods'])
                            for key in plane['point_keys']}
                display_translation,adjustment=_fit_bounded_tangent_shift(
                    fit_rows,rotation,base_translation,fit_planes,index,valid_triangles,
                    max_residual_mm=threshold)
                common_translation,common_adjustment=_fit_bounded_common_translation(
                    fit_rows,rotation,base_translation,index,max_residual_mm=threshold)
                if common_adjustment and (adjustment is None or common_adjustment['max_residual_after_mm']
                                          < adjustment['max_residual_after_mm']):
                    display_translation,adjustment=common_translation,common_adjustment
                minimax_translation,minimax_adjustment=_fit_bounded_minimax_translation(
                    fit_rows,rotation,base_translation,index,max_residual_mm=threshold)
                if minimax_adjustment and (adjustment is None or minimax_adjustment['max_residual_after_mm']
                                           < adjustment['max_residual_after_mm']):
                    display_translation,adjustment=minimax_translation,minimax_adjustment
                if adjustment and support_rows:
                    adjustment['shared_feature_support']={
                        'point_names':sorted({r['name'] for r in support_rows
                                              if (r.get('point_key') or r['name']) not in
                                              {p.get('point_key') or p['name'] for p in scope_rows}}),
                        'source_pages':sorted({page for r in support_rows for page in r.get('pages',[])
                                               if page not in {q for p in scope_rows for q in p.get('pages',[])}}),
                        'used_for_display_refinement_only':True}
                # The bounded refinement is specific to this FAI's complete
                # source group. Reusing the first FAI's translation for every
                # later item in the same measurement frame can move their
                # otherwise valid coplanar groups away from the CAD surface.
                item_transforms[scope]=(rotation,display_translation,fit,adjustment)
            rotation,translation,fit,adjustment=item_transforms[scope]
            target=np.asarray(row['xyz'],float)@rotation.T+translation
            nearest_hit=index.nearest(target)
            hit=nearest_hit
            plane=planes.get(row.get('point_key') or row['name'])
            if plane:
                normal=rotation@np.asarray(plane['normal_source'],float)
                offset=float(plane['offset_source_mm']+normal@translation)
                near=index.nearby_triangles(target,threshold)
                section=closest_plane_section(target,valid_triangles[near],normal,offset)
                section=prefer_near_surface_projection(
                    hit,section,normal,offset,max_surface_distance_mm=threshold,
                    max_section_deviation_mm=MAX_DISPLAY_SECTION_DEVIATION_MM)
                if section is None:
                    # Keep the two failure modes separate in diagnostics: the
                    # transformed XYZ may be far from the CAD, or it may be
                    # close while its source plane has no local intersection.
                    # That distinction drives a general registration fix and
                    # prevents a future fallback from silently snapping points
                    # onto the wrong face.
                    reject(row,'source_section_has_no_nearby_CAD_intersection',
                           transformed_xyz=target.tolist(),
                           nearest_surface_position=hit.get('point'),
                           nearest_surface_residual_mm=hit.get('distance_mm'),
                           nearest_surface_plane_residual_mm=(
                               abs(float(normal@np.asarray(hit['point'],float))-offset)
                               if hit.get('point') is not None else None),
                           section_search_triangle_count=int(len(near)))
                    continue
                if section.get('projection_method'):
                    hit=section
                else:
                    hit=dict(section,triangle_index=int(near[section['triangle_index']]))
            face_id=face_ids[int(valid_ids[hit['triangle_index']])]
            if face_id in invalid:
                reject(row,'candidate_on_invalid_display_face');continue
            if hit['distance_mm']>threshold:
                reject(row,'item_point_residual_rejected',transformed_xyz=target.tolist(),
                       nearest_valid_position=hit['point'],residual_mm=hit['distance_mm'],
                       nearest_surface_position=nearest_hit.get('point'),
                       nearest_surface_residual_mm=nearest_hit.get('distance_mm'),
                       plane_projection_method=hit.get('projection_method'))
                continue
            projected.append(dict(name=row['name'],point_key=row.get('point_key'),program=scope[0],
                                  coordinate_system=scope[1],source_xyz=row['xyz'],source_pages=row['pages'],
                                  source_ijk=row.get('ijk'),
                                  coordinate_precision_equivalent=row.get('coordinate_precision_equivalent',False),
                                  coordinate_variant_span_mm=row.get('coordinate_variant_span_mm',0.0),
                                  semantic_resolution=row.get('semantic_resolution'),
                                  source_coordinate_variants=[dict(page=o.get('page'),xyz=o.get('xyz'),
                                      coordinate_system=o.get('coordinate_system'),program=o.get('program'))
                                      for o in (row.get('source_coordinate_variants') or row.get('occurrences',[]))],
                                  coordinate_page_resolution=row.get('coordinate_page_resolution'),
                                  transformed_xyz=target.tolist(),position=hit['point'],face_id=face_id,
                                  residual_mm=hit['distance_mm'],
                                  nearest_surface_position=nearest_hit.get('point'),
                                  nearest_surface_residual_mm=nearest_hit.get('distance_mm'),
                                  plane_residual_mm=hit.get('plane_residual_mm'),
                                  plane_projection_method=hit.get('projection_method','exact_source_plane_section')))
            frame_evidence[str(scope)]=dict(program=scope[0],coordinate_system=scope[1],
                                          rotation=rotation.tolist(),translation_mm=translation.tolist(),
                                          selected_by='local_model',selected_hypothesis_id=decision['hypothesis_id'],
                                          bounded_local_adjustment=adjustment,
                                          global_status=fit.get('status'),global_max_residual_mm=best['max_residual_mm'],
                                          source_to_cad_registered=False,threshold_mm=threshold)
        if failures:reason=failures[0]['reason']
        record=dict(fai=item['number'],result=reason or 'display_candidate_ready',
                    source_point_count=len(rows),supported_point_count=len(projected),
                    required_point_count=required_point_count,
                    missing_measured_point_count=max(0,required_point_count-len({r['name'] for r in rows})),
                    coordinate_page_resolutions=coordinate_page_resolutions)
        if failures:
            record.update(failed_point=failures[0],failed_points=failures,failed_point_count=len(failures))
        if reason and rows:
            record['source_issues']=[dict(name=r['name'],pages=r.get('pages'),
                program=r.get('program'),coordinate_system=r.get('coordinate_system'),
                coordinate_conflict=r.get('coordinate_conflict',False),
                vector_conflict=r.get('vector_conflict',False),scope_ambiguous=r.get('scope_ambiguous',False))
                for r in rows if r.get('coordinate_conflict') or r.get('scope_ambiguous')]
            if not record['source_issues'] and len(projected)<len(rows) and not failures:
                failed=rows[len(projected)]
                record['failed_point']=dict(name=failed['name'],source_xyz=failed['xyz'],
                    pages=failed.get('pages'),program=failed.get('program'),
                    coordinate_system=failed.get('coordinate_system'))
                if reason=='item_point_residual_rejected':
                    record['failed_point'].update(transformed_xyz=target.tolist(),
                        nearest_valid_position=hit['point'],residual_mm=hit['distance_mm'])
        if not reason:
            anchors=[dict(face_id=r['face_id'],position=r['position'],label_offset=[32,-26]) for r in projected]
            if len({a['face_id'] for a in anchors})>12:
                record['result']='face_group_exceeds_current_model_review_limit'
            else:
                evidence=dict(scope='XYZ display hypothesis only, NOT a registered measurement frame',
                              registration_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              cad_sha256=mesh['source_sha256'],msop_sha256=msop['sha256'],
                              all_source_points_retained=True,source_sections_preserved=True,
                              max_surface_anchor_section_deviation_mm=MAX_DISPLAY_SECTION_DEVIATION_MM,
                              coordinate_page_resolutions=coordinate_page_resolutions,
                              frame_hypotheses=list(frame_evidence.values()),point_correspondences=projected)
                note=f'自动 XYZ 显示候选；全组点位及变换后的原文等 Z 截面均保留。CAD 锚点可在 {MAX_DISPLAY_SECTION_DEVIATION_MM:.2f} mm 显示限值内贴近截面；每点记录截面偏离量。尚未完成坐标系及计量验证。\n'+json.dumps(evidence,ensure_ascii=False,separators=(',',':'))
                candidate=dict(fai=item['number'],anchors=anchors,note=note,evidence=evidence,
                               candidate_kind='xyz_display',known_source_points=rows,
                               required_marker_count=required_marker_count,
                               required_marker_groups=required_marker_groups,
                               max_point_residual_mm=max(r['residual_mm'] for r in projected))
                candidates.append(candidate)
                if persist:
                    store.annotate(project_id,item['number'],anchors,note,'pending_review',mesh['source_sha256'],
                                   item['document_id'],'geometry_candidate','low')
        if not any(candidate['fai']==item['number'] for candidate in candidates):
            seed=_same_source_stale_seed(item,mesh,msop,mesh_faces,invalid,
                                         source_rows=rows,required_point_count=required_point_count)
            if seed and seed.get('automatic_xyz_hypothesis'):
                candidates.append(seed)
                record['geometry_result']=record['result']
                record['result']='same_source_xyz_seed_ready_for_local_model_review'
                record['stale_seed_evidence']=seed['seed_evidence']
            elif ((reason in ('no_source_points','incomplete_measurement_point_coordinates','source_conflict')
                   or reason is None and rows)
                  or (rows and failures and not record.get('source_issues')
                      and all(failure.get('reason') in {
                          'local_model_did_not_select_registration_hypothesis',
                          'invalid_local_model_registration_hypothesis',
                          'missing_valid_rigid_frame_hypothesis','insufficient_source_frame_spread',
                          'source_section_has_no_nearby_CAD_intersection','candidate_on_invalid_display_face',
                          'item_point_residual_rejected'} for failure in failures))):
                # The local multimodal model gets the method/drawing pages and
                # a CAD orthographic atlas, then returns a view pixel. Geometry
                # code only converts that model-selected pixel to a ray hit.
                # This path is also available when XYZ exists but no complete,
                # model-authorized rigid placement survived geometric checks.
                # Include every recovered XYZ as evidence, and require the
                # complete MSOP-declared location count when it is explicit.
                source_points=_source_point_evidence(rows)
                previous_geometry_result=record.get('result')
                registration_fallback=bool(rows and (reason is None or failures))
                registration_diagnostic=None
                if rows:
                    from collections import Counter
                    failure_counts=Counter(failure.get('reason') for failure in failures)
                    registration_diagnostic=dict(
                        geometry_result=previous_geometry_result,
                        rejected_point_count=len(failures),
                        rejected_point_reasons=dict(failure_counts),
                        rejected_point_names=[failure.get('name') for failure in failures[:40]],
                        no_geometric_anchor_authorized=True,
                        frame_transform_hypotheses=_registration_frame_hypotheses(
                            receipt,_source_point_scopes(source_points)),
                        selected_frame_transforms=[transform for scope in sorted(
                            _source_point_scopes(source_points),key=str)
                            if (transform:=selected_model_transform(scope)) is not None])
                candidate=dict(fai=item['number'],anchors=[],
                    note=('已提取 XYZ，但当前没有通过模型授权与全组几何校验的完整放置；等待本地视觉模型逐点定位。'
                          if registration_fallback else
                          'MSOP XYZ 点位不完整；等待本地视觉模型结合原文、已有 XYZ 与 CAD 六视图补齐完整点组。'
                          if rows else 'MSOP 未提取到 XYZ；等待本地视觉模型结合原文与 CAD 六视图定位。'),
                    candidate_kind='model_visual_localization',origin='msop_visual_candidate',
                    evidence=dict(cad_sha256=mesh['source_sha256'],msop_sha256=msop['sha256'],
                                  source_pages=item.get('pages',[]),placement_decision='local_model_required',
                                  required_marker_count=required_marker_count,
                                  required_marker_groups=required_marker_groups,
                                  required_point_count=required_point_count,known_source_points=source_points,
                                  registration_diagnostic=registration_diagnostic,
                                  prior_stale_seed=(seed.get('seed_evidence') if seed else None)))
                candidate['required_marker_count']=required_marker_count
                candidate['required_marker_groups']=required_marker_groups
                candidate['required_point_count']=required_point_count
                candidate['known_source_points']=source_points
                candidates.append(candidate)
                if registration_fallback:
                    record['geometry_result']=previous_geometry_result
                    record['result']='local_visual_registration_fallback_ready'
                else:
                    record['result']='local_visual_candidate_ready' if not rows else 'local_visual_completion_candidate_ready'
                    record['coordinate_status']=reason
            elif seed:
                candidates.append(seed)
                record['geometry_result']=record['result']
                record['result']='same_source_stale_seed_ready'
                record['stale_seed_evidence']=seed['seed_evidence']
        report.append(record)
    result=dict(candidates=candidates,items=report,persisted=persist,
                registration_status=registration_status,
                source_to_cad_registered=False,metrology_accepted=False)
    if registration_path is None and not persist:
        # Compare whole-item hypotheses; never mix transforms point by point
        # or overwrite earlier annotations. The local model still reviews
        # every selected hypothesis before any pin can be written.
        records={r['fai']:r for r in report};ready={c['fai']:c for c in candidates}
        alternatives=store._dir(project_id)/'automatic-learning'/'registration-alternatives'
        for alternate in sorted(alternatives.glob('*.json')):
            candidate_receipt=json.loads(alternate.read_text())
            if (candidate_receipt.get('cad_sha256')!=mesh['source_sha256']
                    or candidate_receipt.get('msop_sha256')!=msop['sha256']
                    or candidate_receipt.get('algorithm_sha256')!=current_algorithm_sha256):continue
            alternate_result=build_candidates(store,project_id,registration_path=alternate,persist=False)
            for candidate in alternate_result['candidates']:
                previous=ready.get(candidate['fai'])
                if previous is None or candidate['max_point_residual_mm']<previous['max_point_residual_mm']:
                    ready[candidate['fai']]=candidate
                    records[candidate['fai']]=next(r for r in alternate_result['items'] if r['fai']==candidate['fai'])
        result.update(candidates=[ready[n] for n in sorted(ready)],items=[records[n] for n in sorted(records)])
    return result
