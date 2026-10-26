"""Local-only FAI candidate selection using the shared Talker model and an isolated adapter."""
from __future__ import annotations

import json
import functools
import hashlib
import itertools
import re
from contextlib import nullcontext
from urllib.parse import urlparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from talker.resolver import discover_latest_adapter
from .candidate_mining import extract_feature_labels, extract_named_points, referenced_points
from .msop_coordinates import (_explicit_measure_point_names, _measure_point_basis,
                               MEASURE_POINT_LIST, MEASURED_GEOMETRY_LIST)
from .cad_view_atlas import (ray_hit_view, render_cad_view_atlas,
                             render_focused_view_retry, nearby_surface_view_candidates,
                             distributed_surface_view_candidates,
                             render_surface_candidate_atlas,project_point_to_view_uv,
                             cached_surface_geometry)
from .annotations import validate_anchors, MAX_ANNOTATION_POINTS
from .cad_view_atlas import _valid_triangles
from .section_geometry import closest_plane_section
from .feature_view_candidates import feature_view_candidates
from .spacing_candidates import evenly_spaced_candidates


VIEWER_LOCATION_TASK=(
    '当前任务是识别并标注 FAI 在产品上的实体作用位置，不是判断 AOI 是否能执行该测量。'
    '拉力、扭矩、剪切、焊点和破坏性试验也需要在 Viewer 中定位相应螺母、焊片或试验区域。'
    '原文写不能由某测量机测量，并不表示该位置不存在。可使用有明确原文特征关联的完整 XYZ 支撑点组来定位试验对象，'
    '但必须说明这些是位置支撑点，不是该力学试验的几何测量点；不能宣称相机可测力、AOI 可完成该试验或计量已经通过。'
    '仍须核对特征名称、源页、点组和 CAD 实体；位置对应不明确或几何不通过时必须弃答。')


def _complete_visual_context_budget(messages,*,limit=58000):
    """Budget the current retry, including every original source attachment.

    A retry adds geometry diagnostics and candidate tables to the question.
    Reusing the initial question's budget silently displaced source text in
    Talker's general conversational excerpting path. Viewer evidence must be
    supplied whole or fail explicitly before a decision is requested.
    """
    total=0
    for message in messages:
        role='User' if message.get('role')=='user' else 'Assistant'
        total+=len(role)+2+len((message.get('content') or '').strip())+2
        for attachment in message.get('attachments',[]):
            parsed=(attachment.get('parsed_text') or '').strip()
            if parsed:
                total+=len(f"[Attachment: {attachment['filename']}]\n")+len(parsed)+1
    if total>limit:
        raise ValueError(f'Complete local visual evidence exceeds context budget: {total} > {limit}; no truncated decision accepted')
    return total+512


def _rigid_correspondence(source_xyz, cad_xyz, *, max_residual_mm=0.75):
    """Fit one proper rigid source-to-CAD transform and reject weak point groups."""
    source=np.asarray(source_xyz,dtype=float);target=np.asarray(cad_xyz,dtype=float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<3:
        return None
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        return None
    source_center=source.mean(axis=0);target_center=target.mean(axis=0)
    a=source-source_center;b=target-target_center
    if np.linalg.matrix_rank(a,tol=1e-7)<2 or np.linalg.matrix_rank(b,tol=1e-7)<2:
        return None
    u,_singular,vt=np.linalg.svd(a.T@b)
    correction=np.eye(3);correction[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@correction@u.T
    translation=target_center-rotation@source_center
    projected=source@rotation.T+translation
    residuals=np.linalg.norm(projected-target,axis=1)
    if float(residuals.max())>max_residual_mm:
        return None
    return dict(rotation=rotation,translation=translation,residuals=residuals,
                max_residual_mm=float(residuals.max()),rms_residual_mm=float(np.sqrt(np.mean(residuals**2))))


def _xyz_rigid_fit_required_scopes(resolved_sources):
    """Scopes whose measured XYZ geometry is sufficient to require a CAD fit."""
    grouped={}
    for source in resolved_sources.values():
        scope=(source.get('program'),source.get('coordinate_system'))
        xyz=source.get('xyz')
        if None in scope or not isinstance(xyz,list) or len(xyz)!=3:
            continue
        grouped.setdefault(scope,[]).append(np.asarray(xyz,float))
    required=set()
    for scope,points in grouped.items():
        array=np.asarray(points,float)
        if (len(array)>=3 and np.isfinite(array).all()
                and np.linalg.matrix_rank(array-array.mean(axis=0),tol=1e-7)>=2):
            required.add(scope)
    return required


def _missing_required_xyz_fits(scopes, required_scopes, rigid_fits):
    """Return model-selected coordinate scopes that cannot fall back to pixels."""
    return sorted((scope for scope in scopes
                   if scope in required_scopes and scope not in rigid_fits),key=str)


def _registration_reference_selection_valid(known_source_points, registration_points,
                                            coordinate_choices, markers):
    """Validate a sparse registration reference set while preserving every FAI point.

    Registration correspondences only establish the rigid frame; they need not
    duplicate every measured point. Completeness is enforced separately by
    requiring the marker groups to cover every distinct source point.
    """
    expected={row.get('name'):row for row in known_source_points if row.get('name')}
    if len(expected)!=len(known_source_points) or not expected:
        return False,{'reason':'invalid_or_duplicate_source_point_names'}
    choices={row.get('name'):row for row in coordinate_choices if isinstance(row,dict)}
    by_scope={}
    for name,point in expected.items():
        choice=choices.get(name) if point.get('requires_variant_choice') else None
        xyz=choice.get('xyz') if choice else point.get('xyz')
        scope=(choice.get('program') if choice else point.get('program'),
               choice.get('coordinate_system') if choice else point.get('coordinate_system'))
        if (not isinstance(xyz,list) or len(xyz)!=3 or any(
                isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value)
                for value in xyz) or None in scope):
            return False,{'reason':'unresolved_source_point_scope_or_xyz','point':name}
        by_scope.setdefault(scope,[]).append(np.asarray(xyz,float))
    if not isinstance(registration_points,list):
        return False,{'reason':'registration_references_missing'}
    reference_names=[row.get('source_point_name') for row in registration_points
                     if isinstance(row,dict)]
    if (len(reference_names)!=len(registration_points) or not reference_names
            or len(set(reference_names))!=len(reference_names)
            or not set(reference_names)<=set(expected)):
        return False,{'reason':'invalid_registration_reference_names'}
    refs_by_scope={}
    for row in registration_points:
        name=row.get('source_point_name')
        point=expected[name]
        choice=choices.get(name) if point.get('requires_variant_choice') else None
        scope=(choice.get('program') if choice else point.get('program'),
               choice.get('coordinate_system') if choice else point.get('coordinate_system'))
        if (row.get('view_id') not in ('+X','-X','+Y','-Y','+Z','-Z')
                or not isinstance(row.get('uv'),list) or len(row['uv'])!=2
                or any(isinstance(value,bool) or not isinstance(value,(int,float))
                       or not 0<=value<=1 for value in row['uv'])):
            return False,{'reason':'invalid_registration_reference_view','point':name}
        refs_by_scope.setdefault(scope,[]).append(np.asarray(
            choice.get('xyz') if choice else point.get('xyz'),float))
    for scope,points in by_scope.items():
        if len(points)<3:
            continue
        refs=refs_by_scope.get(scope,[])
        required=min(4,len(points))
        if len(refs)<required or np.linalg.matrix_rank(np.asarray(refs)-np.mean(refs,axis=0),tol=1e-6)<2:
            return False,{'reason':'insufficient_or_collinear_registration_references',
                          'scope':scope,'required':required,'selected':len(refs)}
    covered=[]
    for marker in markers:
        names=marker.get('source_point_names') if isinstance(marker,dict) else None
        if (not isinstance(names,list) or not names or len(set(names))!=len(names)
                or not set(names)<=set(expected)):
            return False,{'reason':'invalid_marker_source_point_group'}
        covered.extend(names)
    if set(covered)!=set(expected):
        return False,{'reason':'marker_groups_do_not_cover_every_source_point',
                      'source_point_count':len(expected),'covered_point_count':len(set(covered)),
                      'missing':sorted(set(expected)-set(covered))[:20]}
    return True,{'reference_count':len(reference_names),'source_point_count':len(expected),
                 'reference_counts_by_scope':{str(scope):len(rows) for scope,rows in refs_by_scope.items()}}


def _visual_marker_validation_issues(parsed, *, fai, required_marker_count,
                                     known_source_points, rigid_group_possible,
                                     candidate_selection_valid, coordinate_choices_valid,
                                     truncated,candidate_selection_audit=None):
    """Return concrete, machine-checkable reasons a visual decision needs repair."""
    issues=[]
    if truncated: issues.append({'code':'truncated_output'})
    decision=re.sub(r'[^A-Z_]','',str(parsed.get('decision') or '').upper())
    if decision not in ('AUTO_MARK','AUTOMARK'):
        issues.append({'code':'decision_not_auto_mark','decision':decision or None})
    if _fai_number(parsed.get('fai'))!=fai:
        issues.append({'code':'wrong_fai_number','expected':fai,'actual':parsed.get('fai')})
    markers=parsed.get('markers')
    if not isinstance(markers,list) or not markers:
        issues.append({'code':'markers_missing_or_empty'})
    elif required_marker_count and len(markers)!=required_marker_count:
        issues.append({'code':'required_marker_count_mismatch','expected':required_marker_count,
                       'actual':len(markers)})
    if (rigid_group_possible or any(point.get('explicit_method_manifest')
                                    for point in known_source_points)) and isinstance(markers,list):
        expected={point.get('name') for point in known_source_points if point.get('name')}
        observed={name for marker in markers if isinstance(marker,dict)
                  for name in marker.get('source_point_names',[]) if isinstance(name,str)}
        missing=sorted(expected-observed)
        if missing:
            issues.append({'code':'source_point_coverage_incomplete','missing_names':missing[:80],
                           'missing_count':len(missing),'expected_count':len(expected)})
    if not coordinate_choices_valid:
        issues.append({'code':'coordinate_choices_incomplete_or_invalid'})
    if not candidate_selection_valid:
        issue={'code':'candidate_id_selection_invalid'}
        if candidate_selection_audit:
            issue['details']=candidate_selection_audit[:20]
        issues.append(issue)
    return issues


def _complete_marker_registration_references(known_source_points, markers):
    """Use model-selected one-to-one measurement markers as a complete frame fit.

    This avoids forcing the vision model to emit a second, redundant sparse
    reference list when it has already selected every measured XYZ point on
    the CAD. The model still owns every point-to-view correspondence; this
    helper only repackages those same correspondences for geometric checking.
    """
    expected={row.get('name') for row in known_source_points if row.get('name')}
    if not expected or len(expected)!=len(known_source_points) or not isinstance(markers,list):
        return None
    references=[];seen=set()
    for marker in markers:
        if not isinstance(marker,dict):
            return None
        names=marker.get('source_point_names')
        if not names:
            continue
        if (not isinstance(names,list) or len(names)!=1 or names[0] not in expected
                or names[0] in seen or marker.get('view_id') not in ('+X','-X','+Y','-Y','+Z','-Z')
                or not isinstance(marker.get('uv'),list) or len(marker['uv'])!=2
                or any(isinstance(value,bool) or not isinstance(value,(int,float))
                       or not 0<=value<=1 for value in marker['uv'])):
            return None
        seen.add(names[0])
        references.append(dict(source_point_name=names[0],view_id=marker['view_id'],uv=marker['uv']))
    return references if seen==expected else None


def _collapsed_source_marker_groups(markers,resolved_sources):
    """Find separate XYZ markers the model collapsed to one view pixel."""
    groups={}
    for marker in markers:
        names=marker.get('source_point_names') or []
        if len(names)!=1 or not resolved_sources or names[0] not in resolved_sources:
            continue
        groups.setdefault((marker.get('view_id'),round(float(marker['uv'][0]),5),
            round(float(marker['uv'][1]),5)),[]).append(marker)
    collapsed=[]
    for (view_id,u,v),group in groups.items():
        names=[row['source_point_names'][0] for row in group]
        points=np.asarray([resolved_sources[name]['xyz'] for name in names],float)
        if (len(group)>1 and len(set(names))==len(names) and
                np.max(np.linalg.norm(points-points[0],axis=1))>1e-5):
            collapsed.append((view_id,u,v,group))
    return collapsed


def _rigid_source_anchors(mesh, source_xyz, correspondence_xyz, *, max_fit_residual_mm=0.75,
                          max_surface_residual_mm=0.35):
    """Place a complete MSOP point group through one transform, preserving its geometry."""
    fit=_rigid_correspondence(source_xyz,correspondence_xyz,max_residual_mm=max_fit_residual_mm)
    if fit is None:
        return None
    triangles,face_ids,index=cached_surface_geometry(mesh)
    source=np.asarray(source_xyz,dtype=float)
    predicted=source@fit['rotation'].T+fit['translation']
    anchors=[];surface_residuals=[]
    for point in predicted:
        hit=index.nearest(point)
        if not np.isfinite(hit['distance_mm']) or hit['distance_mm']>max_surface_residual_mm:
            return None
        tri_index=int(hit['triangle_index'])
        if tri_index<0 or tri_index>=len(face_ids):
            return None
        anchors.append(dict(face_id=str(face_ids[tri_index]),position=np.asarray(hit['point'],float).tolist(),
                            label_offset=[32,-26]))
        surface_residuals.append(float(hit['distance_mm']))
    placed=np.asarray([anchor['position'] for anchor in anchors])
    shape_error=np.linalg.norm((placed[:,None,:]-placed[None,:,:])-
                               (predicted[:,None,:]-predicted[None,:,:]),axis=2)
    if float(shape_error.max())>max_surface_residual_mm*2:
        return None
    fit.update(surface_residuals_mm=surface_residuals,
               max_surface_residual_mm=float(max(surface_residuals,default=0.0)),
               max_pairwise_shape_error_mm=float(shape_error.max()))
    return anchors,fit


def _object(text: str):
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
        if isinstance(value,list) and len(value)==1 and isinstance(value[0],dict):
            value=value[0]
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group())
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _source_scope_conflicts(old, detail, method_pages, item):
    """Validate the exact source rows used by a geometry candidate.

    XYZ candidates are generated from page-scoped MSOP coordinates and may
    resolve repeated labels using sibling nominal dimensions. Re-expanding
    them from feature labels alone would discard that evidence and reintroduce
    already-resolved cross-page conflicts.
    """
    if old.get('automatic_xyz_hypothesis'):
        evidence = old.get('evidence') or {}
        rows = evidence.get('point_correspondences') or []
        valid = (evidence.get('all_source_points_retained') is True
                 and evidence.get('source_sections_preserved') is True and bool(rows))
        for row in rows:
            key = row.get('point_key')
            pages = row.get('source_pages') or []
            xyz = row.get('source_xyz')
            if (not key or 'unresolved' in str(key).lower() or not pages
                    or not row.get('program') or not row.get('coordinate_system')
                    or not isinstance(xyz, list) or len(xyz) != 3):
                valid = False
                break
        return [] if valid else ['candidate_source_scope_not_proven']
    scoped = referenced_points(None, detail['methods'], extract_named_points(method_pages),
                               extract_feature_labels(detail['methods'], item.get('features', [])))
    return [p['name'] for p in scoped.values() if p.get('coordinate_conflict')]


def _confidence(value):
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in ('high', '高', '高置信'):
        return 'high'
    if normalized in ('medium', '中', '中置信', 'moderate'):
        return 'medium'
    if normalized in ('low', '低', '低置信'):
        return 'low'
    return None


def _resolve_surface_candidate_ids(parsed, surface_candidates):
    """Replace model-copied candidate labels with their exact verified ray hits.

    The local model remains responsible for assigning a visible CAD candidate
    to an MSOP feature. This adapter only resolves the selected candidate ID
    to the UV/view pair that produced a verified CAD ray hit.
    """
    if not isinstance(parsed,dict) or not isinstance(surface_candidates,list):
        return parsed
    by_id={}
    for candidate in surface_candidates:
        if not isinstance(candidate,dict):
            continue
        identifier=candidate.get('id')
        if identifier is None:
            continue
        by_id[str(identifier).strip().upper().removeprefix('C')]=candidate
    resolved=dict(parsed)
    for field in ('markers','registration_points'):
        rows=parsed.get(field)
        if not isinstance(rows,list):
            continue
        updated=[]
        for row in rows:
            if not isinstance(row,dict):
                updated.append(row)
                continue
            candidate_id=row.get('candidate_id',row.get('candidate'))
            key=(str(candidate_id).strip().upper().removeprefix('C')
                 if candidate_id is not None else None)
            candidate=by_id.get(key) if key else None
            if candidate:
                row=dict(row)
                row['view_id']=candidate.get('view_id')
                row['uv']=candidate.get('uv')
            updated.append(row)
        resolved[field]=updated
    return resolved


def _candidate_prompt_payload(candidates):
    """Intern repeated name lists without removing candidates or geometry."""
    groups={};indices={};rows=[]
    for original in candidates:
        row=dict(original)
        names=row.get('source_point_names')
        if isinstance(names,list):
            key=json.dumps(names,ensure_ascii=False,separators=(',',':'))
            if key not in indices:
                identifier=f'G{len(indices)+1}'
                indices[key]=identifier;groups[identifier]=names
            row['source_point_names']={'group':indices[key]}
        rows.append(row)
    return {'encoding':'exact-shared-source-name-groups',
            'source_point_groups':groups,'candidates':rows}


def _surface_candidate_selection_valid(parsed, surface_candidates):
    """Require exact candidate IDs for source points covered by retry choices."""
    return not _surface_candidate_selection_audit(parsed,surface_candidates)


def _surface_candidate_selection_audit(parsed, surface_candidates,*,require_marker_ids=False):
    """Explain candidate-ID mismatches by field, source point, and allowed IDs."""
    if not surface_candidates:
        return []
    candidate_sources={}
    valid_ids=set()
    for candidate in surface_candidates:
        if not isinstance(candidate,dict) or candidate.get('id') is None:
            continue
        key=str(candidate['id']).strip().upper().removeprefix('C')
        valid_ids.add(key)
        for name in candidate.get('source_point_names') or []:
            candidate_sources.setdefault(name,set()).add(key)
    issues=[]
    selected_marker_ids=set()
    for field in ('markers','registration_points'):
        rows=parsed.get(field)
        if not isinstance(rows,list):
            continue
        for row in rows:
            if not isinstance(row,dict):
                continue
            names=(row.get('source_point_names') if field=='markers'
                   else [row.get('source_point_name')])
            candidate_id=row.get('candidate_id',row.get('candidate'))
            key=(str(candidate_id).strip().upper().removeprefix('C')
                 if candidate_id is not None else None)
            if field=='markers' and require_marker_ids and key in selected_marker_ids:
                issues.append({'field':field,'selected_candidate_id':candidate_id,
                    'reason':'candidate_id_reused_for_distinct_markers'})
            if field=='markers' and key:selected_marker_ids.add(key)
            if (candidate_id is not None and key not in valid_ids or
                    field=='markers' and require_marker_ids and candidate_id is None):
                issues.append({'field':field,'selected_candidate_id':candidate_id,
                    'reason':'explicit_valid_candidate_id_required',
                    'valid_candidate_ids':['C'+value for value in sorted(valid_ids)]})
                continue
            for name in names or []:
                allowed=sorted(candidate_sources.get(name,set()),key=lambda value:(not value.isdigit(),int(value) if value.isdigit() else value))
                # Candidate atlases are optional search hints. A model may
                # instead make an explicit view/UV choice; exact CAD ray hits,
                # rigid XYZ fitting, and full-group gates still validate it.
                # If it does provide an ID, however, that ID must match this
                # exact source point.
                if candidate_id is not None and (key not in valid_ids or
                        (allowed and key not in candidate_sources.get(name,set()))):
                    issues.append({'field':field,'source_point_name':name,
                        'selected_candidate_id':candidate_id,'valid_candidate_ids':['C'+value for value in allowed]})
                    if len(issues)>=40:
                        return issues
    return issues


def _compact_registration_diagnostic(value):
    """Keep model-selected frame evidence useful without repeating its full history."""
    if not isinstance(value,dict):
        return value
    result=dict(value)
    hypotheses=result.pop('frame_transform_hypotheses',None)
    if isinstance(hypotheses,list):
        result['additional_frame_hypothesis_count']=len(hypotheses)
    # This flag describes rejection of an earlier geometry-generated proposal;
    # phrased as an authorization rule it can cause the visual model to abstain
    # even when it can independently identify every measured point.
    result.pop('no_geometric_anchor_authorized',None)
    transforms=result.get('selected_frame_transforms')
    if isinstance(transforms,list):
        compact=[]
        keep=('program','coordinate_system','rotation','translation_mm','p95_residual_mm',
              'max_residual_mm','held_out_max_mm','inlier_ratio','initialization',
              'hypothesis_id','model_confidence')
        for row in transforms:
            if not isinstance(row,dict):
                continue
            selected={key:row[key] for key in keep if key in row}
            evidence=row.get('model_evidence')
            if isinstance(evidence,list):
                selected['model_evidence']=[str(entry)[:180] for entry in evidence[:3]]
            elif isinstance(evidence,str):
                selected['model_evidence']=evidence[:540]
            compact.append(selected)
        result['selected_frame_transforms']=compact
    return result


def _source_point_family(name):
    """Keep numbered bare point rows together without merging entity prefixes.

    Q38/Q39 are successive point IDs. F1_PNT_Z1 and F2_PNT_Z1 instead
    explicitly identify different named feature families via their prefixes.
    """
    name=str(name or '')
    if '_' in name:return name.split('_',1)[0]
    match=re.fullmatch(r'([A-Za-z]+)\d+',name)
    return match.group(1) if match else name


def _project_model_selected_xyz_group(mesh, source_points, selected_frames, detail, coordinate_choices):
    """Project a complete FAI XYZ group through model-selected rigid frames.

    The local model selected the frame transforms upstream. This geometry-only
    step retains every source point, projects each one to a valid CAD triangle,
    and rejects groups whose point shape or common source sections drift.
    """
    choices={row['name']:row for row in coordinate_choices}
    transforms={}
    for frame in selected_frames:
        scope=(frame.get('program'),frame.get('coordinate_system'))
        rotation=np.asarray(frame.get('rotation'),float)
        translation=np.asarray(frame.get('translation_mm'),float)
        if (None in scope or rotation.shape!=(3,3) or translation.shape!=(3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5)
                or not np.isclose(np.linalg.det(rotation),1,atol=1e-5)):
            return None,{'reason':'invalid_model_selected_frame_transform'}
        transforms[scope]=(rotation,translation,frame)
    resolved=[]
    for point in source_points:
        selected=choices.get(point.get('name'))
        xyz=selected.get('xyz') if selected else point.get('xyz')
        program=selected.get('program') if selected else point.get('program')
        coordinate_system=selected.get('coordinate_system') if selected else point.get('coordinate_system')
        page=selected.get('page') if selected else (point.get('pages') or [None])[0]
        if (not isinstance(xyz,list) or len(xyz)!=3 or any(
                isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value)
                for value in xyz)):
            return None,{'reason':'source_point_variant_unresolved','point':point.get('name')}
        scope=(program,coordinate_system)
        if scope not in transforms:
            return None,{'reason':'model_selected_transform_missing','scope':scope}
        resolved.append(dict(name=point['name'],xyz=np.asarray(xyz,float),pages=point.get('pages') or [page],
                             program=program,coordinate_system=coordinate_system))
    triangles,face_ids,index=cached_surface_geometry(mesh)
    anchors=[];projection_rows=[];by_scope={}
    prepared=[];section_residuals={}
    for row in resolved:
        scope=(row['program'],row['coordinate_system'])
        rotation,translation,frame=transforms[scope]
        target=row['xyz']@rotation.T+translation
        hit=index.nearest(target)
        distance=float(hit.get('distance_mm',np.inf))
        if not np.isfinite(distance) or distance>0.55:
            return None,{'reason':'point_to_surface_distance_exceeded_display_gate',
                         'point':row['name'],'distance_mm':distance,'gate_mm':0.55}
        normal=rotation@np.array([0.0,0.0,1.0])
        offset=float(round(float(row['xyz'][2]),3)+normal@translation)
        feature=_source_point_family(row['name'])
        key=(scope,feature,round(float(row['xyz'][2]),3))
        section_residuals.setdefault(key,[]).append(float(normal@hit['point'])-offset)
        prepared.append((row,target,hit,distance,normal,offset,key))
    needs_section=set()
    for key,values in section_residuals.items():
        residuals=np.asarray(values,float);center=float(np.median(residuals))
        if abs(center)>0.55 or float(np.max(np.abs(residuals-center)))>0.08:
            needs_section.add(key)
    for row,target,hit,distance,normal,offset,key in prepared:
        scope=(row['program'],row['coordinate_system'])
        rotation,translation,frame=transforms[scope]
        # Preserve a coherent nearest-surface section when it already passes
        # the common-plane gate. Intersecting every point with a nominal plane
        # a few microns above a real flat face can otherwise move interior
        # points sideways to an unrelated vertical edge. Only repair groups
        # with actual height drift; the final whole-group gates stay unchanged.
        if key in needs_section or frame.get('section_refined'):
            nearby=index.nearby_triangles(target,0.55)
            section=closest_plane_section(target,triangles[nearby],normal,offset) if len(nearby) else None
            section_gate=0.15 if frame.get('section_refined') else 0.55
            if section is not None and section.get('distance_mm',np.inf)<=section_gate:
                hit=dict(section,triangle_index=int(nearby[section['triangle_index']]),
                         projection_method='model_selected_frame_source_section')
                distance=float(hit['distance_mm'])
            elif frame.get('section_refined'):
                return None,{'reason':'refined_source_section_residual_exceeded_gate',
                             'point':row['name'],'section_residual_mm':(section or {}).get('distance_mm'),
                             'gate_mm':0.15}
        triangle_index=int(hit['triangle_index'])
        if not 0<=triangle_index<len(face_ids):
            return None,{'reason':'invalid_projected_triangle','point':row['name']}
        position=np.asarray(hit['point'],float)
        anchor=dict(face_id=str(face_ids[triangle_index]),position=position.tolist(),label_offset=[32,-26])
        anchors.append(anchor)
        projection_rows.append(dict(name=row['name'],program=row['program'],
            coordinate_system=row['coordinate_system'],source_xyz=row['xyz'].tolist(),
            transformed_xyz=target.tolist(),position=position.tolist(),
            surface_residual_mm=distance,source_pages=row['pages']))
        by_scope.setdefault(scope,[]).append((row,target,position,distance))
    scope_audits=[]
    for scope,items in by_scope.items():
        positions=np.asarray([item[2] for item in items],float)
        predicted=np.asarray([item[1] for item in items],float)
        pair_error=np.linalg.norm((positions[:,None,:]-positions[None,:,:])-
                                  (predicted[:,None,:]-predicted[None,:,:]),axis=2)
        max_shape_error=float(pair_error.max(initial=0.0))
        if max_shape_error>0.70:
            return None,{'reason':'projected_group_shape_drift_exceeded_gate',
                         'scope':scope,'max_pairwise_shape_error_mm':max_shape_error,'gate_mm':0.70}
        rotation,translation,frame=transforms[scope]
        normal=rotation@np.array([0.0,0.0,1.0])
        plane_rows=[]
        # A coordinate system can contain several named measured features at
        # the same source Z (for example PLASTIC and CHASSIS point families,
        # or a point plus a separately constructed line). Keep the coplanarity
        # invariant within each MSOP feature family instead of incorrectly
        # forcing every feature in the FAI onto one physical plane.
        feature_sections={}
        for item in items:
            name=str(item[0].get('name') or '')
            feature=_source_point_family(name)
            source_z=round(float(item[0]['xyz'][2]),3)
            feature_sections.setdefault((feature,source_z),[]).append(item)
        for (feature,source_z),selected in sorted(feature_sections.items()):
            offset=source_z+float(normal@translation)
            residuals=np.asarray([float(normal@item[2])-offset for item in selected],float)
            center=float(np.median(residuals))
            spread=float(np.max(np.abs(residuals-center))) if len(residuals) else 0.0
            if abs(center)>0.55 or spread>0.08:
                return None,{'reason':'source_section_coplanarity_not_preserved','scope':scope,
                             'feature_group':feature,'source_z_mm':source_z,
                             'common_offset_mm':center,'section_spread_mm':spread,
                             'spread_gate_mm':0.08}
            plane_rows.append(dict(feature_group=feature,source_z_mm=source_z,
                                   common_offset_mm=center,residual_spread_mm=spread,
                                   point_count=len(selected)))
        scope_audits.append(dict(program=scope[0],coordinate_system=scope[1],
            hypothesis_id=frame.get('hypothesis_id'),model_confidence=frame.get('model_confidence'),
            fit_p95_residual_mm=frame.get('p95_residual_mm'),
            fit_max_residual_mm=frame.get('max_residual_mm'),
            point_count=len(items),max_surface_residual_mm=max(item[3] for item in items),
            max_pairwise_shape_error_mm=max_shape_error,source_sections=plane_rows,
            source_to_cad_registered=False,metrology_accepted=False))
    return (anchors,projection_rows,scope_audits),None


def _planar_frame_hypotheses(mesh, source_points, selected_frames, *,
                              max_rotation_deg=0.5,max_translation_mm=0.8):
    """Propose bounded rigid alignment to nearby CAD planes for model review.

    Numerical ICP poses may put a nominal source plane microns above an actual
    flat face. Aligning the entire frame to that face preserves every source
    vector; it is not a per-point Z edit. Geometry only generates alternatives:
    the complete FAI still needs the local model's explicit candidate choice.
    """
    if not source_points or any(point.get('requires_variant_choice') for point in source_points):
        return []
    try:
        triangles,_face_ids,index=cached_surface_geometry(mesh)
    except (ValueError,TypeError):
        return []
    options=[];has_alternative=False
    for original in selected_frames:
        scope=(original.get('program'),original.get('coordinate_system'))
        rows=[point for point in source_points
              if (point.get('program'),point.get('coordinate_system'))==scope
              and isinstance(point.get('xyz'),list) and len(point['xyz'])==3]
        choices=[original]
        if len(rows)<3:
            options.append(choices);continue
        rotation=np.asarray(original.get('rotation'),float)
        translation=np.asarray(original.get('translation_mm'),float)
        if rotation.shape!=(3,3) or translation.shape!=(3,):
            options.append(choices);continue
        source_normal=rotation[:,2]
        planes={}
        for point in rows:
            target=np.asarray(point['xyz'],float)@rotation.T+translation
            hit=index.nearest(target)
            if hit.get('distance_mm',np.inf)>max_translation_mm:
                continue
            triangle=triangles[int(hit['triangle_index'])]
            normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
            length=float(np.linalg.norm(normal))
            if length<1e-12:
                continue
            normal/=length
            if normal@source_normal<0:normal=-normal
            angle=float(np.arccos(np.clip(normal@source_normal,-1.0,1.0)))
            if angle>np.deg2rad(max_rotation_deg):
                continue
            key=tuple(np.round(normal,6))
            planes.setdefault(key,[]).append((normal,
                float(normal@triangle[0]-point['xyz'][2]),point['name']))
        candidates=[]
        for samples in planes.values():
            clusters=[]
            for sample in sorted(samples,key=lambda sample:sample[1]):
                if clusters and abs(sample[1]-float(np.median([s[1] for s in clusters[-1]])))<=0.01:
                    clusters[-1].append(sample)
                else:
                    clusters.append([sample])
            for cluster in clusters:
                if len(cluster)<3:continue
                normal=cluster[0][0]
                cross=np.cross(source_normal,normal);sin=float(np.linalg.norm(cross))
                angle=float(np.arctan2(sin,float(source_normal@normal)))
                delta=(Rotation.from_rotvec(cross/sin*angle).as_matrix()
                       if sin>1e-12 else np.eye(3))
                new_rotation=delta@rotation
                offset=float(np.median([s[1] for s in cluster]))
                shift=offset-float(normal@translation)
                if abs(shift)>max_translation_mm:continue
                new_translation=translation+normal*shift
                if np.allclose(new_rotation,rotation,atol=1e-10) and abs(shift)<1e-9:
                    continue
                audit={'method':'bounded_rigid_alignment_to_CAD_plane',
                    'plane_normal':normal.tolist(),'frame_plane_offset_mm':offset,
                    'support_point_names':[s[2] for s in cluster],
                    'delta_rotation_deg':float(np.rad2deg(angle)),
                    'delta_translation_mm':abs(float(shift)),
                    'source_to_cad_registered':False,'metrology_accepted':False}
                frame={**original,'rotation':new_rotation.tolist(),
                       'translation_mm':new_translation.tolist(),'planar_alignment':audit}
                frame.pop('section_refined',None)
                candidates.append((len(cluster),frame))
        for _support,frame in sorted(candidates,key=lambda pair:-pair[0])[:3]:
            choices.append(frame);has_alternative=True
        options.append(choices)
    if not has_alternative:return []
    return [list(frames) for frames in itertools.islice(itertools.product(*options),32)
            if any('planar_alignment' in frame for frame in frames)]


def _normal_origin_frame_hypotheses(mesh,source_points,selected_frames,*,step_mm=0.05):
    """Search only the frame-normal origin inside the actual CAD envelope.

    Some source alignments reuse their XY definition at different Z origins.
    Keep the model-selected in-plane pose and all source XYZ vectors intact.
    Search complete groups, retain only existing geometry-gate survivors, and
    leave the final frame choice to the local model. This is display recovery,
    never a claim that the source datum origin has been metrologically proved.
    """
    if not source_points or any(point.get('requires_variant_choice') for point in source_points):
        return []
    try:
        triangles,_ids,index=cached_surface_geometry(mesh)
    except (ValueError,TypeError):
        return []
    options=[];recovered=False
    for original in selected_frames:
        scope=(original.get('program'),original.get('coordinate_system'))
        rows=[point for point in source_points if (point.get('program'),point.get('coordinate_system'))==scope]
        if not rows:
            options.append([original]);continue
        if any(not isinstance(row.get('xyz'),list) or len(row['xyz'])!=3 for row in rows):
            return []
        source=np.asarray([row['xyz'] for row in rows],float)
        seeds=[original,*[trial[0] for trial in _planar_frame_hypotheses(mesh,rows,[original])]]
        orientations=set();ranked=[]
        for seed in seeds:
            rotation=np.asarray(seed.get('rotation'),float);translation=np.asarray(seed.get('translation_mm'),float)
            if (rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all()
                    or not np.isfinite(translation).all()):
                continue
            identity=tuple(np.round(rotation.ravel(),7))
            if identity in orientations:continue
            orientations.add(identity)
            normal=rotation[:,2]
            cad_levels=triangles.reshape(-1,3)@normal
            lo=float(cad_levels.min()-source[:,2].min())
            hi=float(cad_levels.max()-source[:,2].max())
            if hi<lo:continue
            count=min(160,max(2,int(np.ceil((hi-lo)/step_mm))+1))
            for level in np.unique(np.linspace(lo,hi,count)):
                target_translation=translation+normal*(float(level)-float(normal@translation))
                targets=source@rotation.T+target_translation
                distances=[float(index.nearest(target)['distance_mm']) for target in targets]
                worst=float(max(distances))
                if not np.isfinite(worst) or worst>0.55:continue
                trial={**seed,'translation_mm':target_translation.tolist(),
                    'normal_origin_search':{'method':'complete_group_normal_origin_search_in_CAD_envelope',
                        'cad_frame_offset_range_mm':[lo,hi],'frame_normal_offset_mm':float(level),
                        'sample_count':count,'point_count':len(rows),'max_nearest_surface_distance_mm':worst,
                        'source_coordinates_modified':False,'source_to_cad_registered':False,
                        'metrology_accepted':False}}
                trial.pop('section_refined',None)
                ranked.append((worst,float(np.median(distances)),trial))
        candidates=[]
        for _distance,_median,trial in sorted(ranked,key=lambda row:row[:2])[:32]:
            projection,error=_project_model_selected_xyz_group(mesh,rows,[trial],{},[])
            if error:continue
            audits=projection[2]
            # Rank by the weakest existing gate margin, rather than replacing
            # the gates or treating only an average point residual as success.
            margin=max([value for audit in audits for value in [
                audit['max_surface_residual_mm']/0.55,
                audit['max_pairwise_shape_error_mm']/0.70,
                *[section['residual_spread_mm']/0.08 for section in audit['source_sections']]]],default=0)
            candidates.append((margin,trial))
        choices=[];centers=[]
        for _score,trial in sorted(candidates,key=lambda row:row[0]):
            center=source.mean(0)@np.asarray(trial['rotation']).T+trial['translation_mm']
            if any(np.linalg.norm(center-old)<0.20 for old in centers):continue
            choices.append(trial);centers.append(center)
            if len(choices)>=3:break
        if not choices:
            projection,error=_project_model_selected_xyz_group(mesh,rows,[original],{},[])
            if error:
                from .scope_review import low_cardinality_translation_candidates
                for trial in low_cardinality_translation_candidates(rows,original,index):
                    projection,error=_project_model_selected_xyz_group(mesh,rows,[trial],{},[])
                    if not error:choices.append(trial)
                if not choices:return []
                recovered=True
            else:choices=[original]
        else:
            recovered=True
        options.append(choices)
    if not recovered:return []
    return [list(frames) for frames in itertools.islice(itertools.product(*options),6)]


def _refine_model_selected_xyz_frames(mesh, source_points, selected_frames, *,
                                      max_residual_mm=0.15, max_translation_mm=0.8,
                                      max_rotation_deg=0.5):
    """Refine a model-selected frame locally while preserving complete XYZ sections.

    The local model remains the authority for the coordinate-frame hypothesis
    and must review the refined overlay before placement. This bounded CAD fit
    only corrects small per-FAI display offsets; it never declares registration
    or metrology acceptance.
    """
    if not source_points or any(row.get('requires_variant_choice') for row in source_points):
        return selected_frames,[]
    try:
        triangles,face_ids,index=cached_surface_geometry(mesh)
    except (ValueError,TypeError):
        return selected_frames,[]
    by_scope={}
    for row in source_points:
        if isinstance(row.get('xyz'),list) and len(row['xyz'])==3:
            by_scope.setdefault((row.get('program'),row.get('coordinate_system')),[]).append(row)
    updated=[];audits=[];rotation_bound=np.deg2rad(max_rotation_deg)
    for original in selected_frames:
        frame=dict(original);scope=(frame.get('program'),frame.get('coordinate_system'))
        rows=by_scope.get(scope,[])
        try:
            source=np.asarray([row['xyz'] for row in rows],float)
            rotation=np.asarray(frame.get('rotation'),float)
            translation=np.asarray(frame.get('translation_mm'),float)
        except (TypeError,ValueError):
            updated.append(frame);continue
        if (len(rows)<3 or source.shape!=(len(rows),3) or rotation.shape!=(3,3)
                or translation.shape!=(3,) or not np.isfinite(source).all()
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()):
            updated.append(frame);continue
        baseline=source@rotation.T+translation
        baseline_hits=[index.nearest(point) for point in baseline]
        baseline_distances=np.asarray([row['distance_mm'] for row in baseline_hits],float)
        if np.max(baseline_distances)<=0.10:
            updated.append(frame);continue
        normal=rotation@np.array([0.0,0.0,1.0])
        shifts=np.arange(-max_translation_mm,max_translation_mm+1e-9,0.10)
        shift_scores=[]
        for shift in shifts:
            hits=[index.nearest(point+normal*shift) for point in baseline]
            distances=np.asarray([row['distance_mm'] for row in hits],float)
            shift_scores.append((float(distances.max()),float(np.median(distances)),abs(float(shift)),float(shift)))
        initial_shift=min(shift_scores)[3]
        def transformed(params):
            delta=Rotation.from_rotvec(params[:3])
            return delta.as_matrix()@rotation,delta.apply(translation)+params[3:]
        def residual(params):
            updated_rotation,updated_translation=transformed(params)
            targets=source@updated_rotation.T+updated_translation
            errors=[]
            for target in targets:
                hit=index.nearest(target)
                errors.extend((target-np.asarray(hit['point'],float)).tolist())
            # Keep the refinement close to the model-selected transform.
            # Keep a weak prior to the model frame without leaving the fitted
            # source plane numerically offset from an exact CAD section.
            errors.extend((params[:3]/rotation_bound*1e-5).tolist())
            errors.extend((params[3:]/max_translation_mm*1e-6).tolist())
            return np.asarray(errors,float)
        lower=np.r_[np.full(3,-rotation_bound),np.full(3,-max_translation_mm)]
        upper=np.r_[np.full(3,rotation_bound),np.full(3,max_translation_mm)]
        initial=np.r_[np.zeros(3),normal*initial_shift]
        try:
            fit=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',
                              f_scale=0.08,max_nfev=24)
        except (ValueError,FloatingPointError):
            updated.append(frame);continue
        new_rotation,new_translation=transformed(fit.x)
        targets=source@new_rotation.T+new_translation
        nearest=[index.nearest(point) for point in targets]
        nearest_distances=np.asarray([row['distance_mm'] for row in nearest],float)
        if (float(nearest_distances.max())>max_residual_mm
                or float(np.median(nearest_distances))>float(np.median(baseline_distances))):
            updated.append(frame);continue
        section_points=[];section_distances=[];failed=False
        for row,target in zip(rows,targets):
            section_normal=new_rotation@np.array([0.0,0.0,1.0])
            offset=float(round(float(row['xyz'][2]),3)+section_normal@new_translation)
            nearby=index.nearby_triangles(target,max_residual_mm+0.05)
            hit=closest_plane_section(target,triangles[nearby],section_normal,offset) if len(nearby) else None
            if hit is None or hit['distance_mm']>max_residual_mm:
                failed=True;break
            section_points.append(hit['point']);section_distances.append(hit['distance_mm'])
        if failed:
            updated.append(frame);continue
        section_points=np.asarray(section_points,float)
        shape_error=np.linalg.norm((section_points[:,None,:]-section_points[None,:,:])-
                                   (targets[:,None,:]-targets[None,:,:]),axis=2)
        if float(shape_error.max(initial=0.0))>0.70:
            updated.append(frame);continue
        delta_rotation=Rotation.from_matrix(new_rotation@rotation.T).as_rotvec()
        delta_translation=new_translation-translation
        frame.update(rotation=new_rotation.tolist(),translation_mm=new_translation.tolist(),
                     section_refined=True,
                     display_refinement=dict(method='bounded_local_xyz_to_CAD_sections',
                        source_point_count=len(rows),delta_rotation_deg=float(np.rad2deg(np.linalg.norm(delta_rotation))),
                        delta_translation_mm=float(np.linalg.norm(delta_translation)),
                        max_surface_residual_before_mm=float(baseline_distances.max()),
                        max_surface_residual_after_mm=float(nearest_distances.max()),
                        max_section_residual_mm=float(max(section_distances)),
                        max_pairwise_shape_change_mm=float(shape_error.max(initial=0.0)),
                        source_to_cad_registered=False,metrology_accepted=False))
        updated.append(frame)
        if frame.get('section_refined'):
            audits.append(dict(program=scope[0],coordinate_system=scope[1],
                               **frame['display_refinement']))
    return updated,audits


def _mark_model_selected_xyz_group(store, project, item, detail, model, mesh, candidate):
    """Ask the local model to visually accept a complete, coordinate-derived FAI group."""
    evidence=candidate.get('evidence') or {}
    source_points=candidate.get('known_source_points') or evidence.get('known_source_points') or []
    declared_manifest=_declared_marker_manifest(detail.get('methods',[]))
    if declared_manifest:
        declared_names={_canonical_source_point_alias(name) for group in declared_manifest
                        for name in group['source_point_names']}
        retrieved_names={_canonical_source_point_alias(point.get('name')) for point in source_points}
        if declared_names!=retrieved_names:
            # Exact cardinality alone cannot authorize a same-family source
            # substitution. The visual path must first ask the local model
            # to confirm the actual directly measured names, including XYZ-null
            # locations, rather than accepting a complete-looking wrong list.
            return None
    required_count=(candidate.get('required_marker_count') or evidence.get('required_marker_count')
                    or candidate.get('required_point_count') or evidence.get('required_point_count') or 0)
    if required_count and len({point.get('name') for point in source_points})!=required_count:
        # A compact group decision must never hide missing measurements or
        # include unrelated semantic-retrieval points. Use the full visual
        # completion path until every declared source location is recovered.
        return None
    if any(not point.get('requires_variant_choice') and
           (not isinstance(point.get('xyz'),list) or len(point['xyz'])!=3)
           for point in source_points):
        return None
    original_source_points=list(source_points)
    coordinate_variant_audit={'status':'not_required','attempts':[]}
    if any(point.get('requires_variant_choice') or point.get('coordinate_conflict')
           or point.get('scope_ambiguous') for point in source_points):
        method_text='\n'.join(f"[MSOP p.{row.get('page','?')}] {row.get('text','')}"
                              for row in detail.get('methods',[]))
        choices,coordinate_variant_audit=_select_coordinate_variants_with_local_model(
            model,item['number'],source_points,
            _coordinate_disambiguation_context(store,project,item,source_points,method_text,
                mesh=mesh,registration_diagnostic=evidence.get('registration_diagnostic')),
            source_images=_coordinate_variant_source_images(store,project,detail,model,item['number'],source_points))
        if choices is None:
            return dict(fai=item['number'],result='abstain_or_invalid',
                        reason='local_coordinate_variant_selection_failed',
                        source_point_names=[point.get('name') for point in source_points
                                            if point.get('requires_variant_choice')],
                        local_model_audit=coordinate_variant_audit)
        choice_map={row['name']:row for row in choices}
        resolved=[]
        for point in source_points:
            row=dict(point)
            selected=choice_map.get(point.get('name'))
            if selected:
                row.update(xyz=selected['xyz'],program=selected['program'],
                           coordinate_system=selected['coordinate_system'],
                           pages=[selected['page']])
            row['requires_variant_choice']=False
            row['coordinate_variants']=[]
            row['coordinate_conflict']=False
            row['scope_ambiguous']=False
            row.pop('unresolved_scopes',None)
            resolved.append(row)
        source_points=resolved
    scope_review_audit=None
    if evidence.get('source_scope_review_requested'):
        from .scope_review import review_source_scopes
        source_points,scope_review_audit=review_source_scopes(store,project,item,detail,model,source_points)
        if source_points is None:
            return dict(fai=item['number'],result='abstain_or_invalid',
                reason='local_measurement_scope_review_unconfirmed',source_scope_review=scope_review_audit)
    diagnostic=evidence.get('registration_diagnostic') or {}
    selected_frames=diagnostic.get('selected_frame_transforms') or []
    frame_hypotheses=diagnostic.get('frame_transform_hypotheses') or []
    # The rigid frame is selected at registration scope, so an individual FAI
    # may contain one or two points (e.g. a vision center or constructed line).
    # Requiring three points here silently excluded valid low-cardinality FAIs.
    if not source_points:
        return None
    coordinate_scopes={(point.get('program'),point.get('coordinate_system')) for point in source_points}
    if scope_review_audit:
        from .scope_review import new_scope_hypotheses
        frame_hypotheses=[*frame_hypotheses,*new_scope_hypotheses(mesh,source_points,coordinate_scopes)]
    # Search only the frames used by this source group. Unrelated programs
    # otherwise inflate the origin-combination budget and can suppress every
    # relevant candidate before the local model gets a chance to compare them.
    selected_frames=[frame for frame in selected_frames
        if (frame.get('program'),frame.get('coordinate_system')) in coordinate_scopes]
    frame_hypotheses=[frame for frame in frame_hypotheses
        if (frame.get('program'),frame.get('coordinate_system')) in coordinate_scopes]
    selected_scopes={(frame.get('program'),frame.get('coordinate_system')) for frame in selected_frames}
    offered_scopes=selected_scopes|{
        (frame.get('program'),frame.get('coordinate_system')) for frame in frame_hypotheses}
    if not coordinate_scopes or not coordinate_scopes<=offered_scopes:
        if scope_review_audit:
            return dict(fai=item['number'],result='abstain_or_invalid',
                reason='reviewed_measurement_scope_has_no_frame_seed',
                required_scopes=sorted(coordinate_scopes),available_scopes=sorted(offered_scopes),
                source_scope_review=scope_review_audit)
        return None
    selected_frames=list(selected_frames)
    for scope in sorted(coordinate_scopes-selected_scopes,key=str):
        # This is just a geometric seed. It cannot create an annotation until
        # the current local-model call explicitly selects its candidate ID.
        selected_frames.append(next(frame for frame in frame_hypotheses
            if (frame.get('program'),frame.get('coordinate_system'))==scope))
    display_refinements=[]
    # A registration frame can be globally correct while one FAI uses a local
    # origin (common when MSOP coordinate systems share rotations but not
    # translations). Generate only explicit, auditable origin hypotheses:
    # keep or zero each large translation component, then require the entire
    # FAI group to pass the same exact CAD geometry gates before asking Strata
    # to choose among the visually inspectable survivors.
    frame_options=[]
    axes=[]
    base_projection,base_error=_project_model_selected_xyz_group(
        mesh,source_points,selected_frames,detail,[])
    if base_error:
        for trial in _planar_frame_hypotheses(mesh,source_points,selected_frames):
            projection,error=_project_model_selected_xyz_group(mesh,source_points,trial,detail,[])
            if not error:
                frame_options.append(dict(candidate_id=f'P{len(frame_options)+1:02d}',
                    frames=trial,projection=projection,zeroed_translation_components=[],
                    candidate_source='bounded_rigid_alignment_to_CAD_plane',geometry_gate='passed'))
    if base_error and not frame_options:
        refined_frames,display_refinements=_refine_model_selected_xyz_frames(
            mesh,source_points,selected_frames)
        if display_refinements:
            selected_frames=refined_frames
            base_projection,base_error=_project_model_selected_xyz_group(
                mesh,source_points,selected_frames,detail,[])
    if base_error and not frame_options:
        axes=[(j,i) for j,frame in enumerate(selected_frames)
              for i,value in enumerate(frame.get('translation_mm',[]))
              if isinstance(value,(int,float)) and abs(value)>0.8]
        if len(axes)<=8:
            for bits in itertools.product((0,1),repeat=len(axes)):
                trial=[{**frame,'translation_mm':list(frame['translation_mm'])}
                       for frame in selected_frames]
                changed=[]
                for bit,(frame_index,axis) in zip(bits,axes):
                    if bit:
                        trial[frame_index]['translation_mm'][axis]=0.0
                        changed.append([frame_index,axis])
                if not changed:
                    continue
                projected,error=_project_model_selected_xyz_group(mesh,source_points,trial,detail,[])
                if error:
                    continue
                frame_options.append(dict(candidate_id=f"O{len(frame_options)+1:02d}",
                    frames=trial,projection=projected,zeroed_translation_components=changed,
                    geometry_gate='passed'))
                if len(frame_options)>=12:
                    break
    elif not base_error:
        frame_options=[dict(candidate_id='BASE',frames=selected_frames,
                            projection=base_projection,zeroed_translation_components=[],
                            geometry_gate='passed')]
    if not frame_options and frame_hypotheses:
        # A frame-wide model decision can fit some feature families while
        # failing another FAI in that alignment. Offer the original rigid
        # hypotheses again at FAI scope; never let geometry pick the answer.
        # The complete source group must pass unchanged gates for each option.
        scopes=sorted(coordinate_scopes,key=str)
        by_scope=[]
        for scope in scopes:
            options=[frame for frame in [*selected_frames,*frame_hypotheses]
                     if (frame.get('program'),frame.get('coordinate_system'))==scope]
            by_scope.append(options)
        for trial in itertools.islice(itertools.product(*by_scope),64):
            if len(frame_options)>=8:
                break
            projected,error=_project_model_selected_xyz_group(mesh,source_points,list(trial),detail,[])
            if error:
                continue
            signature=[(frame.get('rotation'),frame.get('translation_mm')) for frame in trial]
            if any(signature==[(frame.get('rotation'),frame.get('translation_mm'))
                               for frame in option['frames']] for option in frame_options):
                continue
            frame_options.append(dict(candidate_id=f'H{len(frame_options)+1:02d}',
                frames=list(trial),projection=projected,zeroed_translation_components=[],
                candidate_source='FAI_scoped_registration_hypothesis',geometry_gate='passed'))
    peer_frame_seeds=[]
    if callable(getattr(store,'items',None)) and callable(getattr(store,'active_msop',None)):
        from .landmark_transfer import peer_landmark_frame_options
        for trial in peer_landmark_frame_options(store,project,item,source_points,selected_frames):
            peer_frame_seeds.append(trial)
            projection,error=_project_model_selected_xyz_group(mesh,source_points,trial,detail,[])
            if error or len(frame_options)>=8:continue
            signature=[(frame.get('rotation'),frame.get('translation_mm')) for frame in trial]
            if any(signature==[(frame.get('rotation'),frame.get('translation_mm')) for frame in option['frames']] for option in frame_options):continue
            frame_options.append(dict(candidate_id=f'G{len(frame_options)+1:02d}',frames=trial,
                projection=projection,zeroed_translation_components=[],geometry_gate='passed',
                candidate_source='exact_shared_landmark_graph_from_prior_local_model_annotations'))
    if not frame_options:
        # A valid shared landmark relation in one scope must not be discarded
        # just because another scope still has an unresolved origin. Compose
        # each relation with constrained origin proposals, then validate the
        # ENTIRE FAI and let the native model choose the final complete pose.
        for seed in [selected_frames,*peer_frame_seeds]:
            for trial in _normal_origin_frame_hypotheses(mesh,source_points,seed):
                projection,error=_project_model_selected_xyz_group(mesh,source_points,trial,detail,[])
                if error:continue
                frame_options.append(dict(candidate_id=f'N{len(frame_options)+1:02d}',
                    frames=trial,projection=projection,zeroed_translation_components=[],
                    candidate_source='shared_relation_and_complete_group_normal_origin_search',geometry_gate='passed'))
                if len(frame_options)>=8:break
            if len(frame_options)>=8:break
    if not frame_options:
        return dict(fai=item['number'],result='abstain_or_invalid',
                    reason='no_fai_local_frame_candidate_passed_geometry_gate',
                    base_geometry_diagnostic=base_error,
                    resolved_source_points=source_points,
                    coordinate_variant_audit=coordinate_variant_audit,
                    source_scope_review=scope_review_audit,
                    tested_frame_seeds=selected_frames,
                    tested_large_translation_components=len(axes) if base_error else 0)
    images=_method_page_images(store,project,detail,model,item['number'])
    msop_image=next((path for path in images if Path(path).name.startswith('msop-')),None)
    drawing_image=next((path for path in images if Path(path).name.startswith('drawing-')),None)
    if not msop_image:
        return dict(fai=item['number'],result='abstain_or_invalid',reason='source_method_image_unavailable')
    try:
        from .cad_view_atlas import render_registration_group_atlas
        atlas_mesh=mesh
        learning_dir=store._dir(project['id'])/'automatic-learning'
        atlas_paths=[]
        for option in frame_options:
            group_spec=[]
            for frame in option['frames']:
                scope=(frame.get('program'),frame.get('coordinate_system'))
                rows=[point for point in source_points
                      if (point.get('program'),point.get('coordinate_system'))==scope
                      and isinstance(point.get('xyz'),list) and len(point['xyz'])==3]
                if rows:
                    group_spec.append({**frame,'source_xyz':[point['xyz'] for point in rows]})
            if len(group_spec)!=len(coordinate_scopes):
                continue
            path=learning_dir/f"cad-fai-frame-option-{mesh['source_sha256'][:16]}-{item['number']}-{option['candidate_id']}.png"
            render_registration_group_atlas(atlas_mesh,group_spec,path,panel_size=360)
            atlas_paths.append((option['candidate_id'],path))
        if len(atlas_paths)!=len(frame_options):
            return dict(fai=item['number'],result='abstain_or_invalid',
                reason='frame_candidate_atlas_scope_count_mismatch',candidate_count=len(frame_options),atlas_count=len(atlas_paths))
        atlas_path=atlas_paths[0][1]
        if len(atlas_paths)>1:
            from PIL import Image,ImageDraw
            thumbs=[]
            for candidate_id,path in atlas_paths:
                image=Image.open(path).convert('RGB')
                image.thumbnail((720,560))
                tile=Image.new('RGB',(740,600),'white')
                tile.paste(image,((740-image.width)//2,30))
                ImageDraw.Draw(tile).text((12,8),candidate_id,fill='black')
                thumbs.append(tile)
            columns=2;rows_count=(len(thumbs)+columns-1)//columns
            montage=Image.new('RGB',(columns*740,rows_count*600),'#666666')
            for index,tile in enumerate(thumbs):
                montage.paste(tile,((index%columns)*740,(index//columns)*600))
            atlas_path=learning_dir/f"cad-fai-frame-options-{mesh['source_sha256'][:16]}-{item['number']}.jpg"
            montage.save(atlas_path,quality=88)
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        return dict(fai=item['number'],result='abstain_or_invalid',
            reason='frame_candidate_atlas_render_failed',error=f'{type(exc).__name__}: {exc}')
    choices_required=any(point.get('requires_variant_choice') for point in source_points)
    compact_points=[]
    for point in source_points if scope_review_audit else original_source_points:
        row=[point.get('name'),point.get('xyz'),point.get('program'),point.get('coordinate_system'),point.get('pages')]
        if point.get('requires_variant_choice'):
            row.append(point.get('coordinate_variants'))
        compact_points.append(row)
    method_rows=[]
    for page in detail.get('methods',[]):
        text=re.sub(r'\s+',' ',page.get('text','')).strip()
        # The structured source-point rows above retain every XYZ verbatim;
        # drop duplicated location tables while keeping measurement instructions.
        if len(source_points)>24 and len(re.findall(r'X,\s*Y,\s*Z\s*\(mm\)',text,re.I))>4:
            continue
        method_rows.append({'page':page.get('page'),'text':text[:9000]})
    fields={'headings':item.get('headings',[]),'features':item.get('features',[]),
            'specifications':item.get('specifications',[]),'warnings':item.get('warnings',[])}
    scope_review_context=(f"Native measurement-scope review (supersedes prior parser scope only; XYZ unchanged): {json.dumps(scope_review_audit,ensure_ascii=False)}\n" if scope_review_audit else '')
    context=(scope_review_context+f"Drawing {project.get('drawing_no')} Rev {project.get('revision')}; FAI {item['number']}\n"
             f"MSOP fields: {json.dumps(fields,ensure_ascii=False,separators=(',',':'))}\n"
             f"Required MSOP point count: {candidate.get('required_point_count') or evidence.get('required_point_count') or 'not explicit'}\n"
             f"Complete source XYZ rows [name,xyz,program,coordinate_system,pages,optional_conflicting_variants]: "
             f"{json.dumps(compact_points,ensure_ascii=False,separators=(',',':'))}\n"
             f"Local-model coordinate variant resolution: "
             f"{json.dumps(coordinate_variant_audit,ensure_ascii=False,separators=(',',':'))}\n"
             f"Model-selected frame transforms and fit diagnostics: "
             f"{json.dumps(selected_frames,ensure_ascii=False,separators=(',',':'))}\n"
             f"Geometry-valid local frame choices [candidate_id, zeroed_translation_components, transforms]: "
             f"{json.dumps([{ 'candidate_id':o['candidate_id'], 'complete_source_point_count':len(o['projection'][0]), 'scope_point_counts':[{ 'program':a['program'], 'coordinate_system':a['coordinate_system'], 'point_count':a['point_count']} for a in o['projection'][2]], 'zeroed_translation_components':o['zeroed_translation_components'], 'frames':o['frames']} for o in frame_options],ensure_ascii=False,separators=(',',':'))}\n"
             f"Bounded display-only FAI refinement diagnostics: "
             f"{json.dumps(display_refinements,ensure_ascii=False,separators=(',',':'))}\n"
             f"MSOP method pages: {json.dumps(method_rows,ensure_ascii=False,separators=(',',':'))}\n"
             "Attached images: original MSOP method page, additional MSOP coordinate/Plane View page when available, matching 2D drawing page when available, then six-view CAD atlas. "
             "Colored point groups in each atlas panel are every source XYZ row transformed by that panel's labeled frame candidate; "
             "they are proposed display placements, not registered metrology results.")
    if len(context)>22000:
        # Keep all point coordinates, but shrink only redundant page prose if needed.
        context=context[:0]+(scope_review_context+f"Drawing {project.get('drawing_no')} Rev {project.get('revision')}; FAI {item['number']}\n"
             f"MSOP fields: {json.dumps(fields,ensure_ascii=False,separators=(',',':'))}\n"
             f"Required MSOP point count: {candidate.get('required_point_count') or evidence.get('required_point_count') or 'not explicit'}\n"
             f"Complete source XYZ rows [name,xyz,program,coordinate_system,pages,optional_conflicting_variants]: "
             f"{json.dumps(compact_points,ensure_ascii=False,separators=(',',':'))}\n"
             f"Local-model coordinate variant resolution: "
             f"{json.dumps(coordinate_variant_audit,ensure_ascii=False,separators=(',',':'))}\n"
             f"Model-selected frame transforms and fit diagnostics: "
             f"{json.dumps([{k:v for k,v in frame.items() if k!='model_evidence'} for frame in selected_frames],ensure_ascii=False,separators=(',',':'))}\n"
             f"Geometry-valid local frame choices [candidate_id, zeroed_translation_components, transforms]: "
             f"{json.dumps([{ 'candidate_id':o['candidate_id'], 'complete_source_point_count':len(o['projection'][0]), 'scope_point_counts':[{ 'program':a['program'], 'coordinate_system':a['coordinate_system'], 'point_count':a['point_count']} for a in o['projection'][2]], 'zeroed_translation_components':o['zeroed_translation_components'], 'frames':o['frames']} for o in frame_options],ensure_ascii=False,separators=(',',':'))}\n"
             f"Bounded display-only FAI refinement diagnostics: "
             f"{json.dumps(display_refinements,ensure_ascii=False,separators=(',',':'))}\n"
             f"MSOP measurement instructions: {json.dumps(method_rows[:1],ensure_ascii=False,separators=(',',':'))}\n"
             "Attached images: original MSOP method page, additional MSOP coordinate/Plane View page when available, matching 2D drawing page when available, then six-view CAD atlas. "
             "Colored point groups in the atlas are every source XYZ row transformed by the model-selected frame hypothesis; "
             "they are proposed display placements, not registered metrology results.")
    question=(
        VIEWER_LOCATION_TASK+
        '你是本地视觉与几何判定模型。判断图中的完整坐标点组是否对应此 FAI 的 MSOP 测量特征。'
        '图中点由逐坐标系刚体假设生成：包括本地模型此前选定的假设，或等待本轮审核的 FAI 专属备选假设；'
        '本轮必须同时核对原始 MSOP 测量说明、2D视图、点名/XYZ组与 CAD 六视图叠加。'
        '若有 bounded display-only FAI refinement，表示程序在本地模型选定的同一刚体假设附近作了小范围共同位姿优化，所有同源Z点仍投影到对应源截面；这是待你审核的显示候选，不是新的模型决策。'
        '若有 planar_alignment，表示把整个源坐标系在有界范围内刚性对齐到附近 CAD 平面，所有源点间向量保持不变；'
        '它没有逐点改写原文 XYZ，仍须你结合原页和 CAD 叠加审核后选择该候选。'
        '若有peer_landmark_graph，候选来自同一CAD/MSOP中此前本地模型标记的完全同名、同坐标系且原文XYZ相同的共享特征。'
        '此前标记仍不是正确真值，不能直接复制；本轮须重新核对原图并选择完整当前FAI。共线支撑只适用于当前共线点组，不证明完整坐标系。'
        '若有 normal_origin_search，表示只沿原坐标系法向，在实际 CAD 包络内搜索整组共同原点；'
        '平面内位姿和全部源点间向量未改变，但尚未证明原始基准高度。请结合 MSOP 基准、正反面和定义页审核，'
        '不能因为数值距离小就接受错误的试验面或层级；若只能确认显示位置而不能认证测量基准，必须在证据中说明。'
        '先在附图中比较所有带候选编号的 CAD 点组面板；每个编号都已通过完整点组的几何门槛。'
        '必须选择唯一且与 MSOP 原文、2D视图一致的 candidate_id；不能因为几何通过就默认接受。若不唯一或证据不支持则 ABSTAIN。'
        '这些是用于 Viewer 的显示位置，不是计量验收；不应仅因全局框架残差大于0.1mm就弃答，但必须查看点组是否整体偏离目标表面。'
        '接受时程序会将完整 MSOP XYZ 点组投影到有效 CAD 面，保持每组点间向量；不需要重述每个点。'
        '只输出单行 JSON，字段：decision（AUTO_MARK 或 ABSTAIN）、fai、source_point_count（必须等于上下文点数）、candidate_id（AUTO_MARK时必须精确等于一个提供的候选编号）、'
        'evidence、confidence（low/medium/high）、source_pages；若存在坐标冲突还需 coordinate_choices，逐项原样选择 name/page/program/coordinate_system/xyz。'
        f"FAI 必须为 {item['number']}，完整点数必须为 {len(source_points)}。")
    if scope_review_audit:
        question+=(' 本轮的源 XYZ 版本选择和测量坐标系归属已由前置本地模型逐组确认，完整结果在上下文。'
            '这些是已解析的当前输入，不是待你重新展开的冲突行；不要再次输出 coordinate_choices。'
            '核对每个候选所有坐标组的总点数和叠加图，仅选择完整候选或弃答。evidence 控制在80词以内。')
    # Some MSOP inspections contain dozens of explicit XYZ samples. Preserve
    # the complete local-model evidence instead of rejecting those FAI groups
    # at the former 24k-character ceiling; the outer request still hard-fails
    # above its independent 48k-character context budget.
    if len(context)+len(question)>58000:
        return dict(fai=item['number'],result='abstain_or_invalid',reason='visual_evidence_exceeds_context_budget',
                    context_char_count=len(context),question_char_count=len(question),context_budget_chars=58000)
    model_images=list(dict.fromkeys([*images,str(atlas_path)]))
    messages=[{'role':'user','content':question,'attachments':[{'filename':'MSOP, drawing, model-selected CAD XYZ overlay',
        'parsed_text':context,'images':model_images}]}]
    try:
        raw,truncated,tokens=model._generate_once(messages,max_context_chars=len(context)+len(question)+512,
                                                   max_new_tokens=1536,max_images=len(model_images))
    except Exception as exc:
        return dict(fai=item['number'],result='abstain_or_invalid',reason='local_vision_inference_failed',
                    error=f'{type(exc).__name__}: {exc}'[:400])
    parsed=_object(raw) or {}
    decision=re.sub(r'[^A-Z_]','',str(parsed.get('decision') or '').upper())
    reported_choices=parsed.get('coordinate_choices')
    if scope_review_audit:
        coordinate_choices_valid=scope_review_audit.get('status')=='selected'
        coordinate_choices=[]
    elif coordinate_variant_audit.get('status')=='selected':
        confirmed_choices=coordinate_variant_audit['selected_choices']
        coordinate_choices_valid,coordinate_choices=_validate_coordinate_variant_choices(
            original_source_points,reported_choices if reported_choices not in (None,[]) else confirmed_choices)
        coordinate_choices_valid=coordinate_choices_valid and (
            {row['name']:row for row in coordinate_choices}==
            {row['name']:row for row in confirmed_choices})
    else:
        coordinate_choices_valid,coordinate_choices=_validate_coordinate_variant_choices(
            source_points,reported_choices)
    valid=(not truncated and decision in ('AUTO_MARK','AUTOMARK')
           and _fai_number(parsed.get('fai'))==item['number']
           and parsed.get('source_point_count')==len(source_points)
           and _confidence(parsed.get('confidence')) is not None
           and bool(parsed.get('evidence')) and coordinate_choices_valid
           and isinstance(parsed.get('source_pages'),list)
           and any(type(page) is int and page in {row.get('page') for row in detail.get('methods',[])}
                   for page in parsed.get('source_pages',[])))
    selected_option=next((option for option in frame_options
                          if option['candidate_id']==parsed.get('candidate_id')),None)
    valid=valid and selected_option is not None
    if not valid:
        return dict(fai=item['number'],result='abstain_or_invalid',
                    reason='model_xyz_group_review_abstained_or_invalid',decision=decision,
                    truncated=truncated,generated_tokens=tokens,raw_output=raw[:1400],
                    model_candidate_atlas_path=str(atlas_path))
    selected_frames=selected_option['frames']
    projected,projection_error=_project_model_selected_xyz_group(
        mesh,source_points,selected_frames,detail,[] if scope_review_audit else coordinate_choices)
    if projection_error:
        return dict(fai=item['number'],result='abstain_or_invalid',
                    reason='model_xyz_group_geometry_gate_failed',geometry_diagnostic=projection_error,
                    decision=decision,truncated=truncated,generated_tokens=tokens,raw_output=raw[:1000],
                    model_candidate_atlas_path=str(atlas_path))
    anchors,point_audit,scope_audits=projected
    confidence=_confidence(parsed.get('confidence'))
    source_page_list=parsed.get('source_pages') or item.get('pages',[])
    variant_selection_note=(f"本地模型坐标行选择：{json.dumps(coordinate_variant_audit.get('selected_choices',[]),ensure_ascii=False,separators=(',',':'))}。"
                           if coordinate_variant_audit.get('selected_choices') else '')
    note=(f"本地模型自动标记完整 MSOP XYZ 点组，共 {len(anchors)} 点；模型决定 {decision}，选择局部坐标候选 {selected_option['candidate_id']}，置信度 {confidence}。"
          f"模型证据：{parsed.get('evidence')}；来源页：{source_page_list}。{variant_selection_note}"
          f"坐标系审计：{json.dumps(scope_audits,ensure_ascii=False,separators=(',',':'))}。"
          f"模型选定的完整刚体候选：{json.dumps(selected_frames,ensure_ascii=False,separators=(',',':'))}。"
          f"本地模型测量坐标系复核：{json.dumps(scope_review_audit,ensure_ascii=False,separators=(',',':'))}。"
          f"逐点原文 XYZ 与 CAD 投影：{json.dumps(point_audit,ensure_ascii=False,separators=(',',':'))}。"
          '位置用于 Viewer 显示；每点来自 MSOP 原文坐标和本地模型选定的坐标系变换，并按完整组投影到有效 CAD 面。'
          '本标记不是 source_to_cad 注册认证或计量验收，显示置信度封顶为 low。')
    review_locations=[dict(feature=row.get('name') or row.get('source_name') or 'source point',
        source_point_names=[row.get('name') or row.get('source_name')] if row.get('name') or row.get('source_name') else [])
        for row in point_audit]
    entity_review=_review_visual_entity_group(store,project,item,detail,model,mesh,anchors,
        review_locations,_method_page_images(store,project,detail,model,item['number']))
    if entity_review.get('status')!='confirmed':
        return dict(fai=item['number'],result='abstain_or_invalid',reason='local_entity_group_review_not_confirmed',entity_review=entity_review)
    store.annotate(project['id'],item['number'],anchors,note,'auto_marked',
                   mesh['source_sha256'],item['document_id'],'local_model_candidate','low',
                   feature_verification=entity_review)
    return dict(fai=item['number'],result='accepted',confidence='low',decision=decision,
                anchor_count=len(anchors),face_ids=list(dict.fromkeys(row['face_id'] for row in anchors)),
                source_pages=source_page_list,evidence=parsed.get('evidence'),
                generated_tokens=tokens,visual_fallback=True,xyz_group_review=True,
                selected_local_frame_candidate=selected_option['candidate_id'],
                peer_landmark_graph_used=any(frame.get('peer_landmark_graph') for frame in selected_frames),
                coordinate_variant_audit=coordinate_variant_audit,
                source_scope_review=scope_review_audit,
                coordinate_frame_audit=scope_audits)


def _is_local_inference_model(model):
    """Return whether inference runs from a local model/runtime only.

    Viewer marking is prohibited from using a hosted inference endpoint. A
    Transformers/PEFT bundle is local by construction; an OpenAI-compatible
    endpoint is accepted only when it resolves to loopback. This includes the
    local Strata service used by the Viewer.
    """
    endpoint = str(getattr(model, 'openai_base_url', '') or '').strip()
    if not endpoint:
        info = model.info() if hasattr(model, 'info') else {}
        return info.get('inference_backend') in ('transformers-peft', 'strata-native-local')
    try:
        hostname = (urlparse(endpoint).hostname or '').lower().rstrip('.')
    except ValueError:
        return False
    if hostname == 'localhost' or hostname.endswith('.localhost'):
        return True
    if hostname == '::1':
        return True
    if hostname.startswith('127.'):
        return True
    return False


def _local_model_runtime_ready(model):
    """Check the active local backend without requiring a PEFT bundle for APIs."""
    if not model or not getattr(model, 'loaded', False):
        return False
    if getattr(model, 'openai_base_url', ''):
        return True
    return getattr(model, 'bundle', None) is not None


@functools.lru_cache(maxsize=64)
def _verified_pdf_digest(path, mtime_ns, size):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _method_page_images(store, project, detail, model, fai_number=None):
    """Render source MSOP and matching drawing pages for local vision.

    OCR can flatten Plane View/Detail View labels and lose their spatial
    relationship. Supplying the original MSOP page and exact-FAI 2D drawing
    page lets a multimodal local model cross-check the source feature while it
    selects among already generated CAD anchors. Text-only Strata stays on the
    text path; PDFs must still match the ingested source hash.
    """
    info = model.info() if hasattr(model, 'info') else {}
    capability = info.get('image_inference')
    backend = info.get('inference_backend')
    if capability not in ('strata_native_vision', 'local_auxiliary_vision', 'configured_multimodal_backend'):
        # A direct local Transformers model is multimodal unless explicitly
        # reported otherwise; native Strata is text-only unless its local
        # auxiliary vision endpoint is configured.
        if backend != 'transformers-peft':
            return []
    document = store.active_msop(project) or {}
    source = Path(document.get('path') or '')
    if not source.is_file():
        return []
    source_digest = document.get('sha256')
    if not source_digest:
        return []
    method_pages = list(dict.fromkeys(
        int(page['page']) for page in detail.get('methods', [])
        if str(page.get('page', '')).isdigit() and int(page['page']) > 0
    ))
    # MSOP method pages often refer to XYZ rows and Plane Views on earlier
    # coordinate-definition pages. Include the most relevant such page so the
    # local vision model can see the point labels in their actual spatial view.
    # Choose it from exact point names resolved from the method's Measure N
    # instruction; never select a page merely because it is nearby in the PDF.
    coordinate_page = None
    all_pages = (store.active_msop(project) or {}).get('method_pages', [])
    if method_pages and all_pages:
        try:
            named_points = extract_named_points(all_pages)
            point_names = _explicit_measure_point_names(detail.get('methods', []), named_points) or set()
            if not point_names:
                # An incomplete Measure N list cannot resolve as a complete
                # explicit group, but known names from that same instruction
                # still identify useful XYZ definition pages for the model.
                referenced=referenced_points(None,detail.get('methods',[]),named_points,
                    extract_feature_labels(detail.get('methods',[]),[]))
                point_names=set(referenced)
            scores = []
            method_set = set(method_pages)
            dependency_set = {
                int(page['page']) for page in detail.get('dependencies', [])
                if str(page.get('page', '')).isdigit() and int(page['page']) > 0
            }
            for page in all_pages:
                number = page.get('page')
                if not isinstance(number, int) or number in method_set or number not in dependency_set:
                    continue
                body = page.get('text', '')
                matched = sum(bool(re.search(rf'(?<![A-Z0-9_]){re.escape(name)}(?![A-Z0-9_])', body, re.I))
                              for name in point_names)
                if matched:
                    scores.append((matched, number))
            if scores:
                coordinate_page = max(scores)[1]
        except (KeyError, TypeError, ValueError):
            # An ambiguous or malformed point list is not enough evidence to
            # choose a cross-reference page. The ordinary source pages remain.
            coordinate_page = None
    pages = method_pages[:1]
    if coordinate_page is not None and coordinate_page not in pages:
        pages.append(coordinate_page)
    for page in method_pages[1:]:
        if len(pages) >= 2:
            break
        if page not in pages:
            pages.append(page)
    if not pages:
        return []
    output = store._dir(project['id']) / 'automatic-learning' / 'method-page-images'
    output.mkdir(parents=True, exist_ok=True)
    import pymupdf
    images = []
    def render(document, page_numbers, namespace):
        path = Path(document.get('path') or '')
        if not path.is_file() or not document.get('sha256'):
            return
        stat = path.stat()
        if _verified_pdf_digest(str(path), stat.st_mtime_ns, stat.st_size) != document['sha256']:
            return
        with pymupdf.open(path) as pdf:
            for number in page_numbers:
                if not 1 <= number <= len(pdf):
                    continue
                target = output / f"{namespace}-{document['sha256'][:20]}-page-{number}.png"
                if not target.is_file():
                    pixmap = pdf[number - 1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                    temporary = target.with_suffix('.png.tmp')
                    temporary.write_bytes(pixmap.tobytes('png'))
                    temporary.replace(target)
                images.append(str(target))

    render(document, pages, 'msop')
    if fai_number is not None and len(images) < 3:
        drawing = next((value for value in project.get('documents', [])
                        if value.get('role') == 'drawing'), None)
        if drawing:
            drawing_path = Path(drawing.get('path') or '')
            if drawing_path.is_file() and drawing.get('sha256'):
                stat = drawing_path.stat()
                if _verified_pdf_digest(str(drawing_path), stat.st_mtime_ns, stat.st_size) == drawing['sha256']:
                    pattern = re.compile(rf'(?<![A-Z0-9])FAI\s*0*{int(fai_number)}(?!\d)', re.I)
                    with pymupdf.open(drawing_path) as pdf:
                        scores = [(len(pattern.findall(page.get_text())), index + 1)
                                  for index, page in enumerate(pdf)]
                    matches = [page for count, page in sorted(scores, reverse=True) if count]
                    render(drawing, matches[:1], 'drawing')
                    # A full engineering sheet can be several thousand points
                    # wide. The local vision encoder otherwise loses the tiny
                    # balloon and leader even when the PDF itself is vector.
                    # Keep the full sheet and add an exact-label detail; the
                    # crop is source retrieval, never a CAD correspondence.
                    if matches:
                        images.extend(_drawing_fai_detail_images(
                            drawing, matches[0], fai_number, output))
    return images


def _drawing_fai_detail_images(document, page_number, fai_number, output):
    """Render bounded original-PDF detail around an exact FAI callout.

    Two-line balloons (``FAI`` above ``23``) are common. PyMuPDF searches
    across the line break and returns both rectangles; join those adjacent
    rectangles, but never match a bare number or a neighboring FAI ID.
    No detected edge/leader is interpreted as a part location here.
    """
    import pymupdf
    path=Path(document.get('path') or '')
    if not path.is_file() or not document.get('sha256'):
        return []
    stat=path.stat()
    if _verified_pdf_digest(str(path),stat.st_mtime_ns,stat.st_size)!=document['sha256']:
        return []
    result=[]
    with pymupdf.open(path) as pdf:
        if not 1<=page_number<=len(pdf):return []
        page=pdf[page_number-1]
        # Small pages already retain legible callouts at full-page scale.
        if max(page.rect.width,page.rect.height)<1000:return []
        matches=page.search_for(f'FAI {int(fai_number)}')
        groups=[]
        for rect in matches:
            if groups and abs(rect.x0-groups[-1].x0)<40 and abs(rect.y0-groups[-1].y1)<40:
                groups[-1]|=rect
            else:groups.append(pymupdf.Rect(rect))
        for index,rect in enumerate(groups):
            # search_for may match FAI 5 at the start of FAI 55: check the
            # original words beside the hit, including the remaining digits.
            text=page.get_text('text',clip=rect+(-2,-2,12,2))
            if not re.search(rf'\bFAI\s*0*{int(fai_number)}(?!\d)',text,re.I):continue
            center=(rect.tl+rect.br)/2
            half_width=max(160,page.rect.width*.18)
            half_height=max(160,page.rect.height*.22)
            clip=pymupdf.Rect(center.x-half_width,center.y-half_height,
                             center.x+half_width,center.y+half_height)&page.rect
            target=Path(output)/(
                f'drawing-detail-v1-{document["sha256"][:20]}-page-{page_number}'
                f'-fai-{int(fai_number)}-{index+1}.png')
            if not target.is_file():
                pixmap=page.get_pixmap(matrix=pymupdf.Matrix(2,2),clip=clip,alpha=False)
                temporary=target.with_suffix('.png.tmp')
                temporary.write_bytes(pixmap.tobytes('png'));temporary.replace(target)
            metadata={'document_sha256':document['sha256'],'page':page_number,
                'fai':int(fai_number),'clip_pdf_points':list(clip),
                'original_pdf_text':page.get_text('text',clip=clip)}
            temporary=target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(target.with_suffix('.json'))
            result.append(str(target))
            if len(result)>=2:break
    return result


def _resolve_drawing_location_manifest(model,fai,images,document,*,_repair=None):
    """Ask the local model for the whole drawing-defined location group.

    Used only when MSOP has no explicit point/count manifest. A readable
    balloon is useful evidence but does not authorize one marker to stand in
    for a multi-spot requirement. Exact source quotes and model arithmetic
    remain in the receipt; this function never chooses a CAD point.
    """
    sources=[];detail_images=[]
    for path in images:
        if not Path(path).name.startswith('drawing-detail-v1-'):continue
        metadata_path=Path(path).with_suffix('.json')
        if not metadata_path.is_file():continue
        data=json.loads(metadata_path.read_text(encoding='utf-8'))
        if data.get('document_sha256')!=document.get('sha256') or data.get('fai')!=fai:continue
        sources.append(data);detail_images.append(path)
    if not sources:return None,{'status':'not_available'}
    text=json.dumps(sources,ensure_ascii=False)
    if len(text)>24000:return None,{'status':'source_evidence_exceeds_budget'}
    evidence_images=list(dict.fromkeys([
        *[path for path in images if Path(path).name.startswith('drawing-')
          and not Path(path).name.startswith('drawing-detail-')],*detail_images,*([path for path in images if Path(path).name.startswith('cad-part-catalog-')]
                           if document.get('local_model_part_identity_context') else [])]))
    question=(VIEWER_LOCATION_TASK+
        f'仅从原始图纸核对 FAI {fai} 的实体位置清单，不进行 CAD 定位。'
        'MSOP 未逐项列出点名/数量，因此必须读这个编号自己的引线、视图与说明；不能套用附近另一个 FAI 的焊点说明。'
        '若明确要求多个焊点/位置，要算出整个零件上的总数，说明 per side/per corner 等重复数量如何从原图确认。'
        '单个 marker 不能代替多个焊点。若重复次数或编号归属无法确定则 ABSTAIN。'
        '只输出 JSON：decision(CONFIRM_LOCATIONS 或 ABSTAIN),fai,feature,marker_count(整数),'
        'locations_per_group(每组位置数整数),group_count(重复组数整数),'
        'source_quotes(逐字引用局部原图文字的非空数组),multiplicity_evidence,source_pages。'
        'marker_count 必须等于 locations_per_group × group_count；原文数量和图中重复次数必须分别说明。'
        'source_quotes 必须来自以下原始 PDF 局部文字，保留数字和单位，不得修正或编造原文。'
        f'\nSource drawing details: {text}'+
        (f'\n本地模型已独立匹配的CAD实体：{json.dumps(document.get("local_model_part_identity_context"),ensure_ascii=False)}。'
         '请用原图重复实例与这些实体对照核对group_count，不能默认每个body就是一个焊点组。'
         if document.get('local_model_part_identity_context') else '')+
        (f'\n前次你已选择确认，但机器校验失败：{json.dumps(_repair,ensure_ascii=False)}。'
         '请只修正原文引用、字段和乘法；引用逐字复制原始 PDF 文本，不能把 O.05 修成 0.05，'
         '不必引用编号或水印等无关短词。per side/per rail 需要说明整个零件的重复数；无法确认则弃答。'
         if _repair else ''))
    messages=[{'role':'user','content':question,'attachments':[{
        'filename':'Original drawing sheet and exact-FAI high-resolution details','images':evidence_images}]}]
    try:
        raw,truncated,tokens=model._generate_once(messages,max_context_chars=len(question)+512,
            max_new_tokens=1024,max_images=len(evidence_images),temperature=0.0)
    except Exception as exc:return None,{'status':'local_model_failed','error':str(exc)[:500]}
    parsed=_object(raw) or {};quotes=parsed.get('source_quotes')
    normalize=lambda value:re.sub(r'\s+',' ',value).strip().casefold()
    source_texts=[normalize(row.get('original_pdf_text','')) for row in sources]
    valid_quotes=(isinstance(quotes,list) and bool(quotes) and
        all(isinstance(quote,str) and bool(normalize(quote)) and
            any(normalize(quote) in body for body in source_texts) for quote in quotes))
    count=parsed.get('marker_count')
    per_group=parsed.get('locations_per_group');group_count=parsed.get('group_count')
    multiplicity_valid=(type(count) is int and type(per_group) is int and type(group_count) is int
        and per_group>0 and group_count>0 and count==per_group*group_count)
    quantity_quote_valid=(type(count) is int and (count==1 or isinstance(quotes,list) and bool(quotes) and any(
        isinstance(quote,str) and re.search(
            rf'\b{re.escape(str(per_group))}\s*(?:[x×]|(?:weld\s+)?spots?\b|(?:vector\s+)?points?\b|per\b)',
            normalize(quote),re.I) for quote in quotes)))
    valid=(not truncated and parsed.get('decision')=='CONFIRM_LOCATIONS'
        and _fai_number(parsed.get('fai'))==fai and type(count) is int
        and 0<count<=MAX_ANNOTATION_POINTS and valid_quotes and quantity_quote_valid and multiplicity_valid
        and bool(parsed.get('feature')) and bool(parsed.get('multiplicity_evidence')))
    audit={'status':'confirmed' if valid else 'abstained_or_invalid','raw_output':raw,
        'truncated':bool(truncated),'generated_tokens':tokens,'source_quotes_valid':valid_quotes,
        'quantity_quote_valid':quantity_quote_valid,
        'multiplicity_valid':multiplicity_valid,
        'source_details':sources,'source_images':evidence_images,'decision':parsed}
    if not valid:
        if _repair is None and parsed.get('decision')=='CONFIRM_LOCATIONS':
            group,retried=_resolve_drawing_location_manifest(model,fai,images,document,_repair={
                'previous_answer':parsed,'source_quotes_valid':valid_quotes,
                'quantity_quote_valid':quantity_quote_valid,'multiplicity_valid':multiplicity_valid,
                'output_truncated':bool(truncated)})
            retried['previous_attempts']=[audit]
            return group,retried
        return None,audit
    return {'kind':'drawing_declared_locations','declared_count':count,'source_point_names':[],
        'locations_per_group':per_group,'group_count':group_count,
        'instruction':parsed['feature']+'; '+parsed['multiplicity_evidence'],
        'source_page':sources[0]['page'],'source_document_sha256':document['sha256'],
        'source_quotes':quotes},audit


def _drawing_spacing_diagnostic(anchors,group,*,max_neighbor_ratio=1.8):
    """Reject obviously clustered candidates for a single evenly spaced group.

    This loose search-consistency gate is not a drawing tolerance or metrology
    acceptance rule. It is independent of view and permits curved/circular
    layouts. Multi-group layouts need explicit group identities before this
    check can be applied without mixing their separate intervals.
    """
    quotes=group.get('source_quotes') or []
    if (group.get('group_count')!=1 or len(anchors)<3 or
            not any(re.search(r'\bevenly\s+spaced\b',str(quote),re.I) for quote in quotes)):
        return None
    points=np.asarray([anchor['position'] for anchor in anchors],float)
    distances=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
    np.fill_diagonal(distances,np.inf)
    nearest=distances.min(axis=1)
    ratio=float(nearest.max()/nearest.min()) if nearest.min()>1e-9 else float('inf')
    if ratio<=max_neighbor_ratio:return None
    return {'reason':'source_even_spacing_constraint_failed','source_quotes':quotes,
        'nearest_neighbor_distances_mm':nearest.tolist(),
        'largest_to_smallest_neighbor_ratio':ratio,'candidate_search_ratio_limit':max_neighbor_ratio,
        'is_source_tolerance':False,'metrology_accepted':False}


def _review_even_spacing_options(store,project,item,model,mesh,anchors,markers,group,images):
    """Request a fresh model decision on complete geometric layout proposals."""
    views={row.get('view_id') for row in markers}
    if len(views)!=1:return None,{'status':'not_available','reason':'mixed_model_selected_views'}
    view=next(iter(views))
    options=evenly_spaced_candidates(mesh,anchors,view,group['declared_count'])
    if not options:return None,{'status':'not_available','reason':'no_complete_visible_spacing_hypothesis'}
    from .cad_view_atlas import render_registration_group_atlas
    diagram_key=hashlib.sha256(json.dumps(options,sort_keys=True).encode()+
        Path(__file__).with_name('cad_view_atlas.py').read_bytes()).hexdigest()[:16]
    output=store._dir(project['id'])/'automatic-learning'/(
        f"cad-spacing-hypotheses-{mesh['source_sha256'][:16]}-{item['number']}-{diagram_key}.png")
    render_registration_group_atlas(mesh,[{'program':'CAD hypothesis','coordinate_system':row['pattern'],
        'hypothesis_id':row['candidate_id'],'source_xyz':[anchor['position'] for anchor in row['anchors']],
        'rotation':np.eye(3).tolist(),'translation_mm':[0,0,0]} for row in options],output,panel_size=360,
        title='CAD spacing hypotheses / inferred locations / local model must select / NOT MSOP XYZ')
    context={'fai':item['number'],'source_group':group,
        'prior_model_selected_positions':[row['position'] for row in anchors],
        'options':options,'original_MSOP_pages':item.get('pages',[])}
    question=(VIEWER_LOCATION_TASK+
        '此前你选择的实体点数量正确，但不满足原图 EVENLY SPACED 的分布要求。'
        '程序只根据你选过的 CAD 点及其所在面生成整组等距直线/圆周候选，所有点已通过表面距离、可见性与间距一致性检查；'
        '这些是推定位置候选，不是原文 XYZ，也不证明实际焊点或基准。'
        '请结合原图 FAI 引线、实体形态、正反面与完整数量，重新独立核对图中整组位置。'
        '不得因为数量或数学等距通过就接受；不在正确实体/区域、重复实例遗漏或分布路径不符时必须 ABSTAIN。'
        '只有整组确实对应当前原文实体位置时选择唯一 candidate_id；本轮不能修改候选点或改数量。'
        '只输出 JSON：decision(AUTO_MARK/ABSTAIN),fai,candidate_id,source_point_count,evidence,confidence。')
    messages=[{'role':'user','content':question,'attachments':[{'filename':'Original sources and complete spacing proposals',
        'parsed_text':json.dumps(context,ensure_ascii=False),'images':list(dict.fromkeys([*images,str(output)]))}]}]
    try:
        raw,truncated,tokens=model._generate_once(messages,
            max_context_chars=_complete_visual_context_budget(messages),max_new_tokens=512,
            max_images=len(messages[0]['attachments'][0]['images']),temperature=0.0)
    except Exception as exc:return None,{'status':'model_failed','error':str(exc)[:500]}
    parsed=_object(raw) or {}
    selected=next((row for row in options if row['candidate_id']==parsed.get('candidate_id')),None)
    valid=(not truncated and parsed.get('decision')=='AUTO_MARK' and _fai_number(parsed.get('fai'))==item['number']
        and parsed.get('source_point_count')==group['declared_count'] and selected is not None
        and bool(parsed.get('evidence')) and _confidence(parsed.get('confidence')) is not None)
    audit={'status':'selected' if valid else 'abstained_or_invalid','raw_output':raw,
        'generated_tokens':tokens,'truncated':bool(truncated),'options':options,
        'source_group':group,'atlas':str(output),'atlas_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'original_source_coordinates_used':False,
        'metrology_accepted':False}
    return (selected if valid else None),audit



def _review_visual_entity_group(store,project,item,detail,model,mesh,anchors,locations,images,*,
        source_manifest=None,drawing_manifest=None):
    """Fresh local-model verification of entity identity, not just CAD contact.

    This review cannot move or create anchors. Geometry-only displays can hide
    an otherwise wrong physical association, so both source pages and the
    immutable full-group overlay are supplied without the placement rationale.
    """
    from .cad_view_atlas import render_registration_group_atlas
    from .feature_integrity import printed_feature_names, feature_links, consistency_conflicts, VERSION
    rows=[{'marker_id':index,'feature':location.get('feature'),
        'source_point_names':location.get('source_point_names',[]),
        'view_id':location.get('view_id'),'position':anchor['position'],'face_id':anchor['face_id'],
        'instance_path':next((f.get('instance_path',[]) for f in mesh['faces'] if f['id']==anchor['face_id']),[])}
        for index,(anchor,location) in enumerate(zip(anchors,locations),start=1)]
    if not rows or len(rows)!=len(anchors) or len(locations)!=len(anchors):
        return {'status':'invalid','reason':'complete_entity_group_required'}
    evidence={'fai':item['number'],'CAD_sha256':mesh['source_sha256'],
        'document_id':item.get('document_id'),
        'document_sha256':next((row.get('sha256') for row in project.get('documents',[])
            if row.get('id')==item.get('document_id')),None),'markers':rows,
        'methods':detail.get('methods',[]),'allowed_source_names':printed_feature_names(detail.get('methods',[])),
        'integrity_version':VERSION,'source_manifest':source_manifest or {},
        'drawing_manifest':drawing_manifest or {}}
    content=json.dumps(evidence,sort_keys=True,ensure_ascii=False).encode()
    key=hashlib.sha256(content+Path(__file__).with_name('cad_view_atlas.py').read_bytes()).hexdigest()
    output=store._dir(project['id'])/'automatic-learning'/f'entity-review-{key[:24]}.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.is_file():
        render_registration_group_atlas(mesh,[{'program':'CAD proposal','coordinate_system':'marker IDs',
            'source_xyz':[row['position'] for row in rows],'rotation':np.eye(3).tolist(),
            'translation_mm':[0,0,0]}],output,
            title='UNVERIFIED entity proposal / marker IDs match table / through-surface overlay')
    # Do not pass the previous placement justification as independent evidence.
    peers=store.get(project['id']) if callable(getattr(store,'get',None)) else project
    shared=[{'fai_key':key,'feature_links':(a.get('feature_verification') or {}).get('feature_links',[])}
        for key,a in peers.get('annotations',{}).items() if a.get('status')=='auto_marked'
        and key.rsplit(':',1)[-1]!=str(item['number'])
        and any(link.get('source_name') in evidence['allowed_source_names']
                for link in (a.get('feature_verification') or {}).get('feature_links',[]))]
    context={**evidence,'peer_feature_claims_for_consistency_only':shared,'source_manifest':{k:v for k,v in (source_manifest or {}).items()
        if k in ('groups','selected_point_names','coordinate_evidence_missing')},
        'drawing_manifest':{'decision':(drawing_manifest or {}).get('decision')}}
    question=(VIEWER_LOCATION_TASK+
        '这是一次独立实体复核，当前整组标记尚未获准。只比较原文/原图与最后一张 CAD 整组编号叠加图；'
        '点数正确、等距或落在 CAD 面上都不能证明实体对应正确。叠加图会透过遮挡显示点，不能据此判断可见性。'
        '逐个检查 marker ID 的实际所属零件、特征、正反面、引线对应、同一特征的点组是否位于同一实际实体，'
        '并确认 PER NUT/PER SIDE/PER CORNER 等重复实例完整，无论此前清单声称多少组。'
        '只能引用随附原文和图像的具体引线或实体几何来判定，不能用焊点通常在哪里之类经验猜测代替证据。'
        '不允许移动、补点或借用邻近实体。实体不符用 REJECT；无法判定用 ABSTAIN。'
        'peer_feature_claims仅用于检查同名特征是否相互矛盾，不是正确答案，不得照抄此前点位。'
        '只有所有点和所有重复实例均有明确原文/图像支持时才 CONFIRM_ENTITY_GROUP。'
        '输出 JSON：decision(CONFIRM_ENTITY_GROUP/REJECT/ABSTAIN),fai,'
        'reviewed_marker_ids(逐个列出所有整数ID),all_entities_match(bool),'
        'all_instances_covered(bool),issues(数组，每项含marker_ids与reason),evidence(简短证据)，'
        'feature_assignments(每个marker_id恰一次，每项含marker_id,source_names,instance_path,evidence)。'
        'source_names只能原样选择allowed_source_names中的实际对应特征名，不得用当前FAI编号猜名。'
        '若没有打印特征名则source_names为空数组；有打印名但无法判断对应关系则ABSTAIN。'
        'instance_path必须逐项复制被复核CAD面的实例路径，evidence必须解释原页特征如何对应此实例与落点。'
        '使用紧凑JSON，每项evidence只写简短证据，不重复抄写XYZ；必须完整输出所有marker的feature_assignments。')
    messages=[{'role':'user','content':question,'attachments':[{'filename':'Original sources and unverified entity group',
        'parsed_text':json.dumps(context,ensure_ascii=False),
        'images':list(dict.fromkeys([*images,str(output)]))}]}]
    try:
        raw,truncated,tokens=model._generate_once(messages,
            max_context_chars=_complete_visual_context_budget(messages),max_new_tokens=min(24000,1200+len(rows)*240),
            max_images=len(messages[0]['attachments'][0]['images']),temperature=0.0)
        parsed=_object(raw) or {}
        ids=parsed.get('reviewed_marker_ids')
        valid=(not truncated and parsed.get('decision')=='CONFIRM_ENTITY_GROUP'
            and _fai_number(parsed.get('fai'))==item['number']
            and isinstance(ids,list) and all(type(value) is int for value in ids)
            and sorted(ids)==list(range(1,len(rows)+1))
            and parsed.get('all_entities_match') is True and parsed.get('all_instances_covered') is True
            and parsed.get('issues')==[] and bool(parsed.get('evidence')))
        links,identity_errors=feature_links(item,detail,anchors,locations,parsed.get('feature_assignments'),mesh)
        # Fetch live peers; a project snapshot taken before this run omits
        # features accepted earlier in the same batch.
        live_project=store.get(project['id']) if callable(getattr(store,'get',None)) else project
        conflicts=consistency_conflicts(live_project,item['number'],links) if valid and not identity_errors else []
        valid=valid and not identity_errors and not conflicts
        if conflicts and callable(getattr(store,'quarantine_feature_conflicts',None)):
            store.quarantine_feature_conflicts(project['id'],conflicts)
        audit={'status':'confirmed' if valid else 'rejected_or_unconfirmed',
            'integrity_version':VERSION,'feature_links':links,'identity_errors':identity_errors,
            'consistency_conflicts':conflicts,
            'raw_output':raw,'decision':parsed,'truncated':bool(truncated),'generated_tokens':tokens}
    except Exception as exc:
        audit={'status':'model_failed','error':str(exc)[:500]}
    audit.update(evidence_sha256=key,atlas=str(output),
        atlas_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        human_selected_or_moved_points=False,metrology_accepted=False)
    receipt_content=json.dumps({'evidence':evidence,'review':audit},ensure_ascii=False,indent=2)
    receipt_key=hashlib.sha256(receipt_content.encode()).hexdigest()[:20]
    receipt=output.with_name(output.stem+'-'+receipt_key+'.json')
    temporary=receipt.with_suffix('.json.tmp')
    temporary.write_text(receipt_content,encoding='utf-8')
    temporary.replace(receipt)
    return {**audit,'receipt':str(receipt),'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}


def _fai_number(value):
    """Normalize an exact FAI label; never coerce booleans or neighboring IDs."""
    if type(value) is int:return value
    if isinstance(value,str):
        match=re.fullmatch(r'(?:FAI\s*)?0*(\d+)',value.strip(),re.I)
        if match:return int(match.group(1))
    return None


def _canonical_source_point_alias(value):
    """Canonicalize only the common omitted ``_PNT_`` spelling variation."""
    if not isinstance(value,str):
        return None
    return re.sub(r'_PNT_','_',value.strip().upper())


def _normalize_model_source_aliases(parsed, known_source_points, allowed_source_pages):
    """Normalize unambiguous OCR-style point-name and numeric page aliases.

    This does not infer a feature association: the alias must resolve one-to-one
    to an exact MSOP source point name, and geometry selection remains the
    local model's responsibility.
    """
    if not isinstance(parsed,dict):
        return parsed,[]
    expected={row.get('name') for row in known_source_points if row.get('name')}
    aliases={}
    canonical={}
    for name in expected:
        key=_canonical_source_point_alias(name)
        canonical.setdefault(key,[]).append(name)
    for name in expected:
        key=_canonical_source_point_alias(name)
        if len(canonical.get(key,[]))==1:
            aliases[key]=name
    def page_alias(value):
        if not isinstance(value,str):return value
        match=re.fullmatch(r'(?:MSOP\s+(?:p(?:age)?\.?\s*)?)?(\d+)',value.strip(),re.I)
        if match and int(match.group(1)) in allowed_source_pages:return int(match.group(1))
        return value
    changed=[]
    result=dict(parsed)
    for field,key_field in (('markers','source_point_names'),
                            ('registration_points','source_point_name'),
                            ('coordinate_choices','name')):
        rows=result.get(field)
        if not isinstance(rows,list):
            continue
        normalized=[]
        for row in rows:
            if not isinstance(row,dict):
                normalized.append(row);continue
            row=dict(row)
            if field=='markers':
                names=row.get(key_field)
            else:
                names=[row.get(key_field)]
            if isinstance(names,list):
                new_names=[]
                for name in names:
                    canonical_name=_canonical_source_point_alias(name)
                    resolved=(name if name in expected else aliases.get(canonical_name,name))
                    if resolved!=name:
                        changed.append({'from':name,'to':resolved})
                    new_names.append(resolved)
                row[key_field]=new_names if field=='markers' else (new_names[0] if new_names else None)
            if 'page' in row:row['page']=page_alias(row['page'])
            pages=row.get('source_pages')
            if isinstance(pages,list):row['source_pages']=[page_alias(v) for v in pages]
            normalized.append(row)
        result[field]=normalized
    pages=result.get('source_pages')
    if isinstance(pages,list):result['source_pages']=[page_alias(v) for v in pages]
    return result,changed


def _partition_visual_source_points(source_points, *, required_marker_groups=None,
                                   chunk_size=10, threshold=14):
    """Partition large XYZ marker sets for local-model decisions.

    The partition changes only the inference payload, never the validation
    scope. The caller must merge every point exactly once and run the ordinary
    full-group validators before any annotation can be stored.
    """
    points=[row for row in source_points if isinstance(row,dict) and row.get('name')]
    if len(points)<=threshold or len(points)!=len(source_points):
        return []
    by_name={row['name']:row for row in points}
    source_names=set(by_name)
    groups=[];assigned=set()
    for group in required_marker_groups or []:
        names=[name for name in group.get('source_point_names',[])
               if name in source_names and name not in assigned]
        if not names:
            continue
        assigned.update(names)
        # Keep each printed MSOP measurement group intact where possible; a
        # long group is split only at the bounded output size.
        groups.extend(names[index:index+chunk_size] for index in range(0,len(names),chunk_size))
    remaining=[row['name'] for row in points if row['name'] not in assigned]
    groups.extend(remaining[index:index+chunk_size] for index in range(0,len(remaining),chunk_size))
    packed=[];current=[]
    for group_names in groups:
        if current and len(current)+len(group_names)>chunk_size:
            packed.append(current);current=[]
        current.extend(group_names)
        if len(current)>=chunk_size:
            packed.append(current);current=[]
    if current:
        packed.append(current)
    return [[by_name[name] for name in names] for names in packed]


def _merge_visual_marker_chunks(chunk_results, expected_chunks, *, fai):
    """Merge model-selected point markers only when every chunk is complete."""
    merged={'decision':'AUTO_MARK','fai':fai,'markers':[],'registration_points':[],
            'coordinate_choices':[],'source_pages':[],'confidence':'high','evidence':''}
    failures=[];choices_by_name={};refs_by_name={};coverage=[];confidence_rank={'low':0,'medium':1,'high':2}
    expected_all={name for chunk in expected_chunks for name in chunk}
    for index,(result,expected_names) in enumerate(zip(chunk_results,expected_chunks)):
        if not isinstance(result,dict):
            failures.append({'chunk':index+1,'reason':'invalid_json_object'});continue
        if result.get('decision') not in ('AUTO_MARK','AUTOMARK') or _fai_number(result.get('fai'))!=fai:
            failures.append({'chunk':index+1,'reason':'decision_or_fai_invalid'});continue
        markers=result.get('markers')
        if not isinstance(markers,list) or not markers:
            failures.append({'chunk':index+1,'reason':'markers_missing'});continue
        chunk_coverage=[]
        for marker in markers:
            if not isinstance(marker,dict):
                failures.append({'chunk':index+1,'reason':'marker_not_object'});continue
            names=marker.get('source_point_names')
            if not isinstance(names,list) or not names or any(name not in expected_names for name in names):
                failures.append({'chunk':index+1,'reason':'marker_outside_chunk'});continue
            chunk_coverage.extend(names)
            merged['markers'].append(marker)
        if set(chunk_coverage)!=set(expected_names) or len(chunk_coverage)!=len(expected_names):
            failures.append({'chunk':index+1,'reason':'chunk_point_coverage_incomplete',
                             'expected':len(expected_names),'covered':len(set(chunk_coverage)),
                             'missing':sorted(set(expected_names)-set(chunk_coverage))[:20]})
        coverage.extend(chunk_coverage)
        choices=result.get('coordinate_choices') or []
        if not isinstance(choices,list):
            failures.append({'chunk':index+1,'reason':'coordinate_choices_invalid'});continue
        for choice in choices:
            if not isinstance(choice,dict) or choice.get('name') not in expected_all:
                failures.append({'chunk':index+1,'reason':'coordinate_choice_outside_fai'});continue
            old=choices_by_name.get(choice['name'])
            if old is not None and old!=choice:
                failures.append({'chunk':index+1,'reason':'coordinate_choice_conflict','name':choice['name']})
            choices_by_name[choice['name']]=choice
        refs=result.get('registration_points') or []
        if not isinstance(refs,list):
            failures.append({'chunk':index+1,'reason':'registration_points_invalid'});continue
        for ref in refs:
            name=ref.get('source_point_name') or ref.get('name') if isinstance(ref,dict) else None
            if name not in expected_all:
                failures.append({'chunk':index+1,'reason':'registration_reference_outside_fai'});continue
            old=refs_by_name.get(name)
            if old is not None and old!=ref:
                failures.append({'chunk':index+1,'reason':'registration_reference_conflict','name':name})
            refs_by_name[name]=ref
        confidence=_confidence(result.get('confidence'))
        if confidence is None:
            failures.append({'chunk':index+1,'reason':'confidence_missing_or_invalid'})
        elif confidence_rank[confidence]<confidence_rank[merged['confidence']]:
            merged['confidence']=confidence
        if not merged['evidence']:
            merged['evidence']=str(result.get('evidence') or '')[:300]
        for page in result.get('source_pages') or []:
            if page not in merged['source_pages']:
                merged['source_pages'].append(page)
    if len(chunk_results)!=len(expected_chunks):
        failures.append({'reason':'chunk_result_count_mismatch','expected':len(expected_chunks),
                         'actual':len(chunk_results)})
    merged['coordinate_choices']=list(choices_by_name.values())
    merged['registration_points']=list(refs_by_name.values())
    if set(coverage)!=expected_all:
        failures.append({'reason':'merged_point_coverage_incomplete',
                         'missing':sorted(expected_all-set(coverage))[:30]})
    if failures:
        merged['decision']='ABSTAIN'
    merged['_chunk_failures']=failures
    return merged



def _visual_chunk_contract_issues(parsed,expected_names,*,fai,surface_candidates=()):
    """Validate subset emission without choosing, deleting or moving a point."""
    if not isinstance(parsed,dict):return [{'reason':'invalid_json_object'}]
    if parsed.get('decision') not in ('AUTO_MARK','AUTOMARK') or _fai_number(parsed.get('fai'))!=fai:
        return [{'reason':'decision_or_fai_invalid'}]
    markers=parsed.get('markers')
    if not isinstance(markers,list) or not markers:return [{'reason':'markers_missing'}]
    names=[];issues=[]
    for marker in markers:
        row=marker.get('source_point_names') if isinstance(marker,dict) else None
        if not isinstance(row,list) or not row or any(not isinstance(name,str) for name in row):
            issues.append({'reason':'marker_names_missing_or_invalid'});continue
        names.extend(row)
    expected=set(expected_names)
    if set(names)!=expected or len(names)!=len(expected_names):
        issues.append({'reason':'chunk_point_coverage_incomplete',
            'missing':sorted(expected-set(names)),'outside_chunk':sorted(set(names)-expected),
            'duplicates':sorted({name for name in names if names.count(name)>1}),
            'expected_count':len(expected_names),'returned_name_count':len(names)})
    issues.extend(_surface_candidate_selection_audit(parsed,list(surface_candidates)))
    return issues


def _request_visual_marker_chunk(model,message,*,fai,expected_names,source_points,
        allowed_source_pages,surface_candidates=(),temperature=0.,max_repairs=2):
    """Ask the native model to repair only a malformed subset response.

    A genuine abstention stays an abstention. No valid neighboring chunks are
    regenerated here, and full FAI geometry gates remain mandatory after merge.
    """
    options={name:['C'+str(candidate['id']).removeprefix('C') for candidate in surface_candidates
        if candidate.get('id') is not None and
        (name in (candidate.get('source_point_names') or []) or
         candidate.get('candidate_role')=='global_surface_search')]
        for name in expected_names}
    contract=('\nCurrent subset output contract: markers must cover ONLY '+json.dumps(expected_names)+
        ', each name exactly once. Do not add a point from another subset to close a loop or build a plane. '+
        'Full-group totals in the source apply after the service merges all subsets. '+
        'Return the complete corrected subset yourself; the service will not trim your response. '+
        'Per-point CAD candidate IDs (search hints, not proof of entity identity): '+json.dumps(options)+'.')
    base={**message,'content':message['content']+contract}
    current=base;attempts=[];total_tokens=0
    for index in range(max_repairs+1):
        try:
            raw,truncated,tokens=model._generate_once([current],
                max_context_chars=_complete_visual_context_budget([current]),
                max_new_tokens=4096,max_images=8,temperature=temperature)
        except Exception as exc:
            attempts.append({'error':f'{type(exc).__name__}: {exc}'[:1000]})
            return None,'',True,total_tokens,[],attempts
        total_tokens+=tokens
        parsed=_object(raw) or {}
        parsed,aliases=_normalize_model_source_aliases(parsed,source_points,allowed_source_pages)
        issues=_visual_chunk_contract_issues(parsed,expected_names,fai=fai,surface_candidates=surface_candidates)
        attempts.append({'raw_output':raw,'truncated':bool(truncated),'generated_tokens':tokens,
            'source_name_aliases':aliases,'contract_issues':issues})
        if not issues and not truncated:return parsed,raw,False,total_tokens,aliases,attempts
        if parsed.get('decision')=='ABSTAIN':break
        current={**base,'content':base['content']+
            '\nYour last subset response failed the output contract. This is a naming/count/candidate check, '
            'not a geometry authorization. Re-read the same original evidence and return the entire corrected '
            'subset with the exact required names. Never invent coordinates or accept a wrong entity. '
            'Use ABSTAIN if evidence is insufficient. Diagnostics: '+json.dumps(issues,ensure_ascii=False)+
            '\nPrevious native response (unaccepted): '+raw}
    return parsed,raw,bool(truncated),total_tokens,aliases,attempts


def _declared_marker_manifest(methods):
    """Expand explicit printed measurement names without inventing XYZ rows.

    Unlike coordinate retrieval, this manifest can retain a point whose name
    has no coordinate definition in the corpus. A local model must confirm
    the complete list before it replaces a semantic-retrieval point group.
    Prose such as 'as Plane View show' is deliberately not a name list.
    """
    groups=[];seen=set()
    for page in methods or []:
        from .msop_tables import table_measurement_manifest
        table_group=table_measurement_manifest(page)
        if table_group:
            signature=tuple(sorted(table_group['source_point_names']))
            if signature not in seen:
                seen.add(signature);groups.append(table_group)
        for pattern in (MEASURE_POINT_LIST,MEASURED_GEOMETRY_LIST):
            for match in pattern.finditer(page.get('text','')):
                expected=int(match.group(1));names=[]
                if not 0<expected<=MAX_ANNOTATION_POINTS:
                    return None
                clause=re.split(r',?\s*(?:then\s+)?\bconstruct\b|[，,]\s*use\b',
                                match.group(2),maxsplit=1,flags=re.I)[0]
                from .point_names import expand_point_clause
                names=expand_point_clause(clause,limit=MAX_ANNOTATION_POINTS,
                    plane_view_suffix=pattern is MEASURE_POINT_LIST)
                if names is None:return None
                if (len(names)!=expected or len(set(names))!=expected
                        or any(not re.fullmatch(r'[A-Z][A-Z0-9]*_[A-Z0-9_]+',name) for name in names)):
                    return None
                signature=tuple(sorted(names))
                if signature not in seen:
                    seen.add(signature)
                    groups.append({'source_page':page.get('page'),'declared_count':expected,
                                   'source_point_names':names,'instruction':match.group(0)})
    return groups or None


def _resolve_declared_marker_manifest_with_local_model(model,fai,methods,source_points,
                                                       required_count,source_images):
    from .task_models import task_model
    model=task_model(model,'fai_manifest')
    groups=_declared_marker_manifest(methods)
    if not groups:
        return source_points,{'status':'not_required'}
    names=list(dict.fromkeys(name for group in groups for name in group['source_point_names']))
    if required_count and len(names)!=required_count:
        return source_points,{'status':'not_required','reason':'incomplete_explicit_manifest'}
    aliases={}
    for point in source_points:
        aliases.setdefault(_canonical_source_point_alias(point.get('name')),[]).append(point)
    matches={name:aliases.get(_canonical_source_point_alias(name),[]) for name in names}
    if (len(source_points)==len(names) and all(len(rows)==1 for rows in matches.values())
            and len({rows[0]['name'] for rows in matches.values()})==len(names)):
        return source_points,{'status':'not_required'}
    images=list(dict.fromkeys(source_images or []))[:3];attempts=[]
    question=(f'你是本地 MSOP 测量清单提取模型，只核准 FAI {fai} 的直接测量位置清单，不做 CAD 定位。'
        '下面的候选清单只按原文逗号、斜杠和连续编号范围展开；请结合原始方法页图，核对每组测量点名及数量。'
        '不要把同族其他 FAI 的点混进来，不要把 Z1 改称 L2，不要猜造坐标或认为相似点名共用 XYZ。'
        '有些明确测量点没有被坐标表检索到；这不阻止核准其点名，后续由本地视觉模型独立定位。'
        '只有每组完整、确为当前 FAI 的直接测量清单才返回 CONFIRM_MANIFEST，否则 ABSTAIN。'
        '仅输出 JSON：decision, fai, source_point_names（完整精确清单，每个点恰一次）, evidence, confidence（只能是 high、medium 或 low 字符串，不能是数值）。\n'
        f'Explicit source instructions and expanded manifest: {json.dumps(groups,ensure_ascii=False)}\n'
        f'Retrieved coordinate names (auxiliary evidence, not authoritative measurements): '
        f'{json.dumps([point.get("name") for point in source_points],ensure_ascii=False)}')
    for _ in range(2):
        messages=[{'role':'user','content':question,'attachments':[
            {'filename':'Original MSOP measurement manifest','images':images}]}]
        try:
            raw,truncated,tokens=model._generate_once(messages,max_context_chars=len(question)+512,
                max_new_tokens=min(2048,256+len(names)*24),max_images=len(images),temperature=0.0)
        except Exception as exc:
            attempts.append({'error':f'{type(exc).__name__}: {exc}'[:300]});continue
        parsed=_object(raw) or {};selected=parsed.get('source_point_names')
        valid=(not truncated and parsed.get('decision')=='CONFIRM_MANIFEST'
            and _fai_number(parsed.get('fai'))==fai and isinstance(selected,list)
            and all(isinstance(name,str) for name in selected)
            and len(selected)==len(names) and set(selected)==set(names)
            and bool(parsed.get('evidence')) and _confidence(parsed.get('confidence')) is not None)
        errors=[]
        if truncated:errors.append('response_truncated')
        if _confidence(parsed.get('confidence')) is None:errors.append('confidence_must_be_high_medium_or_low_string')
        if _fai_number(parsed.get('fai'))!=fai:errors.append('fai_identifier_mismatch')
        if not isinstance(selected,list) or any(not isinstance(n,str) for n in selected) or len(selected)!=len(names) or set(selected)!=set(names):errors.append('incomplete_or_incorrect_direct_measurement_names')
        if parsed.get('decision')!='CONFIRM_MANIFEST':errors.append('model_did_not_confirm_manifest')
        if not parsed.get('evidence'):errors.append('source_evidence_missing')
        attempts.append({'valid':bool(valid),'raw_output':raw[:3000],
                         'validation_errors':errors,'truncated':bool(truncated),'generated_tokens':tokens})
        if not valid:
            question+='\nPrevious response rejected: '+json.dumps(errors)+'. Re-read the same original evidence and emit complete corrected JSON; abstain if evidence is insufficient.'
            continue
        resolved=[];source_page={name:group['source_page'] for group in groups
                                for name in group['source_point_names']}
        method_by_page={page.get('page'):page for page in methods}
        for name in selected:
            if len(matches[name])==1:
                row=dict(matches[name][0])
            else:
                page=source_page[name];method=method_by_page.get(page,{})
                row={'name':name,'xyz':None,'pages':[page],
                     'program':method.get('program'),'coordinate_system':method.get('measured_system'),
                     'requires_variant_choice':False,'coordinate_variants':[]}
            row.update(explicit_method_manifest=True,declared_source_name=name)
            resolved.append(row)
        audit={'status':'confirmed','groups':groups,'attempts':attempts,
               'model_alias':getattr(model,'openai_model',None),
               'selected_point_names':[row['name'] for row in resolved],
               'coordinate_evidence_missing':[row['name'] for row in resolved if row.get('xyz') is None
                                              and not row.get('requires_variant_choice')],
               'excluded_retrieved_names':[point.get('name') for point in source_points
                    if point.get('name') not in {row['name'] for row in resolved}],
               'evidence':parsed['evidence'],'confidence':_confidence(parsed['confidence'])}
        return resolved,audit
    return None,{'status':'abstained','groups':groups,'attempts':attempts}


def _coordinate_variant_options(point):
    """Build exact source-row choices from explicit variants and scoped occurrences."""
    variants=list(point.get('coordinate_variants') or [])
    for occurrence in point.get('occurrences') or []:
        systems=([occurrence.get('coordinate_system')]
                 if occurrence.get('coordinate_system') else
                 occurrence.get('coordinate_system_candidates') or [])
        for coordinate_system in systems:
            variants.append(dict(name=point.get('name'),page=occurrence.get('page'),
                program=occurrence.get('program'),coordinate_system=coordinate_system,
                xyz=occurrence.get('xyz')))
    for scope in point.get('unresolved_scopes') or []:
        variants.append(dict(name=point.get('name'),page=(scope.get('pages') or [None])[0],
            program=scope.get('program'),coordinate_system=scope.get('coordinate_system'),
            xyz=scope.get('xyz')))
    unique=[];seen=set()
    for row in variants:
        xyz=row.get('xyz')
        if not isinstance(xyz,list) or len(xyz)!=3:
            continue
        key=(row.get('page'),row.get('program'),row.get('coordinate_system'),tuple(xyz))
        if key in seen:continue
        seen.add(key);unique.append({key:row.get(key) for key in
            ('name','page','program','coordinate_system','xyz')})
    return unique


def _coalesce_source_point_definitions(points):
    """Keep one marker identity and all explicit coordinate-scope alternatives.

    Source retrieval may return the same printed point name once per alignment.
    Those are alternatives for the model to disambiguate, not additional
    measurement locations. Never pick the first alignment or average its XYZ.
    """
    groups={}
    for point in points:
        groups.setdefault(point.get('name'),[]).append(point)
    result=[]
    for name,rows in groups.items():
        if len(rows)==1:
            result.append(rows[0]);continue
        variants=[];seen=set();pages=[]
        for row in rows:
            pages.extend(page for page in row.get('pages',[]) if page not in pages)
            options=_coordinate_variant_options(row)
            xyz=row.get('xyz')
            if isinstance(xyz,list) and len(xyz)==3:
                options.extend({'name':name,'page':page,'program':row.get('program'),
                    'coordinate_system':row.get('coordinate_system'),'xyz':xyz}
                    for page in row.get('pages') or [None])
            for option in options:
                key=(option.get('page'),option.get('program'),option.get('coordinate_system'),
                     tuple(option.get('xyz') or []))
                if key in seen:continue
                seen.add(key);variants.append({**option,'name':name})
        # The raw definitions stay intact, including identical rows from
        # different pages: the local model must still select source provenance.
        merged={**rows[0],'name':name,'pages':pages,'xyz':None,'program':None,
                'coordinate_system':None,'requires_variant_choice':True,
                'coordinate_variants':variants}
        result.append(merged)
    return result


def _coordinate_variant_geometry_evidence(mesh,source_points,registration_diagnostic):
    """Show every source variant under its own model-selected frame.

    This is diagnostic evidence, not a source-row selection. In particular,
    the nearest point alone cannot decide which physical feature was meant.
    """
    frames=(registration_diagnostic or {}).get('selected_frame_transforms') or []
    if mesh is None or not mesh.get('faces') or not frames:return []
    _triangles,face_ids,index=cached_surface_geometry(mesh)
    rows=[]
    for point in source_points:
        options=_coordinate_variant_options(point)
        if not point.get('requires_variant_choice') and len(options)<2:continue
        for option in options:
            frame=next((frame for frame in frames
                if frame.get('program')==option.get('program')
                and frame.get('coordinate_system')==option.get('coordinate_system')),None)
            if not frame:continue
            rotation=np.asarray(frame.get('rotation'),dtype=float)
            translation=np.asarray(frame.get('translation_mm'),dtype=float)
            if (rotation.shape!=(3,3) or translation.shape!=(3,) or
                    not np.isfinite(rotation).all() or not np.isfinite(translation).all() or
                    not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or
                    not np.isclose(np.linalg.det(rotation),1,atol=1e-5)):continue
            mapped=np.asarray(option['xyz'],float)@rotation.T+translation
            nearest=index.nearest(mapped)
            rows.append({'name':point['name'],'page':option.get('page'),
                'program':option.get('program'),'coordinate_system':option.get('coordinate_system'),
                'source_xyz':option['xyz'],'mapped_xyz':mapped.tolist(),
                'distance_to_CAD_mm':float(nearest['distance_mm']),
                'nearest_face_id':str(face_ids[nearest['triangle_index']]),
                'source_to_cad_registered':False})
    return rows


def _coordinate_variant_source_images(store,project,detail,model,fai,source_points):
    """Retain every exact competing definition page, not only the highest hit page."""
    images=_method_page_images(store,project,detail,model,fai)
    if not images:return []
    document=store.active_msop(project) or {};path=Path(document.get('path') or '')
    if not path.is_file() or not document.get('sha256'):return images
    stat=path.stat()
    if _verified_pdf_digest(str(path),stat.st_mtime_ns,stat.st_size)!=document['sha256']:return []
    pages=sorted({row['page'] for point in source_points for row in _coordinate_variant_options(point)
                  if type(row.get('page')) is int and row['page']>0})
    import pymupdf
    directory=store._dir(project['id'])/'automatic-learning'/'method-page-images'
    directory.mkdir(parents=True,exist_ok=True);coordinate_images=[]
    with pymupdf.open(path) as pdf:
        for number in pages:
            if number>len(pdf):continue
            target=directory/f"msop-{document['sha256'][:20]}-page-{number}.png"
            if not target.is_file():
                pix=pdf[number-1].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False)
                temporary=target.with_suffix('.png.tmp');temporary.write_bytes(pix.tobytes('png'))
                temporary.replace(target)
            coordinate_images.append(str(target))
    return list(dict.fromkeys([images[0],*coordinate_images,*images[1:]]))


def _coordinate_disambiguation_context(store,project,item,source_points,method_text,*,
                                      mesh=None,registration_diagnostic=None):
    """Provide the target row and definition-page context, not an entire test table alone."""
    fields={key:item.get(key,[]) for key in ('headings','features','specifications','warnings')}
    numbers={row.get('page') for point in source_points for row in _coordinate_variant_options(point)
             if isinstance(row.get('page'),int)}
    document=store.active_msop(project) or {}
    contexts=[{key:page.get(key) for key in ('page','heading','program','measured_system')} | {
        # Exact competing XYZ rows are supplied separately and never cropped.
        # This excerpt adds the definition's assembly stage and instructions.
        'definition_header_and_instructions_excerpt':page.get('text','')[:1500]}
        for page in document.get('method_pages',[]) if page.get('page') in numbers]
    geometry=_coordinate_variant_geometry_evidence(mesh,source_points,registration_diagnostic)
    return (f'Target FAI {item["number"]} fields (individual row, not all rows on the page): '
            f'{json.dumps(fields,ensure_ascii=False)}\n'
            f'Coordinate definition-page context: {json.dumps(contexts,ensure_ascii=False)}\n'
            f'CAD consistency of each exact variant under its own local-model-selected frame '
            f'(auxiliary evidence only; source_to_cad_registered=false is global background, '
            f'not a prohibition on display marking; never choose one point by nearest distance alone): '
            f'{json.dumps([{k:v for k,v in row.items() if k!="source_to_cad_registered"} for row in geometry],ensure_ascii=False,separators=(",",":"))}\n'
            f'Original target MSOP measurement methods: {method_text}')


def _validate_coordinate_variant_choices(source_points, choices):
    """Require the local model to choose among conflicting source XYZ rows."""
    ambiguous={}
    for point in source_points:
        variants=_coordinate_variant_options(point)
        distinct={(v.get('program'),v.get('coordinate_system'),tuple(v.get('xyz') or []))
                  for v in variants if isinstance(v.get('xyz'),list) and len(v['xyz'])==3}
        if (point.get('requires_variant_choice') or point.get('coordinate_conflict')
                or point.get('scope_ambiguous') or len(distinct)>1):
            ambiguous[point['name']]=variants
    if not ambiguous:return choices in (None,[]),[]
    if not isinstance(choices,list) or len(choices)!=len(ambiguous):return False,[]
    selected=[]
    for choice in choices:
        if not isinstance(choice,dict) or choice.get('name') not in ambiguous:return False,[]
        valid=next((row for row in ambiguous[choice['name']]
                    if choice.get('page')==row.get('page')
                    and choice.get('program')==row.get('program')
                    and choice.get('coordinate_system')==row.get('coordinate_system')
                    and choice.get('xyz')==row.get('xyz')),None)
        if valid is None or choice['name'] in {row['name'] for row in selected}:return False,[]
        selected.append(dict(name=choice['name'],page=valid.get('page'),program=valid.get('program'),
                             coordinate_system=valid.get('coordinate_system'),xyz=valid.get('xyz')))
    return {row['name'] for row in selected}==set(ambiguous),selected


def _select_coordinate_variants_with_local_model(model, fai, source_points, method_text,
                                                *, source_images=None):
    """Ask the configured local model to resolve only explicit XYZ conflicts."""
    ambiguous=[]
    for point in source_points:
        variants=_coordinate_variant_options(point)
        distinct={(v.get('program'),v.get('coordinate_system'),tuple(v.get('xyz') or []))
                  for v in variants if isinstance(v.get('xyz'),list) and len(v['xyz'])==3}
        if (point.get('requires_variant_choice') or point.get('coordinate_conflict')
                or point.get('scope_ambiguous') or len(distinct)>1):
            ambiguous.append({'name':point.get('name'),'variants':[
                {key:variant.get(key) for key in ('page','program','coordinate_system','xyz')}
                for variant in variants if isinstance(variant.get('xyz'),list)
                and len(variant['xyz'])==3]})
    if not ambiguous:
        return [],{'status':'not_required','attempts':[]}
    if any(not row['name'] or not row['variants'] for row in ambiguous):
        return None,{'status':'invalid_source_variants','attempts':[]}
    # Full FAI groups can have dozens of conflicting source definitions. A
    # single response repeating all rows exceeds the model's output budget.
    # Split the source-row decision, not the geometry: validate and join every
    # exact model choice before attempting the complete FAI placement.
    if len(ambiguous)>6:
        selected=[];chunk_audits=[]
        by_name={point['name']:point for point in source_points}
        for start in range(0,len(ambiguous),6):
            chunk=[by_name[row['name']] for row in ambiguous[start:start+6]]
            choices,audit=_select_coordinate_variants_with_local_model(
                model,fai,chunk,method_text,source_images=source_images)
            chunk_audits.append(audit)
            if choices is None:
                return None,{'status':'abstained','attempts':[],
                    'failed_chunk':len(chunk_audits),'chunks':chunk_audits}
            selected.extend(choices)
        valid,normalized=_validate_coordinate_variant_choices(source_points,selected)
        if not valid:
            return None,{'status':'invalid_merged_source_variants','attempts':[],'chunks':chunk_audits}
        return normalized,{'status':'selected','attempts':[],'chunks':chunk_audits,
            'selected_choices':normalized,
            'evidence':'Complete exact source-row choices from local-model batches.'}
    attempts=[];correction=''
    for attempt in range(3):
        question=(
            VIEWER_LOCATION_TASK+
            f"你是 Oracle Viewer 的本地 MSOP 坐标消歧模型。仅为 FAI {fai} 选择原文中已有的 XYZ 行；"
            "这一步不做 CAD 定位，也不推算/修改坐标。依据本 FAI 的实体特征、原文几何定义、程序/坐标系与三件套一致性证据判断每个点应采用哪一个 page/program/coordinate_system/xyz 版本；"
            "对每个冲突点必须恰选一个完整原文选项，不得按列表先后、平均坐标或自行补造。若全部证据仍无法唯一决定，返回 ABSTAIN。"
            "若提供了原始 MSOP 页图，请核对坐标表、测量步骤与坐标系标题在版面上的对应区域；"
            "同一页有多个坐标系时不能只看整页标题或忽略图中的分块。"
            "只输出 JSON：decision (SELECT_VARIANTS 或 ABSTAIN), fai, coordinate_choices (数组，每项完整复制 name/page/program/coordinate_system/xyz), evidence, confidence。\n"
            '若附有各原文候选在其各自已选坐标系下的 CAD 一致性诊断，可结合实体特征和完整点组判断；'
            '不能只给单个点选最小残差。力学试验本身不需要定义 CMM 坐标系，但同一实体的几何支撑点仍可用于位置标记。'
            '同一实体在其他 FAI 方法页中定义的点可作为支撑，不因页码属于另一 FAI 就自动失效；必须核对实体名称和关联。'
            'source_to_cad_registered=false 表示未通过全局计量注册，不是禁止选择原文行或低置信显示位置；本轮也不得宣称注册已通过。'
            f"MSOP measurement methods: {method_text}\n"
            f"Competing source rows: {json.dumps(ambiguous,ensure_ascii=False,separators=(',',':'))}\n"
            f"CURRENT_OUTPUT_POINT_NAMES: {json.dumps([row['name'] for row in ambiguous])}\n"
            f"本次 coordinate_choices 必须恰好 {len(ambiguous)} 项，点名仅限 CURRENT_OUTPUT_POINT_NAMES，每名一次。"
            '原始全组点表或图片中出现的其他点只供上下文核对，不得加入本次输出；evidence 只写一句，最多40个英文词。\n'
            +correction)
        if len(question)>24000:
            return None,{'status':'source_variant_context_exceeds_budget','attempts':attempts,
                         'context_chars':len(question),'budget_chars':24000}
        messages=[{'role':'user','content':question}]
        supplied=list(dict.fromkeys(source_images or []))
        wanted_pages={row['page'] for point in source_points for row in _coordinate_variant_options(point)
                      if type(row.get('page')) is int}
        images=[]
        if supplied:
            images.append(supplied[0])  # original target method page
            for path in supplied[1:]:
                match=re.search(r'^msop-.*-page-(\d+)\.png$',Path(path).name)
                if not match or int(match.group(1)) in wanted_pages:images.append(path)
        if len(images)>16:
            return None,{'status':'source_variant_image_context_exceeds_budget','attempts':attempts,
                         'source_image_count':len(images),'budget_images':16}
        if images:
            messages[0]['attachments']=[{'filename':'Original MSOP coordinate and method pages',
                                        'images':images}]
        try:
            lock=getattr(model,'_lock',None)
            with lock if lock is not None else nullcontext():
                raw,truncated,tokens=model._generate_once(
                    messages,max_context_chars=len(question)+512,
                    max_new_tokens=2048,max_images=len(images),temperature=0.0)
        except Exception as exc:
            attempts.append({'error':f'{type(exc).__name__}: {exc}'[:500]})
            correction='上次本地调用失败，请按同一输入重试，只返回规定 JSON。'
            continue
        parsed=_object(raw) or {}
        valid_choices,normalized=_validate_coordinate_variant_choices(
            source_points,parsed.get('coordinate_choices'))
        valid=(not truncated and parsed.get('decision')=='SELECT_VARIANTS'
               and _fai_number(parsed.get('fai'))==fai and valid_choices)
        attempts.append({'raw_output':raw,'generated_tokens':tokens,
                         'truncated':bool(truncated),'valid':bool(valid)})
        if valid:
            return normalized,{'status':'selected','attempts':attempts,
                'selected_choices':normalized,
                'source_image_paths':images,
                'confidence':_confidence(parsed.get('confidence')),
                'evidence':str(parsed.get('evidence') or '')[:500]}
        expected_names={row['name'] for row in ambiguous}
        reported_choices=parsed.get('coordinate_choices')
        reported_names=[row.get('name') for row in (reported_choices if isinstance(reported_choices,list) else [])
            if isinstance(row,dict) and isinstance(row.get('name'),str)]
        scope_diagnostic={'missing':sorted(expected_names-set(reported_names)),
            'outside_current_subset':sorted(set(reported_names)-expected_names),
            'duplicates':sorted({name for name in reported_names if reported_names.count(name)>1})}
        correction=(
            '上次回答未能为每个冲突点精确选择有效原文行。校验结果：'
            f"decision={parsed.get('decision')!r}, valid_choices={valid_choices}, truncated={truncated}。"
            f"本次点名范围审计：{json.dumps(scope_diagnostic,ensure_ascii=False)}。"
            '请检查上述各点自己的 variants 清单，逐项精确复制一个选项；只能输出 CURRENT_OUTPUT_POINT_NAMES，'
            '程序不会替你删掉多余点或默认选择原文行；无法唯一判断时才 ABSTAIN。')
    return None,{'status':'abstained','attempts':attempts}


def _prompt_data(project, item, detail, old, mesh_faces, method_pages):
    candidates = old.get('anchors', [])
    candidate_ids = list(dict.fromkeys(a['face_id'] for a in candidates if a['face_id'] in mesh_faces))
    catalog = []
    for face_id in candidate_ids:
        anchors = [a for a in candidates if a['face_id'] == face_id]
        face = mesh_faces[face_id]
        catalog.append({'face_id': face_id, 'surface_type': face.get('surface_type') or face.get('surface'),
                        'center_mm': face.get('center'), 'candidate_anchor_count': len(anchors),
                        'sample_anchor_mm': anchors[0]['position']})
    method = detail['methods'][0] if detail['methods'] else {}
    method_text = '\n'.join(
        f"[MSOP page {p.get('page', '?')}] " + re.sub(r'\s+', ' ', p.get('text', '')).strip()
        for p in detail['methods'])
    explicit_measure_points = _explicit_measure_point_names(
        detail['methods'], extract_named_points(method_pages))
    point_basis = _measure_point_basis(detail['methods'], extract_named_points(method_pages))
    proposal = {
        'candidate_face_catalog': catalog,
        'candidate_anchors_mm': [{'face_id': a['face_id'], 'position': [round(v, 3) for v in a['position']]}
                                 for a in candidates],
        'candidate_count': len(candidates), 'omitted_anchor_count': 0,
        'provenance': old.get('origin', 'geometry_candidate'),
        'surface_validation': 'Each anchor is on its listed active CAD display triangle; it may be interpolated inside a triangle.',
        'review_state': 'automatic candidate; evidence and model confidence retained',
        'note': ('Automatic XYZ candidate; complete per-point evidence is in point_correspondences and the numeric audit below.'
                 if old.get('automatic_xyz_hypothesis') else old.get('note', '')[:1600]),
    }
    if explicit_measure_points:
        proposal['explicit_measured_point_names'] = sorted(explicit_measure_points)
        proposal['explicit_measure_point_count'] = len(explicit_measure_points)
        proposal['explicit_measured_point_basis'] = point_basis
    if old.get('candidate_kind') == 'same_source_stale_seed':
        proposal['same_source_stale_seed_audit'] = old.get('seed_evidence', {})
    if old.get('automatic_xyz_hypothesis'):
        evidence=old.get('evidence') or {}
        correspondences=evidence.get('point_correspondences') or []
        residuals=[float(row['residual_mm']) for row in correspondences
                   if isinstance(row.get('residual_mm'),(int,float))]
        plane_residuals=[float(row['plane_residual_mm']) for row in correspondences
                         if isinstance(row.get('plane_residual_mm'),(int,float))]
        proposal['fai_scoped_numeric_audit']={
            'source_point_count':len(correspondences),
            'mapped_point_count':len(candidates),
            'all_source_points_retained':evidence.get('all_source_points_retained') is True,
            'max_point_to_CAD_surface_distance_mm':max(residuals) if residuals else None,
            'point_distance_gate_mm':0.1,
            'all_point_distances_within_gate':bool(residuals) and len(residuals)==len(correspondences)
                and max(residuals)<=0.1,
            'max_anchor_deviation_from_transformed_source_section_mm':max(plane_residuals,default=0.0),
            'section_deviation_gate_mm':evidence.get('max_surface_anchor_section_deviation_mm',0.02),
            'all_source_sections_preserved':evidence.get('source_sections_preserved') is True,
            'scope':'this FAI only; excludes unrelated points from the same program/frame',
        }
        proposal['frame_level_registration_diagnostic']=[{
            'program':frame.get('program'),'coordinate_system':frame.get('coordinate_system'),
            'status':frame.get('global_status'),'frame_wide_max_residual_mm':frame.get('global_max_residual_mm'),
            'interpretation':'unpaired whole-frame diagnostic; includes other FAI point groups and is not this FAI point residual'
        } for frame in evidence.get('frame_hypotheses',[])]
        proposal['coordinate_page_resolution_audit']=evidence.get('coordinate_page_resolutions',[])
    if old.get('automatic_xyz_hypothesis'):
        # Keep model-review evidence aligned with candidate generation. A broad
        # second expansion can reintroduce duplicate point names that the
        # nominal-axis/page resolver has already disambiguated.
        cross_ref_payload=[]
        for source_row in old.get('evidence', {}).get('point_correspondences', []):
            row={key:source_row[key] for key in (
                'name','point_key','program','coordinate_system','source_pages','source_xyz',
                'transformed_xyz','position','face_id','residual_mm','plane_residual_mm',
                'semantic_resolution','coordinate_page_resolution','coordinate_variant_span_mm')
                if key in source_row}
            # Repeated identical definitions are covered by source_pages and
            # the page-resolution audit. Keep full alternatives whenever the
            # source evidence signals a real coordinate span or a conflict.
            variants=source_row.get('source_coordinate_variants') or []
            if source_row.get('coordinate_conflict') or source_row.get('coordinate_variant_span_mm',0)>0:
                row['source_coordinate_variants']=variants
            cross_ref_payload.append(row)
        named_points=extract_named_points(method_pages)
        feature_labels=extract_feature_labels(detail['methods'],item.get('features',[]))
        from .candidate_mining import semantic_point_references
        semantic_links=semantic_point_references(feature_labels,named_points)
        for point in cross_ref_payload:
            name=point.get('name')
            point['semantic_feature_links']=sorted(feature for feature,rows in semantic_links.items()
                if any(row.get('name')==name for row in rows))
            source=named_points.get(name,{})
            point['constructed_point_sets']=source.get('constructed_point_sets',[])
    else:
        named_points = extract_named_points(method_pages)
        feature_labels = extract_feature_labels(detail['methods'], item.get('features', []))
        cross_refs = referenced_points(None, detail['methods'], named_points, feature_labels)
        from .candidate_mining import semantic_point_references
        semantic_links=semantic_point_references(feature_labels,named_points)
        linked_features={name:sorted(feature for feature,rows in semantic_links.items()
                                     if any(row.get('name')==name for row in rows))
                         for name in cross_refs}
        cross_ref_payload=[]
        for point in cross_refs.values():
            row={k:point.get(k) for k in ('name','point_key','xyz','pages','fai_ids','program',
                 'coordinate_system','coordinate_conflict','scope_ambiguous','unresolved_scopes')}
            row['semantic_feature_links']=linked_features.get(point.get('name'),[])
            row['constructed_point_sets']=point.get('constructed_point_sets',[])
            cross_ref_payload.append(row)
    proposal['cross_page_point_references'] = cross_ref_payload
    context = (
        f"Drawing {project.get('drawing_no')} Rev {project.get('revision')}; FAI {item['number']}\n"
        f"MSOP extracted fields: {json.dumps({'headings': item.get('headings', []), 'features': item.get('features', []), 'specifications': item.get('specifications', []), 'warnings': item.get('warnings', [])}, ensure_ascii=False)}\n\n"
        f"Explicit measured point list: {json.dumps(sorted(explicit_measure_points) if explicit_measure_points else [], ensure_ascii=False)}\n"
        f"Explicit point-list basis: {point_basis or 'no complete MSOP point group could be resolved'}\n"
        f"Interpretation: when this list is present, it is resolved from the MSOP 'Measure N vector points' instruction, or from an exact-count XYZ table on that same method page when the instruction says 'as Plane View show'. It is the authoritative input point group. When no complete explicit list resolves, inspect cross_page_point_references: a specific semantic_feature_links entry is a retrieved association hypothesis from a distinctive shared feature family, not an arbitrary nearby point. It may support a constructed vision circle/line/plane only when program, coordinate system, point geometry, and source-page evidence are coherent; the local model must make that association decision. Keep all support-point provenance and abstain on mixed families or unresolved scope. The extracted 'features' field can also contain nearby XYZ location-definition labels that are not measured by this FAI; labels without a specific semantic association may be auxiliary and must not be treated as measured points.\n\n"
        f"Geometry proposal and provenance: {json.dumps(proposal, ensure_ascii=False)}\n\n"
        f"[MSOP methods and cross-page point references; location rows are cited above]\n"
        f"{project.get('title') or project.get('drawing_no') or 'FAI project'} | MSOP: {method_text}"
    )
    question = (
        VIEWER_LOCATION_TASK+
        '审核该 FAI→CAD 几何候选。预处理已验证 face_id 和锚点位于当前 CAD 显示网格；候选生成仍可能有坐标系或近邻误差，不能因候选非空就接受。'
        '若附有原始MSOP方法页或2D工程图图像，先核对图中FAI编号、测量特征、视图和点组；图像只用于核对来源对应关系，不得据图猜造XYZ或候选face_id。'
        '这是CAD上的FAI位置可视化建议，不是测量结果。先核对MSOP点名/页码、校准公式和逐点对应表。逐点表中的映射坐标已经应用变换；CAD表面锚点与点位是最近XY检索，CAD面中心不等于点位。'
        '点名必须结合程序和坐标系解释，不得跨UP/DOWN套用坐标。若上下文提供完整的explicit_measured_point_names，以MSOP测量指令列出的点名和数量作为本FAI的测量点组；提取字段features可能混入同页其他位置定义点，这些未列入测量指令的点不得作为冲突依据。若该列表为空，只有cross_page_point_references中明确列出的semantic_feature_links或MSOP构造点集，才可作为待模型判断的构造特征支撑组；对vision circle可核对同族XYZ点是否构成一致圆周并用其完整支撑点定位，不能仅因方法页未逐一列出支撑点就自动弃答。模型必须确认点组与特征、程序、坐标系、页码和CAD候选面一致，否则ABSTAIN。coordinate_conflict或scope_ambiguous为true时必须ABSTAIN，不能以低置信度代替消除冲突。'
        '不得用未变换的原始坐标与CAD锚点直接比较，也不得把检索残差误作测量误差或拒绝所有非零残差。'
        '若特征名和页码对应，至少一半显式点位在校准门限内，且候选CAD锚点与mapped_xy一致，可自动标记并给出置信度；内点率低于一半、变换缺失或特征/区域冲突时弃答。'
        '自动显示标记不表示计量验证。只能从 candidate_face_catalog 原样选择 face_id，最多选择 12 个，不得编造 ID 或坐标。'
        '只输出单行 JSON，字段 decision,fai,face_ids,source_pages,evidence,confidence。'
        'decision只能为AUTO_MARK或ABSTAIN。候选生成器已硬性检查完整点组、坐标来源、逐点距离和截面门限；整帧注册状态只作背景，不决定本FAI的自动显示标记。'
        '本流程不等待人工审批；低置信度仍可自动落标，但必须保留低置信度和原始证据。'
        'face_ids 必须是数组；弃答时为空数组。'
        f"fai 字段应为整数 {item['number']}，不要输出带FAI前缀的字符串。"
    )
    if old.get('candidate_kind') == 'same_source_stale_seed':
        question += ('候选来自同一CAD与MSOP文件哈希下的旧自动标记，旧状态因配准算法更新而失效。它只是待重新判断的位置线索，'
                     '不得继承旧决策；必须重新核对当前MSOP方法、点位/视图和2D图纸。只有FAI特征与候选位置明确对应时才自动标记，否则ABSTAIN。')
        if not old.get('automatic_xyz_hypothesis'):
            question += ('该旧位置没有可完整复核的XYZ对应证明，输出显示置信度必须为low，且不可宣称坐标系注册或计量验证通过。')
    if old.get('automatic_xyz_hypothesis') and old.get('evidence',{}).get('coordinate_page_resolutions'):
        question += ('若存在coordinate_page_resolution_audit，重复点名并未被静默覆盖：候选选择了同一程序/坐标系中覆盖率唯一最高的完整坐标定义页，'
                     '并保留旧页版本供核对。只有当所选页的点组确实与本FAI特征对应时才可提出候选；若页码/特征不能对应，必须ABSTAIN。'
                     '不得因已解释的旧页坐标差异自动放宽0.1 mm点面距离门限。')
    if old.get('automatic_xyz_hypothesis'):
        question += ('该候选使用完整三维刚体变换假设，不是旧的最近XY检索。'
                     '先读fai_scoped_numeric_audit：这是生成器针对当前FAI全部点计算的逐点门限检查。'
                     'frame_level_registration_diagnostic是同程序/坐标系内所有FAI混合后的背景诊断，不能代替本FAI逐点残差。'
                     'note列出每个程序/坐标系的rotation和translation_mm、全局诊断，以及本FAI每点的source_xyz/transformed_xyz/position/residual_mm。'
                     '全局source_to_cad_registered=false必须保留：不能宣称配准或计量通过。'
                     '若fai_scoped_numeric_audit显示所有原文点均被保留、逐点三维距离≤0.1mm、截面偏离≤其明确上限且MSOP特征区域一致，允许提出低置信位置候选；'
                     '不得将frame_wide_max_residual_mm套用为本FAI点残差，也不得因其他FAI点造成的整帧拟合失败而拒绝本项。'
                     '只有MSOP特征/点名/区域不对应、源点冲突、点缺失或本FAI自己的数值门限失败时才弃答。'
                     '必须接受完整候选面组，不得删点或只保留好匹配的点。')
    return question, context, candidate_ids, candidates, method


def _mark_visual_fallback(store, project, item, detail, model, mesh, candidate=None):
    """Let the local vision model choose an orthographic CAD pixel for no-XYZ FAI."""
    info=model.info() if hasattr(model,'info') else {}
    if info.get('image_inference') not in ('strata_native_vision','local_auxiliary_vision','configured_multimodal_backend'):
        return dict(fai=item['number'],result='abstain_or_invalid',
                    reason='local_multimodal_vision_model_unavailable')
    images=_method_page_images(store,project,detail,model,item['number'])
    part_context=(candidate or {}).get('evidence',{}).get('part_identity_context') or {}
    allowed_part_faces=set()
    if part_context:
        from .part_identity import cad_parts,render_part_catalog
        parts=cad_parts(mesh);selected_ids=part_context.get('part_ids') or []
        selected=[part for part in parts if part['part_id'] in selected_ids]
        if (part_context.get('cad_sha256')!=mesh['source_sha256'] or not selected
                or len(selected)!=len(set(selected_ids))):
            return dict(fai=item['number'],result='abstain_or_invalid',reason='part_identity_evidence_invalid')
        allowed_part_faces={face_id for part in selected for face_id in part['face_ids']}
        images=list(dict.fromkeys([*images,*render_part_catalog(mesh,selected,
            store._dir(project['id'])/'automatic-learning'/'part-catalog')]))
    msop_image=next((path for path in images if Path(path).name.startswith('msop-')),None)
    drawing_image=next((path for path in images if Path(path).name.startswith('drawing-')),None)
    if not msop_image:
        return dict(fai=item['number'],result='abstain_or_invalid',reason='source_method_image_unavailable')
    # Renderer format is part of the cache identity. UV-grid overlays must
    # invalidate pre-grid atlases for projects already opened in the Viewer.
    atlas_path=store._dir(project['id'])/'automatic-learning'/f"cad-view-atlas-uv-grid-v2-{mesh['source_sha256'][:20]}.png"
    try:
        if not atlas_path.is_file():render_cad_view_atlas(mesh,atlas_path)
    except (OSError,ValueError) as exc:
        return dict(fai=item['number'],result='abstain_or_invalid',reason='cad_view_atlas_failed',
                    detail=str(exc)[:300])
    model_images=list(dict.fromkeys([*images,str(atlas_path)]))
    method_text='\n'.join(f"[MSOP p.{method.get('page','?')}] {method.get('text','')}"
                          for method in detail.get('methods',[]))
    dependency_text='\n'.join(f"[MSOP p.{page.get('page','?')}] {page.get('text','')}"
                              for page in detail.get('dependencies',[])
                              if page.get('page') not in {method.get('page') for method in detail.get('methods',[])})
    extracted_fields=json.dumps({'headings':item.get('headings',[]),'features':item.get('features',[]),
        'specifications':item.get('specifications',[]),'warnings':item.get('warnings',[])},ensure_ascii=False)
    candidate=candidate or {}
    evidence=candidate.get('evidence') or {}
    known_source_points=_coalesce_source_point_definitions(
        candidate.get('known_source_points') or evidence.get('known_source_points') or [])
    required_marker_count=(candidate.get('required_marker_count') or evidence.get('required_marker_count') or 0)
    required_marker_groups=(candidate.get('required_marker_groups') or evidence.get('required_marker_groups') or [])
    drawing_manifest_audit={'status':'not_required'}
    if not required_marker_count and not known_source_points:
        drawing=next((row for row in project.get('documents',[]) if row.get('role')=='drawing'),{})
        drawing_group,drawing_manifest_audit=_resolve_drawing_location_manifest(
            model,item['number'],images,{**drawing,'local_model_part_identity_context':part_context})
        if drawing_manifest_audit['status']!='not_available':
            audit_path=store._dir(project['id'])/'automatic-learning'/(
                f"fai-drawing-manifest-{mesh['source_sha256'][:16]}-{item['number']}.json")
            temporary=audit_path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(drawing_manifest_audit,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(audit_path)
            if drawing_group is None:
                return dict(fai=item['number'],result='abstain_or_invalid',
                    reason='local_drawing_location_manifest_not_confirmed',local_model_audit=drawing_manifest_audit)
            required_marker_count=drawing_group['declared_count']
            required_marker_groups=[drawing_group]
            evidence={**evidence,'drawing_location_manifest_audit':drawing_manifest_audit}
    known_source_points,source_manifest_audit=_resolve_declared_marker_manifest_with_local_model(
        model,item['number'],detail.get('methods',[]),known_source_points,
        required_marker_count,images)
    if source_manifest_audit['status']!='not_required':
        manifest_path=store._dir(project['id'])/'automatic-learning'/(
            f"fai-source-manifest-{mesh['source_sha256'][:16]}-{item['number']}.json")
        manifest_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(source_manifest_audit,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(manifest_path)
    if known_source_points is None:
        return dict(fai=item['number'],result='abstain_or_invalid',
                    reason='local_source_manifest_not_confirmed',local_model_audit=source_manifest_audit)
    strategy=evidence.get('fai_strategy') or {}
    if (strategy.get('next_route')=='evidence_repair'
            and source_manifest_audit.get('coordinate_evidence_missing')):
        return dict(fai=item['number'],result='abstain_or_invalid',
            reason='classified_source_relation_graph_unresolved',
            local_model_audit=source_manifest_audit,strategy_route='evidence_repair')
    if source_manifest_audit['status']=='confirmed':
        chosen={row['declared_source_name']:row['name'] for row in known_source_points}
        required_marker_groups=[{**group,'source_point_names':[
            chosen[name] for name in group['source_point_names']]}
            for group in source_manifest_audit['groups']]
        evidence={**evidence,'source_manifest_audit':source_manifest_audit}
    evidence={**evidence,'known_source_points':known_source_points}
    candidate={**candidate,'evidence':evidence,'known_source_points':known_source_points,
               'required_marker_groups':required_marker_groups}
    registration_diagnostic=evidence.get('registration_diagnostic')
    rejected_atlas=None
    group_result=(None if part_context else _mark_model_selected_xyz_group(store,project,item,detail,model,mesh,candidate))
    group_review_diagnostic=None
    if group_result is None and strategy.get('evidence_class','').startswith('complete_xyz_'):
        return dict(fai=item['number'],result='abstain_or_invalid',
            reason='classified_xyz_frame_evidence_unavailable',strategy_route=strategy.get('next_route'),
            source_point_count=len(known_source_points),classified_route_no_pixel_guessing=True)
    if group_result is not None:
        if group_result.get('result')=='accepted':
            return group_result
        # A bad frame-level transform or a first-pass visual abstention is not
        # the end of automatic marking. Let the local model make a fresh,
        # FAI-scoped correspondence from the original MSOP points and CAD
        # atlas below. Keep the failed attempt visible as evidence so the
        # retry cannot silently inherit its placement.
        if strategy.get('evidence_class','').startswith('complete_xyz_'):
            return {**group_result,'strategy_route':strategy.get('next_route'),
                'classified_route_no_pixel_guessing':True}
        if group_result.get('reason') in (
                'model_xyz_group_review_abstained_or_invalid',
                'model_xyz_group_geometry_gate_failed',
                'no_fai_local_frame_candidate_passed_geometry_gate'):
            group_review_diagnostic={key:value for key,value in group_result.items()
                                     if key not in ('raw_output',)}
            rejected_atlas=group_result.get('model_candidate_atlas_path')
            if rejected_atlas and Path(rejected_atlas).is_file():
                # A rejected overlay can still point the local model toward
                # the right feature region. It is a search hint only; the
                # subsequent FAI-level correspondences must pass fresh gates.
                model_images=list(dict.fromkeys([*images,rejected_atlas]))
        else:
            return group_result
    ambiguous_coordinate_points=[point['name'] for point in known_source_points
        if point.get('requires_variant_choice') or len({(variant.get('program'),variant.get('coordinate_system'),tuple(variant.get('xyz') or []))
                for variant in point.get('coordinate_variants',[])
                if isinstance(variant.get('xyz'),list) and len(variant['xyz'])==3})>1]
    preselected_coordinate_choices=[];coordinate_variant_audit={'status':'not_required','attempts':[]}
    if ambiguous_coordinate_points:
        preselected_coordinate_choices,coordinate_variant_audit=_select_coordinate_variants_with_local_model(
            model,item['number'],known_source_points,
            _coordinate_disambiguation_context(store,project,item,known_source_points,method_text,
                mesh=mesh,registration_diagnostic=registration_diagnostic),
            source_images=_coordinate_variant_source_images(store,project,detail,model,item['number'],known_source_points))
        variant_audit_path=store._dir(project['id'])/'automatic-learning'/(
            f"fai-coordinate-variants-{mesh['source_sha256'][:16]}-{item['number']}.json")
        variant_audit_path.parent.mkdir(parents=True,exist_ok=True)
        variant_audit_temp=variant_audit_path.with_suffix('.json.tmp')
        variant_audit_temp.write_text(json.dumps(coordinate_variant_audit,ensure_ascii=False,indent=2),
                                      encoding='utf-8')
        variant_audit_temp.replace(variant_audit_path)
        if preselected_coordinate_choices is None:
            return dict(fai=item['number'],result='abstain_or_invalid',
                reason='local_coordinate_variant_selection_failed',
                source_point_names=ambiguous_coordinate_points,
                local_model_audit=coordinate_variant_audit)
    rigid_group_possible=(len(known_source_points)>=3
        and len({point.get('name') for point in known_source_points})==len(known_source_points)
        and all(point.get('requires_variant_choice') or
                isinstance(point.get('xyz'),list) and len(point['xyz'])==3
                for point in known_source_points))
    image_description=("Attached images in order: original MSOP method page; additional MSOP coordinate/Plane View page when available; matching 2D drawing and exact-FAI high-resolution original drawing details when available; rejected model-selected XYZ overlay. Drawing details retain source geometry and callout leaders; they are not CAD views and their pixel coordinates cannot be used as CAD UV. The colored XYZ overlay is only a coarse search hint, not a validated placement. "
        if rejected_atlas else
        "Attached images in order: original MSOP method page; additional MSOP coordinate/Plane View page when available; matching 2D drawing and exact-FAI high-resolution original drawing details when available; active CAD orthographic atlas. Drawing details retain source geometry and callout leaders; they are not CAD views and their pixel coordinates cannot be used as CAD UV. ")
    context=(f"Drawing {project.get('drawing_no')} Rev {project.get('revision')}; FAI {item['number']}\n"
             f"Extracted FAI fields: {extracted_fields}\n"
             f"Local-model physical part identity (all anchors must belong to these selected CAD bodies): {json.dumps(part_context,ensure_ascii=False)}\n"
             f"Required distinct marker count from explicit MSOP Measure instructions: {required_marker_count or 'not explicit'}\n"
             f"Required marker groups transcribed from distinct MSOP instructions: {json.dumps(required_marker_groups,ensure_ascii=False,separators=(',',':'))}\n"
             f"MSOP source point rows (XYZ null means no coordinate evidence; never substitute similar names): {json.dumps(known_source_points,ensure_ascii=False,separators=(',',':'))}\n"
             f"Local-model direct-measurement manifest confirmation: {json.dumps({k:v for k,v in source_manifest_audit.items() if k!='attempts'},ensure_ascii=False,separators=(',',':'))}\n"
             f"Local-model method classification and selected strategy: {json.dumps({k:v for k,v in evidence.get('fai_strategy',{}).items() if k in ('measurement_family','evidence_class','failure_class','next_route','reason')},ensure_ascii=False)}\n"
             f"Point names with competing coordinate variants requiring a local-model choice: {json.dumps(ambiguous_coordinate_points,ensure_ascii=False)}\n"
             f"Independent local-model coordinate-version selections (source rows only; not yet a CAD registration): {json.dumps(preselected_coordinate_choices,ensure_ascii=False,separators=(',',':'))}\n"
             f"Rejected geometry diagnostic (never an authorized placement): {json.dumps(_compact_registration_diagnostic(registration_diagnostic),ensure_ascii=False,separators=(',',':'))}\n"
             f"Rejected first-pass XYZ group review (must not be reused): {json.dumps(_compact_registration_diagnostic(group_review_diagnostic),ensure_ascii=False,separators=(',',':'))}\n"
             f"MSOP method evidence:\n{method_text}\nRelated datum/context pages:\n{dependency_text[:9000]}\n"+
             image_description+
             "The atlas panels are +Z, -Z, +X, -X, +Y, -Y in row-major order. Each panel uses its own normalized u,v coordinates from 0 to 1, left-to-right and top-to-bottom; faint grid lines and numeric edge labels are spaced at 0.1. Never measure UV across the whole 3x2 atlas.")
    count_instruction=(f'原文明确要求 {required_marker_count} 个位置，markers 必须恰好包含 {required_marker_count} 项。'
                       if required_marker_count else '')
    group_instruction=(
        '按 Required marker groups 逐条兑现原文测量组：每个 group 的 declared_count 都是该指令要求的不同测量位置数，'
        'source_point_names 列出的点必须全部且仅覆盖一次（同一物理点被多个构造特征共同引用时可在 source_point_names 中复用，但 marker 位置不得重复）；'
        '不同指令组不能相互抵消。measured_features 组若没有点名，也必须为该组提供 declared_count 个独立 marker，并在 feature 中标明原文特征。'
        '除原文直接测量的特征外，不要把 DATUM/构造平面或构造线额外计作测量点。'
        if required_marker_groups else '')
    coordinate_variant_instruction=(
        '冲突 XYZ 行已由本地模型在独立坐标消歧步骤选定；其完整原文选择为 '
        f"{json.dumps(preselected_coordinate_choices,ensure_ascii=False,separators=(',',':'))}。"
        '后续只能原样复用这些 name/page/program/coordinate_system/xyz，不得重选、改写或省略。'
        if ambiguous_coordinate_points else '')
    location_only_test=bool(re.search(
        r"\b(?:destructive|pull\s+test|shear\s+test|torque\s+test|weld[- ]tab\s+pull|"
        r"cannot\s+be\s+measured|can[’']t\s+measure)\b",method_text,re.I))
    location_only_instruction=(
        '原文说明这是破坏性/拉力/剪切/扭矩测试，或不能由当前测量机测量；这不取消 Viewer 的位置标注任务。'
        '本轮只需把该 FAI 对应的实体试验位置标在 CAD 上，使用同一条目关联的 XYZ 点名、特征名和图纸视图；'
        '不得声称 AOI 可以执行破坏性或力学测试，也不得把试验力值当成视觉尺寸。只有找不到可识别的物理位置时才弃答。'
        if location_only_test else '')
    question=(
        VIEWER_LOCATION_TASK+
        '你是负责完成本次自动标记的本地模型；你现在有权依据随附的 MSOP 原页、2D工程图、已提取 XYZ 证据和当前 CAD 六视图图集自行提出并授权一个新的 FAI 级放置。无需等待外部或人工批准。'
        '此前的坐标变换或点组审核若被标为 rejected，表示该方案已被数值门限否决，不能沿用；它不代表这个 FAI 无法定位。'
        '若附图中有彩色 XYZ 叠加，那是已被数值门限拒绝的粗略搜索提示，只帮助找到对应 CAD 特征区域；不得将它当成已验证位置。必须重新给出稀疏配准参考和完整测量点归属，程序会独立拟合并校验。'
        '先从原文确认确切 FAI、被测特征和视图，再在 CAD 图集中选择能看到该特征的视图及特征中心位置。'
        '请列出此 FAI 所需的全部不同测量位置：一个特征一个位置；若原文表示多个点/焊点，则必须逐个列出，不能只给代表点。'
        + count_instruction +
        group_instruction +
        ('原文直接测量点清单已由本地模型确认；markers 每项必须含 source_point_names，'
         '完整覆盖清单中的每个不同点名，不能替换为相似点。XYZ 为 null 的点只能依据原图独立定位，不得借用其他点的坐标。'
         if source_manifest_audit['status']=='confirmed' else '')+
        'MSOP XYZ 属于原测量坐标系，不能直接当成 CAD 坐标。registration_points 通常用于稀疏建立坐标到 CAD 的刚体变换：每个有至少3个点的程序/坐标系可选4个空间分布较大、非共线且在CAD上能明确识别的参考点；若该组仅3点则选全3点。'
        '如果每个已知 XYZ 点都已在 markers 中各自独立列出，且 source_point_names 与 MSOP 点名一一对应，可将 registration_points 留空，系统会复用完整 marker 对应做刚体校验；不要重复输出整套参考点。若不能完整逐点标记，再使用稀疏 registration_points。不要把整组所有点都强行映射成视图像素。'
        +coordinate_variant_instruction+
        'Rejected geometry diagnostic 仅用于说明候选为何未获授权，不可沿用其中任何预测位置。'
        '不要把工程图上的尺寸数字直接当成 CAD 坐标；不要猜工件正反面；图集中有遮挡或特征无法唯一对应时必须弃答。'
        + location_only_instruction +
        '只写简短证据（一句话，最多30个英文词或40个汉字），不要重述点表。仅输出紧凑单行 JSON：decision（AUTO_MARK 或 ABSTAIN）、fai（整数）、markers（非空数组，每项含 view_id（+X/-X/+Y/-Y/+Z/-Z）、'
        'uv（[0到1之间的u, 0到1之间的v]）、feature、source_pages）。重试附有绿色编号候选图时，markers 和 registration_points 可改用 candidate_id（如 C7）选择精确候选；不要手抄候选 UV。source_pages 必须引用上下文中本 FAI 的 MSOP 页码。'
        '另含 evidence、confidence（low/medium/high）、coordinate_choices（每个冲突点一个含name,page,program,coordinate_system,xyz的原文选项；无冲突时为空数组）。'
        + ('已知 XYZ 点组可用于建立本 FAI 的坐标到 CAD 对应：registration_points 为稀疏参考点数组，每项字段名必须是 source_point_name（原文点名）、view_id 和 uv；按前述规则每坐标系选4个（该组不足4点时选3个），并在各自坐标系内分布充分。markers 每项必须输出 source_point_names；所有 markers 合起来必须覆盖本 FAI 已知点表中的每个不同点名，允许同一个点作为多个构造特征的共同支撑点。随后由同一坐标系的刚体变换和全点组形状/表面校验核实每个测量位置。没有至少3个非共线参考点的坐标系，可用模型选择的完整 marker 视图位置，但不得声称已通过XYZ变换。'
           if rigid_group_possible else '') +
        '模型选中的所有像素由程序逐一沿对应正交视线投射到有效 CAD 面；程序不会替模型删点、补点或另选视图/特征。'
        f"FAI 必须是 {item['number']}。"
    )
    if drawing_manifest_audit.get('status')=='confirmed':
        question+=(f"图纸位置清单来自 drawing p.{required_marker_groups[0]['source_page']}；"
            f"本 FAI 的 MSOP 来源页是 {item.get('pages',[])}。"
            'JSON markers.source_pages 仍引用 MSOP 页码，图纸页码及哈希已在 Required marker groups 中独立保存，'
            '不得把图纸页号冒充为 MSOP 页号。')
    if len(context)+len(question)>58000:
        # Keep every source point and every competing coordinate variant, but
        # collapse repeated PDF location tables and frame-wide residual arrays.
        # The full XYZ rows already carry the authoritative point definitions.
        compact_points=[]
        for point in known_source_points:
            row={key:point[key] for key in ('name','xyz','program','coordinate_system','pages')
                 if key in point}
            variants=point.get('coordinate_variants') or []
            if variants:
                row['coordinate_variants']=[{key:value[key] for key in
                    ('page','program','coordinate_system','xyz') if key in value}
                    for value in variants if isinstance(value,dict)]
            compact_points.append(row)
        diagnostic=registration_diagnostic if isinstance(registration_diagnostic,dict) else {}
        compact_frames=[]
        for frame in diagnostic.get('selected_frame_transforms',[]):
            if isinstance(frame,dict):
                compact_frames.append({key:frame[key] for key in (
                    'program','coordinate_system','rotation','translation_mm','hypothesis_id',
                    'model_confidence','p95_residual_mm','max_residual_mm','held_out_max_mm')
                    if key in frame})
        method_instruction=re.sub(r'Location Definitions[\s\S]*','',method_text,flags=re.I).strip()[:5000]
        dependency_summary=[]
        for page in detail.get('dependencies',[]):
            body=re.sub(r'Location Definitions[\s\S]*','',page.get('text',''),flags=re.I).strip()
            if body:
                dependency_summary.append({'page':page.get('page'),'text':body[:900]})
        context=(f"Drawing {project.get('drawing_no')} Rev {project.get('revision')}; FAI {item['number']}\n"
                 f"Extracted FAI fields: {extracted_fields}\n"
                 f"Required distinct marker count: {required_marker_count or 'not explicit'}\n"
                 f"Complete MSOP XYZ rows [name,xyz,program,coordinate_system,pages,all conflicting variants]: "
                 f"{json.dumps(compact_points,ensure_ascii=False,separators=(',',':'))}\n"
                 f"Competing point names: {json.dumps(ambiguous_coordinate_points,ensure_ascii=False,separators=(',',':'))}\n"
                 f"Selected frame hypotheses [display only, not registered]: "
                 f"{json.dumps(compact_frames,ensure_ascii=False,separators=(',',':'))}\n"
                 f"Rejected geometry summary: {json.dumps(_compact_registration_diagnostic(registration_diagnostic),ensure_ascii=False,separators=(',',':'))[:4000]}\n"
                 f"Rejected first-pass XYZ review: {json.dumps(_compact_registration_diagnostic(group_review_diagnostic),ensure_ascii=False,separators=(',',':'))[:2500]}\n"
                 f"MSOP method instructions: {method_instruction}\n"
                 f"Datum/context instructions: {json.dumps(dependency_summary,ensure_ascii=False,separators=(',',':'))}\n"
                 + image_description +
                 "CAD atlas panels are +Z, -Z, +X, -X, +Y, -Y, row-major. UV is normalized within the selected panel only.")
    if len(context)+len(question)>58000:
        return dict(fai=item['number'],result='abstain_or_invalid',reason='visual_evidence_exceeds_context_budget',
                    context_char_count=len(context),question_char_count=len(question),context_budget_chars=58000,
                    source_point_count=len(known_source_points),coordinate_variant_count=sum(
                        len(point.get('coordinate_variants') or []) for point in known_source_points))
    allowed_source_pages={page.get('page') for page in [*detail.get('methods',[]),*detail.get('dependencies',[])]
                          if isinstance(page.get('page'),int)}
    messages=[{'role':'user','content':question,'attachments':[{'filename':'MSOP, drawing, and active CAD orthographic atlas',
                                                                  'parsed_text':context,'images':model_images}]}]
    # Multiple point groups need iterative repair: fixing several missed rays
    # at once may leave one edge point unresolved, so allow another focused
    # local-model decision rather than terminating after only two retries.
    max_visual_attempts=6
    surface_candidates=[]
    visual_point_chunks=(_partition_visual_source_points(known_source_points,
        required_marker_groups=required_marker_groups)
        if required_marker_count==len(known_source_points) else [])
    visual_chunk_audit=[]
    for attempt in range(max_visual_attempts):
        if visual_point_chunks:
            # Large MSOP point sets are difficult for the vision model to emit
            # in one long JSON list. Keep the full drawing, atlas, and XYZ
            # corpus visible, but ask the local model for disjoint marker
            # subsets. The merge below accepts only exact one-time coverage;
            # the ordinary full-group registration and CAD geometry gates
            # still run after the merge.
            chunk_results=[];chunk_raw=[];chunk_trace=[];tokens=0;truncated=False
            ambiguous_names=set(ambiguous_coordinate_points)
            for chunk_index,chunk in enumerate(visual_point_chunks):
                chunk_names=[point['name'] for point in chunk]
                chunk_ambiguous=(sorted(ambiguous_names.intersection(chunk_names))
                                 if not preselected_coordinate_choices else [])
                update_progress=getattr(store,'update_local_model_marking',None)
                if update_progress:
                    run_progress=(store.get(project['id']).get('local_model_marking') or {}).get('progress') or {}
                    update_progress(project['id'],progress={**run_progress,
                        'current_fai':item['number'],
                        'message':(f"FAI {item['number']} 本地模型自动补点：子集 {chunk_index+1}/"
                                   f"{len(visual_point_chunks)}，{len(chunk_names)} 个源点")})
                chunk_instruction=(
                    f"\n\nChunked emission protocol: this is chunk {chunk_index+1}/{len(visual_point_chunks)} of one FAI, "
                    "not a separate FAI decision. The full measurement-group counts and full XYZ list above remain "
                    "authoritative for the final merged result. In THIS JSON return markers for exactly these source "
                    f"point names, each exactly once, and no other source point names: {json.dumps(chunk_names,ensure_ascii=False)}. "
                    "The service will merge disjoint chunks and then apply the complete FAI count, point coverage, "
                    "coordinate-frame, rigid-fit, and CAD ray-hit gates. Do not reduce or reinterpret the MSOP group. "
                    f"For coordinate variants, return exact coordinate_choices only for these names: {json.dumps(chunk_ambiguous,ensure_ascii=False)}. "
                    "registration_points may be empty in this chunk; the complete merged marker correspondences will "
                    "be checked together. Keep the usual JSON fields and cite the MSOP pages for this FAI.")
                chunk_message={'role':'user','content':messages[0]['content']+chunk_instruction,
                    'attachments':messages[0].get('attachments',[])}
                try:
                    chunk_parsed,chunk_raw_text,chunk_truncated,chunk_tokens,_chunk_aliases,repair_audit=(
                        _request_visual_marker_chunk(model,chunk_message,fai=item['number'],
                            expected_names=chunk_names,source_points=known_source_points,
                            allowed_source_pages=allowed_source_pages,surface_candidates=surface_candidates,
                            temperature=min(.35,.12*attempt)))
                except Exception as exc:
                    chunk_results.append(None)
                    chunk_raw.append({'chunk':chunk_index+1,'error':f'{type(exc).__name__}: {exc}'[:300]})
                    chunk_trace.append({'chunk':chunk_index+1,'source_point_names':chunk_names,
                        'error':f'{type(exc).__name__}: {exc}'[:1000]})
                    continue
                tokens+=chunk_tokens;truncated=truncated or chunk_truncated
                chunk_raw.append(chunk_raw_text[:1200])
                chunk_results.append(chunk_parsed if not chunk_truncated else None)
                chunk_trace.append({'chunk':chunk_index+1,'source_point_names':chunk_names,
                    'truncated':bool(chunk_truncated),'generated_tokens':chunk_tokens,
                    'source_name_aliases':_chunk_aliases,'raw_output':chunk_raw_text[:6000],
                    'parsed_output':chunk_parsed,'subset_repair_attempts':repair_audit})
            parsed_chunks=_merge_visual_marker_chunks(
                chunk_results,[[point['name'] for point in chunk] for chunk in visual_point_chunks],
                fai=item['number'])
            visual_chunk_audit.append({'attempt':attempt+1,'chunk_count':len(visual_point_chunks),
                'chunk_failures':parsed_chunks.get('_chunk_failures',[]),'chunks':chunk_trace})
            audit_path=store._dir(project['id'])/'automatic-learning'/(
                f"fai-chunk-decisions-{mesh['source_sha256'][:16]}-{item['number']}.json")
            audit_path.parent.mkdir(parents=True,exist_ok=True)
            audit_temp=audit_path.with_suffix('.json.tmp')
            audit_temp.write_text(json.dumps({'fai':item['number'],'attempts':visual_chunk_audit},
                ensure_ascii=False,indent=2),encoding='utf-8')
            audit_temp.replace(audit_path)
            parsed_chunks['chunk_aliases']=[alias for raw_result in chunk_results
                if isinstance(raw_result,dict)
                for alias in _normalize_model_source_aliases(raw_result,known_source_points,allowed_source_pages)[1]]
            raw=json.dumps(parsed_chunks,ensure_ascii=False,separators=(',',':'))
        else:
            try:
                raw,truncated,tokens=model._generate_once(messages,max_context_chars=_complete_visual_context_budget(messages),
                    max_new_tokens=4096,max_images=8,
                    temperature=min(.35,.12*attempt))
            except Exception as exc:
                return dict(fai=item['number'],result='abstain_or_invalid',reason='local_vision_inference_failed',
                            error=f'{type(exc).__name__}: {exc}'[:500])
        raw_parsed=_object(raw) or {}
        raw_parsed,source_name_aliases=_normalize_model_source_aliases(
            raw_parsed,known_source_points,allowed_source_pages)
        if preselected_coordinate_choices:
            selected_by_name={row['name']:row for row in preselected_coordinate_choices}
            reported=raw_parsed.get('coordinate_choices') or []
            conflicts=[]
            if isinstance(reported,list):
                for row in reported:
                    if not isinstance(row,dict):
                        conflicts.append({'reason':'model_coordinate_choice_not_object'});continue
                    name=row.get('name');selected=selected_by_name.get(name)
                    if not selected or any(row.get(key)!=selected.get(key)
                        for key in ('page','program','coordinate_system','xyz')):
                        conflicts.append({'name':name,'reason':'visual_stage_disagrees_with_local_variant_selection'})
            elif reported:
                conflicts.append({'reason':'model_coordinate_choices_not_list'})
            raw_parsed['coordinate_choices']=preselected_coordinate_choices
            if conflicts:
                raw_parsed['_coordinate_choice_conflicts']=conflicts
        candidate_selection_audit=_surface_candidate_selection_audit(raw_parsed,surface_candidates,
            require_marker_ids=drawing_manifest_audit.get('status')=='confirmed')
        candidate_selection_valid=not candidate_selection_audit
        parsed=_resolve_surface_candidate_ids(raw_parsed,surface_candidates)
        decision=re.sub(r'[^A-Z_]','',str(parsed.get('decision') or '').upper())
        markers=parsed.get('markers')
        coordinate_choices_valid,coordinate_choices=_validate_coordinate_variant_choices(
            known_source_points,parsed.get('coordinate_choices'))
        validation_failures=_visual_marker_validation_issues(parsed,fai=item['number'],
            required_marker_count=required_marker_count,known_source_points=known_source_points,
            rigid_group_possible=rigid_group_possible,
            candidate_selection_valid=candidate_selection_valid,
            coordinate_choices_valid=coordinate_choices_valid,truncated=truncated,
            candidate_selection_audit=candidate_selection_audit)
        if isinstance(markers,list):
            bad_pages=[{'marker_index':index,'source_pages':marker.get('source_pages')}
                for index,marker in enumerate(markers) if isinstance(marker,dict)
                and (not isinstance(marker.get('source_pages'),list) or not marker['source_pages']
                    or any(type(page) is not int or page not in allowed_source_pages
                           for page in marker['source_pages']))]
            if bad_pages:
                validation_failures.append({'code':'marker_MSOP_source_pages_invalid',
                    'allowed_MSOP_pages':sorted(allowed_source_pages),'details':bad_pages[:30],
                    'instruction':'Keep drawing provenance in the drawing manifest; marker source_pages must cite the MSOP method/context.'})
        if parsed.get('_coordinate_choice_conflicts'):
            validation_failures.append({'code':'local_model_coordinate_choice_disagreement',
                                        'details':parsed['_coordinate_choice_conflicts'][:10]})
        if parsed.get('_chunk_failures'):
            validation_failures.append({'code':'chunked_source_point_emission_incomplete',
                                        'details':parsed['_chunk_failures'][:20]})
        valid=(not validation_failures
               and isinstance(markers,list) and 1<=len(markers)<=2000
               and all(isinstance(marker,dict) and marker.get('view_id') in ('+X','-X','+Y','-Y','+Z','-Z')
                       and isinstance(marker.get('uv'),list) and len(marker['uv'])==2
                       and all(isinstance(value,(int,float)) and not isinstance(value,bool)
                               and 0<=value<=1 for value in marker['uv'])
                       and isinstance(marker.get('feature'),str) and bool(marker['feature'].strip())
                       and isinstance(marker.get('source_pages'),list) and bool(marker['source_pages'])
                       and all(type(page) is int and page in allowed_source_pages
                               for page in marker['source_pages'])
                       for marker in markers)
               and _confidence(parsed.get('confidence')) is not None
               and bool(parsed.get('evidence')))
        registration_points=parsed.get('registration_points') if rigid_group_possible else []
        reference_audit={}
        if valid and rigid_group_possible:
            # Vision models sometimes emit the point identifier under the
            # generic JSON key "name". It is safe to normalize that alias
            # only when it exactly matches one unique MSOP source point; no
            # coordinate or point association is inferred by this adapter.
            expected={point.get('name') for point in known_source_points}
            if isinstance(registration_points,list):
                registration_points=[
                    ({**row,'source_point_name':row['name']}
                     if isinstance(row,dict) and 'source_point_name' not in row
                     and row.get('name') in expected else row)
                    for row in registration_points]
            valid,reference_audit=_registration_reference_selection_valid(
                known_source_points,registration_points,coordinate_choices,markers)
            if not valid:
                complete_references=_complete_marker_registration_references(
                    known_source_points,markers)
                if complete_references:
                    complete_valid,complete_audit=_registration_reference_selection_valid(
                        known_source_points,complete_references,coordinate_choices,markers)
                    if complete_valid:
                        registration_points=complete_references
                        valid=True
                        reference_audit={**complete_audit,
                                         'reference_source':'complete_model_selected_marker_correspondences'}
            if not valid:
                validation_failures.append({'code':'registration_reference_audit_failed',
                    'details':reference_audit})
        if not valid:
            if attempt<max_visual_attempts-1:
                if decision in ('ABSTAIN','REJECT','CANNOT_DETERMINE'):
                    correction=(
                        ' 上次选择了弃答，但本任务就是由你在本轮自动建立 FAI 级候选，不需要任何外部授权。'
                        + (' 分块输出的合并审计失败：'+json.dumps(parsed.get('_chunk_failures'),ensure_ascii=False)+
                           '。请按本次子集点名重新完成每个分块，不要弃答完整点组。'
                           if parsed.get('_chunk_failures') else '')+
                        '请重新检查 MSOP 特征说明、原文 XYZ 点组、2D视图和 CAD 六视图：若点组可对应到 CAD，即使置信度只有 low，也输出完整 AUTO_MARK 候选；'
                        '有 XYZ 时可以给每坐标系4个（该组不足4点时3个）分布充分的 registration_points；若每个已知 XYZ 点都已有各自独立 marker 且 source_point_names 一一对应，可省略 registration_points，系统会复用这些完整 marker 对应做刚体校验。所有 marker 的 source_point_names 必须完整覆盖已知点名；没有可用 XYZ 时，选择最可能的完整特征位置组。'
                        '优先使用附件中的绿色 C 编号 CAD 命中点；不同XYZ点不要重复使用同一个C编号。若上一轮的视角把不同深度压在同一UV，保留对应UV并用已通过拟合的XYZ变换分别落到CAD表面，或改选能分开深度的视图。'
                        '只有图纸证据完全没有可辨认目标特征时才返回 ABSTAIN。')
                else:
                    count_failure=next((row for row in validation_failures
                        if row.get('code')=='required_marker_count_mismatch'),None)
                    coverage_failure=next((row for row in validation_failures
                        if row.get('code')=='source_point_coverage_incomplete'),None)
                    reference_failure=next((row for row in validation_failures
                        if row.get('code')=='registration_reference_audit_failed'),None)
                    candidate_failure=next((row for row in validation_failures
                        if row.get('code')=='candidate_id_selection_invalid'),None)
                    correction=(
                    ' 上次没有为每个冲突点逐一选择一个完全匹配的 coordinate_variants 项。请补全 coordinate_choices，'
                    '每项必须原样复制 name、page、program、coordinate_system、xyz；不得改写、平均或省略。'
                    if not coordinate_choices_valid else
                    f" 上次 markers 只有 {count_failure['actual']} 项，原文要求 {count_failure['expected']} 项；"
                    f"请按每条 MSOP 测量组分别列齐，分组要求为 {json.dumps(required_marker_groups,ensure_ascii=False)}。"
                    if count_failure else
                    f" 上次 marker 点名覆盖不全，缺少 {coverage_failure['missing_count']} 个源点："
                    f"{json.dumps(coverage_failure['missing_names'],ensure_ascii=False)}。请让所有直接测量点名恰好归属到完整 marker 组。"
                    if coverage_failure else
                    ' 上次稀疏坐标配准参考未通过几何结构审计：'
                    f"{json.dumps(reference_failure['details'],ensure_ascii=False)}。请修正 reference 点身份、坐标系归属和非共线分布；"
                    'registration_points 只需为每个可解坐标系选 3-4 个有效且分布充分的点，markers 仍必须覆盖全部测量点。'
                    if reference_failure else
                    ' 上次候选编号与源点不匹配。下面逐项给出源点对应的有效编号，请从该源点自己的清单里选；'
                    '不要把同一编号跨用到没有列出的点，也不要编造编号。未列入候选图册的点才可使用 view_id 和归一化 uv 直接选点。'
                    f"编号审计：{json.dumps(candidate_failure.get('details',[]),ensure_ascii=False)}。"
                    if candidate_failure else
                    ' 上次选择了 ABSTAIN 或没有返回完整 markers。请检查逐条 MSOP 测量组 manifest；'
                    '只要图纸/MSOP 有可辨认的实体位置，就给出满足每组 declared_count 的完整候选；'
                    '只有证据确实无法定位时才弃答。'
                    if any(row.get('code')=='decision_not_auto_mark' for row in validation_failures) else
                    ' 上次 JSON 未满足完整标记组或字段约束。请重新核对 FAI 编号、每个必需位置和允许视图，'
                    f"按以下独立校验失败原因修正：{json.dumps(validation_failures,ensure_ascii=False)}。"
                    '严格按 JSON schema 输出；不能完整判断时返回 ABSTAIN。')
                messages=[{'role':'user','content':question+correction,
                           'attachments':messages[0]['attachments']}]
                continue
            return dict(fai=item['number'],result='abstain_or_invalid',reason='model_visual_decision_invalid',
                        truncated=truncated,validation_failures=validation_failures,
                        raw_output=raw[:3000],visual_chunk_audit=visual_chunk_audit[-1:] if visual_chunk_audit else None)
        hits=[];rigid_fit=None
        selected_by_name={row.get('name'):row for row in coordinate_choices}
        resolved_sources={}
        if rigid_group_possible:
            for point in known_source_points:
                choice=selected_by_name.get(point['name']) if point.get('requires_variant_choice') else None
                xyz=(choice.get('xyz') if choice else point.get('xyz'))
                program=(choice.get('program') if choice else point.get('program'))
                coordinate_system=(choice.get('coordinate_system') if choice else point.get('coordinate_system'))
                page=(choice.get('page') if choice else (point.get('pages') or [None])[0])
                if (not isinstance(xyz,list) or len(xyz)!=3 or not program or not coordinate_system
                        or any(isinstance(value,bool) or not isinstance(value,(int,float)) for value in xyz)):
                    resolved_sources={};break
                resolved_sources[point['name']]=dict(name=point['name'],xyz=xyz,program=program,
                    coordinate_system=coordinate_system,page=page)
            if len(resolved_sources)!=len(known_source_points):
                resolved_sources={}
        registration_hits=[];rigid_fits={};failed_registration_points=[];failed_registration_rows=[]
        surface_index=None;surface_face_ids=None
        if rigid_group_possible and resolved_sources:
            surface_triangles,surface_face_ids,surface_index=cached_surface_geometry(mesh)
            grouped={};expected_by_scope={}
            for source in resolved_sources.values():
                scope=(source['program'],source['coordinate_system'])
                expected_by_scope[scope]=expected_by_scope.get(scope,0)+1
            for row in registration_points:
                hit=ray_hit_view(mesh,row['view_id'],float(row['uv'][0]),float(row['uv'][1]))
                if not hit:
                    failed_registration_points.append(row.get('source_point_name'))
                    failed_registration_rows.append(dict(
                        feature=row.get('source_point_name'),
                        source_point_names=[row.get('source_point_name')],
                        view_id=row.get('view_id'),uv=row.get('uv'),
                        candidate_role='registration_reference'))
                    continue
                registration_hits.append((row,hit))
                source=resolved_sources[row['source_point_name']]
                scope=(source['program'],source['coordinate_system'])
                grouped.setdefault(scope,([],[]))
                grouped[scope][0].append(source['xyz']);grouped[scope][1].append(hit['position'])
            for scope,(source_xyz,cad_xyz) in grouped.items():
                # Sparse references establish the frame. Completeness is
                # checked independently: every MSOP point must belong to one
                # and only one marker group, and every marker must pass CAD
                # surface projection and group-shape validation.
                if len(source_xyz)>=3:
                    fit=_rigid_correspondence(source_xyz,cad_xyz,max_residual_mm=.75)
                    if fit:rigid_fits[scope]=fit
        fit_required_scopes=_xyz_rigid_fit_required_scopes(resolved_sources)
        failure_diagnostic=None;missed_markers=[]
        collapsed_marker_groups=_collapsed_source_marker_groups(markers,resolved_sources)
        if collapsed_marker_groups:
            _view_id,_u,_v,group=max(collapsed_marker_groups,key=lambda row:len(row[3]))
            failure_diagnostic={'reason':'distinct_source_points_collapsed_to_same_view_uv',
                'view_id':_view_id,'uv':[_u,_v],
                'source_point_names':list(dict.fromkeys(name for marker in group
                    for name in marker.get('source_point_names',[]))),
                'missed_marker_points':[dict(feature=marker.get('feature'),
                    source_point_names=marker.get('source_point_names'),
                    view_id=marker.get('view_id'),uv=marker.get('uv')) for marker in group]}
        for marker in markers:
            hit=ray_hit_view(mesh,marker['view_id'],float(marker['uv'][0]),float(marker['uv'][1]))
            if not hit:
                missed_markers.append(marker)
                if failure_diagnostic is None:
                    failure_diagnostic={'reason':'marker_view_ray_missed_CAD',
                        'feature':marker.get('feature'),'view_id':marker.get('view_id'),'uv':marker.get('uv')}
                continue
            hits.append((marker,dict(face_id=hit['face_id'],position=hit['position'],label_offset=[32,-26])))
        if missed_markers and failure_diagnostic:
            failure_diagnostic['missed_marker_points']=[dict(
                feature=marker.get('feature'),source_point_names=marker.get('source_point_names'),
                view_id=marker.get('view_id'),uv=marker.get('uv')) for marker in missed_markers]
        collapsed_groups_have_rigid_fit=bool(collapsed_marker_groups)
        collapsed_marker_ids={id(marker) for _view_id,_u,_v,group in collapsed_marker_groups
                              for marker in group}
        for _view_id,_u,_v,group in collapsed_marker_groups:
            for marker in group:
                name=(marker.get('source_point_names') or [None])[0]
                source=resolved_sources.get(name) if name else None
                scope=(source.get('program'),source.get('coordinate_system')) if source else None
                if scope in fit_required_scopes and scope not in rigid_fits:
                    collapsed_groups_have_rigid_fit=False
        if collapsed_marker_groups and not collapsed_groups_have_rigid_fit:
            # Without an independently fitted XYZ frame, one ray cannot
            # authorize several distinct source locations.
            hits=[]
        anchors=[];placement_modes=['model_view_ray']*len(markers)
        if len(hits)==len(markers) and not collapsed_groups_have_rigid_fit:
            for index,(marker,anchor) in enumerate(hits):
                if any(np.linalg.norm(np.asarray(anchor['position'])-np.asarray(previous['position']))<1e-5
                       for previous in anchors[:index]):
                    failure_diagnostic={'reason':'duplicate_marker_positions',
                        'feature':marker.get('feature'),'view_id':marker.get('view_id'),
                        'uv':marker.get('uv')}
                    hits=[]
                    break
                anchors.append(anchor)
        if len(hits)==len(markers):
            if rigid_group_possible:
                adjusted=[];fit_summaries=[];placement_modes=[]
                for marker,(_model_marker,visual_anchor) in zip(markers,hits):
                    selected_names=marker['source_point_names']
                    scopes={(resolved_sources[name]['program'],resolved_sources[name]['coordinate_system'])
                            for name in selected_names} if resolved_sources else set()
                    fit=rigid_fits.get(next(iter(scopes))) if len(scopes)==1 else None
                    if not fit:
                        required_scopes=_missing_required_xyz_fits(
                            scopes,fit_required_scopes,rigid_fits)
                        if required_scopes:
                            failure_diagnostic={'reason':'required_xyz_rigid_fit_not_established',
                                'feature':marker.get('feature'),'source_point_names':selected_names,
                                'required_scopes':[list(scope) for scope in required_scopes],
                                'registration_point_count':len(registration_hits),
                                'missed_registration_points':failed_registration_points,
                                'fit_required_point_counts':{str(scope):sum(
                                    (row.get('program'),row.get('coordinate_system'))==scope
                                    for row in resolved_sources.values()) for scope in required_scopes}}
                            adjusted=[]
                            break
                        adjusted.append(visual_anchor)
                        placement_modes.append('model_view_ray')
                        continue
                    source_xyz=np.asarray([resolved_sources[name]['xyz'] for name in selected_names],float)
                    feature_xyz=source_xyz.mean(axis=0)
                    expected_cad=fit['rotation']@feature_xyz+fit['translation']
                    is_collapsed=id(marker) in collapsed_marker_ids
                    if is_collapsed:
                        expected_uv,scale=project_point_to_view_uv(mesh,marker['view_id'],expected_cad)
                        projected_error_mm=(float(np.linalg.norm(np.asarray(expected_uv)-
                            np.asarray(marker['uv'])))*(480-44)/scale)
                        if projected_error_mm>.75:
                            failure_diagnostic={'reason':'rigid_transform_disagrees_with_marker_view_projection',
                                'feature':marker.get('feature'),'source_point_names':selected_names,
                                'projected_error_mm':projected_error_mm,'gate_mm':.75,
                                'registration_point_count':len(registration_hits),
                                'missed_registration_points':failed_registration_points}
                            adjusted=[];break
                    elif not candidate_selection_valid:
                        correction=(
                            ' 上次 candidate_id 与源点对应关系不正确。请按以下逐点编号审计更正，只能为每个源点选择其 valid_candidate_ids 中的编号；'
                            '程序会将编号解析为精确 CAD 射线命中。不得给这些候选点手写或估算 UV。'
                            f"编号审计：{json.dumps(candidate_selection_audit[:20],ensure_ascii=False)}。"
                            '未被候选清单覆盖的其他点可保留已有 UV。完整重列标记和配准参考。')
                    else:
                        visual_xyz=np.asarray(visual_anchor['position'],float)
                        marker_displacement=float(np.linalg.norm(expected_cad-visual_xyz))
                        if marker_displacement>.75:
                            failure_diagnostic={'reason':'rigid_transform_disagrees_with_marker_view',
                                'feature':marker.get('feature'),'source_point_names':selected_names,
                                'displacement_mm':marker_displacement,'gate_mm':.75,
                                'registration_point_count':len(registration_hits),
                                'missed_registration_points':failed_registration_points}
                            adjusted=[];break
                    hit=surface_index.nearest(expected_cad)
                    if hit['distance_mm']>.35:
                        failure_diagnostic={'reason':'transformed_marker_off_CAD_surface',
                            'feature':marker.get('feature'),'source_point_names':selected_names,
                            'surface_distance_mm':float(hit['distance_mm']),'gate_mm':.35}
                        adjusted=[];break
                    adjusted.append(dict(face_id=str(surface_face_ids[int(hit['triangle_index'])]),
                        position=np.asarray(hit['point'],float).tolist(),label_offset=[32,-26]))
                    placement_modes.append('rigid_xyz_transform')
                    fit_summaries.append(fit)
                if len(adjusted)==len(markers):
                    placed=np.asarray([anchor['position'] for anchor in adjusted])
                    if len(fit_summaries)>0:
                        # Compare only anchors obtained from the same rigid frame.
                        for scope,fit in rigid_fits.items():
                            indices=[i for i,marker in enumerate(markers) if len({
                                (resolved_sources[name]['program'],resolved_sources[name]['coordinate_system'])
                                for name in marker['source_point_names']})==1 and
                                (resolved_sources[marker['source_point_names'][0]]['program'],
                                 resolved_sources[marker['source_point_names'][0]]['coordinate_system'])==scope]
                            if len(indices)>1:
                                predicted=np.asarray([fit['rotation']@np.mean(
                                    [resolved_sources[name]['xyz'] for name in markers[i]['source_point_names']],axis=0)
                                    +fit['translation'] for i in indices])
                                group_placed=placed[indices]
                                shape_error=np.linalg.norm((group_placed[:,None,:]-group_placed[None,:,:])-
                                    (predicted[:,None,:]-predicted[None,:,:]),axis=2)
                                if float(shape_error.max())>.7:
                                    failure_diagnostic={'reason':'rigid_marker_group_shape_drift',
                                        'scope':scope,'shape_error_mm':float(shape_error.max()),'gate_mm':.7}
                                    adjusted=[];break
                                fit['max_surface_residual_mm']=max(float(np.linalg.norm(group_placed[j]-predicted[j]))
                                    for j in range(len(indices)))
                                fit['max_pairwise_shape_error_mm']=float(shape_error.max())
                    anchors=adjusted if len(adjusted)==len(markers) else []
            try:
                if not rigid_group_possible:
                    anchors=validate_anchors(mesh,anchors)
                else:
                    anchors=validate_anchors(mesh,anchors) if anchors else []
            except ValueError as exc:
                failure_diagnostic={'reason':'anchor_validation_failed','detail':str(exc)[:250]}
                anchors=[]
        locations=[]
        if anchors:
            if len(placement_modes)!=len(anchors):
                failure_diagnostic={'reason':'placement_method_count_mismatch',
                    'anchor_count':len(anchors),'placement_method_count':len(placement_modes),
                    'marker_count':len(markers),'rigid_group_possible':rigid_group_possible,
                    'resolved_source_count':len(resolved_sources),
                    'rigid_fit_scopes':[list(scope) for scope in rigid_fits],
                    'prior_geometry_diagnostic':failure_diagnostic}
                anchors=[]
        if anchors:
            locations=[dict(feature=marker['feature'],source_point_names=marker.get('source_point_names',[]),view_id=marker['view_id'],uv=marker['uv'],
                            placement_method=placement_modes[index],
                            face_id=anchor['face_id'],position=anchor['position'])
                       for index,((marker,_),anchor) in enumerate(zip(hits,anchors))]
            if (len(locations)!=len(anchors) or len(anchors)!=len(markers)
                    or required_marker_count and len(anchors)!=required_marker_count):
                failure_diagnostic={'reason':'accepted_anchor_group_count_mismatch',
                    'model_marker_count':len(markers),'validated_anchor_count':len(anchors),
                    'location_audit_count':len(locations),'required_marker_count':required_marker_count}
                anchors=[]
        if anchors and drawing_manifest_audit.get('status')=='confirmed':
            spacing_failure=_drawing_spacing_diagnostic(anchors,required_marker_groups[0])
            if spacing_failure:
                failure_diagnostic=spacing_failure
                selected_spacing,spacing_layout_audit=_review_even_spacing_options(
                    store,project,item,model,mesh,anchors,markers,required_marker_groups[0],images)
                if selected_spacing:
                    anchors=validate_anchors(mesh,selected_spacing['anchors'])
                    view_id=markers[0]['view_id'];feature=markers[0]['feature'];pages=markers[0]['source_pages']
                    markers=[{'view_id':view_id,'uv':project_point_to_view_uv(mesh,view_id,anchor['position'])[0],
                        'feature':f'{feature} #{index+1}','source_pages':pages}
                        for index,anchor in enumerate(anchors)]
                    hits=list(zip(markers,anchors))
                    placement_modes=['local_model_selected_spacing_hypothesis']*len(anchors)
                    locations=[{'feature':marker['feature'],'view_id':view_id,'uv':marker['uv'],
                        'placement_method':placement_modes[index],'face_id':anchor['face_id'],'position':anchor['position']}
                        for index,(marker,anchor) in enumerate(zip(markers,anchors))]
                    drawing_manifest_audit['spacing_layout_review']=spacing_layout_audit
                    failure_diagnostic=None
                else:
                    failure_diagnostic={**spacing_failure,'local_model_spacing_review':spacing_layout_audit}
                    anchors=[]
        if anchors and allowed_part_faces:
            outside=[index+1 for index,anchor in enumerate(anchors) if anchor['face_id'] not in allowed_part_faces]
            if outside:
                anchors=[]
                failure_diagnostic={'reason':'marker_outside_local_model_selected_parts',
                    'marker_ids':outside,'selected_parts':part_context.get('part_ids'),
                    'evidence':part_context.get('evidence')}
        entity_review={'status':'not_required'}
        if anchors:
            entity_review=_review_visual_entity_group(store,project,item,detail,model,mesh,
                anchors,locations,images,source_manifest=source_manifest_audit,
                drawing_manifest=drawing_manifest_audit)
            if entity_review.get('status')!='confirmed':
                anchors=[]
                failure_diagnostic={'reason':'local_entity_group_review_not_confirmed',
                    'local_model_decision':entity_review.get('decision'),
                    'review_status':entity_review.get('status'),'receipt':entity_review.get('receipt'),
                    'receipt_sha256':entity_review.get('receipt_sha256')}
        if anchors:
            model_confidence=_confidence(parsed.get('confidence'))
            confidence=('low' if entity_review.get('status')=='confirmed' else model_confidence)
            fitted=[fit for fit in rigid_fits.values() if 'max_surface_residual_mm' in fit]
            rigid_note=(f"；{len(fitted)} 个 MSOP 坐标系通过独立刚体变换约束，最大对应残差 "
                        f"{max(fit['max_residual_mm'] for fit in fitted):.4f} mm、最大 CAD 表面投影残差 "
                        f"{max(fit['max_surface_residual_mm'] for fit in fitted):.4f} mm、最大点间几何改变量 "
                        f"{max(fit['max_pairwise_shape_error_mm'] for fit in fitted):.4f} mm"
                        if fitted else '')
            note=(f"本地视觉模型自动定位回退，共 {len(anchors)} 个位置：{json.dumps(locations,ensure_ascii=False)}{rigid_note}；"
                  f"本地模型选定的原文坐标版本：{json.dumps(coordinate_choices,ensure_ascii=False)}；"
                  f"证据：{parsed['evidence']}；来源页：{parsed.get('source_pages') or item.get('pages')}。"
                  f'模型自评置信度：{model_confidence}；显示置信度：{confidence}。'
                  '模型所选正交视线用于核对特征位置；已注册坐标系的 marker 由源 XYZ 几何中心变换并贴合 CAD 表面，其他 marker 使用模型视线交点。'
                  f'稀疏配准参考审计：{json.dumps(reference_audit,ensure_ascii=False)}。'
                  f'直接测量点清单审计：{json.dumps(source_manifest_audit,ensure_ascii=False)}。'
                  f'图纸位置数量清单审计：{json.dumps(drawing_manifest_audit,ensure_ascii=False)}。'
                  f'独立实体复核：{json.dumps(entity_review,ensure_ascii=False)}。'
                  '这些位置是显示标记，不代表计量验收。')
            store.annotate(project['id'],item['number'],anchors,note,'auto_marked',
                           mesh['source_sha256'],item['document_id'],'local_model_candidate',confidence,
                           feature_verification=entity_review)
            return dict(fai=item['number'],result='accepted',confidence=confidence,
                        face_ids=list(dict.fromkeys(anchor['face_id'] for anchor in anchors)),
                        anchor_count=len(anchors),decision=decision,locations=locations,
                        source_pages=parsed.get('source_pages'),coordinate_choices=coordinate_choices,
                        registration_reference_audit=reference_audit,
                        source_name_aliases=source_name_aliases,
                        source_manifest_audit=source_manifest_audit,entity_review=entity_review,
                        evidence=parsed.get('evidence'),
                        generated_tokens=tokens,visual_fallback=True)
        if attempt<max_visual_attempts-1:
            diagnostic=failure_diagnostic or {'reason':'complete_marker_group_not_validated',
                'registration_point_count':len(registration_hits),
                'rigid_fit_scopes':{str(scope):{'max_residual_mm':fit.get('max_residual_mm')}
                                    for scope,fit in rigid_fits.items()}}
            retry_images=model_images
            surface_candidates=[]
            try:
                focused_path=store._dir(project['id'])/'automatic-learning'/(
                    f"cad-focused-retry-{mesh['source_sha256'][:16]}-{item['number']}-{attempt+1}.png")
                failed_view=diagnostic.get('view_id')
                failed_uv=diagnostic.get('uv')
                if diagnostic.get('reason') in ('marker_view_ray_missed_CAD',
                        'distinct_source_points_collapsed_to_same_view_uv',
                        'duplicate_marker_positions','accepted_anchor_group_count_mismatch') and failed_view and failed_uv:
                    # Search around every failed marker in the group. A single
                    # failed pixel is rarely enough to reconstruct a complete
                    # feature group, and retrying only that point encouraged
                    # the model to keep the other off-surface UVs unchanged.
                    missed_points=diagnostic.get('missed_marker_points') or [diagnostic]
                    grouped_misses={}
                    for missed in missed_points:
                        candidate_view=missed.get('view_id')
                        candidate_uv=missed.get('uv')
                        if not candidate_view or not candidate_uv:
                            continue
                        key=(candidate_view,tuple(candidate_uv))
                        grouped_misses.setdefault(key,[]).append(missed)
                    # Registration anchors establish the CAD/MSOP coordinate
                    # transform. Repairing only measurement markers cannot
                    # recover an XYZ-based placement if these rays still miss.
                    for missed in failed_registration_rows:
                        candidate_view=missed.get('view_id')
                        candidate_uv=missed.get('uv')
                        if not candidate_view or not candidate_uv:
                            continue
                        key=(candidate_view,tuple(candidate_uv))
                        grouped_misses.setdefault(key,[]).append(missed)
                    triangle_data=cached_surface_geometry(mesh)[:2]
                    for (candidate_view,candidate_uv),miss_group in grouped_misses.items():
                        source_names=list(dict.fromkeys(name for missed in miss_group
                            for name in (missed.get('source_point_names') or [])))
                        features=list(dict.fromkeys(missed.get('feature') for missed in miss_group
                            if missed.get('feature')))
                        roles=list(dict.fromkeys(missed.get('candidate_role','measurement_marker')
                            for missed in miss_group))
                        choices=nearby_surface_view_candidates(mesh,candidate_view,candidate_uv,
                            radii=(0.003,0.006,0.02,0.05,0.10),
                            max_candidates=min(12,max(3,len(miss_group))),
                            triangle_data=triangle_data)
                        feature_choices=feature_view_candidates(mesh,candidate_view,candidate_uv,
                            max_candidates=min(8,max(4,len(miss_group))),triangle_data=triangle_data)
                        for choice in feature_choices:
                            if not any(np.linalg.norm(np.asarray(choice['position'])-
                                    np.asarray(old['position']))<.1 for old in choices):
                                choices.append(choice)
                        for choice in choices:
                            choice['id']=len(surface_candidates)+1
                            choice['feature']='/'.join(features[:3])
                            choice['features']=features
                            choice['source_point_names']=source_names
                            choice['candidate_roles']=roles
                            choice['original_uv']=candidate_uv
                            surface_candidates.append(choice)
                    failed_names=list(dict.fromkeys(
                        name for row in [*missed_points,*failed_registration_rows]
                        for name in (row.get('source_point_names') or [])))
                    collapsed_depth=(diagnostic.get('reason')==
                        'distinct_source_points_collapsed_to_same_view_uv')
                    has_global_pool=any(row.get('candidate_role')=='global_surface_search'
                                        for row in surface_candidates)
                    if (not has_global_pool and
                            (not surface_candidates or collapsed_depth or
                             attempt>=2 and diagnostic.get('reason')=='marker_view_ray_missed_CAD')):
                        # A local neighborhood cannot resolve points that the
                        # selected orthographic view collapses onto one pixel.
                        # Search every view for exact CAD ray hits and let the
                        # local model choose semantic correspondences. Try
                        # actual nearby thin-feature probes first instead of
                        # immediately drowning them in coarse corner hits; all
                        # rigid-fit and complete-group gates still apply.
                        global_candidates=distributed_surface_view_candidates(
                            mesh,grid_size=14,max_candidates_per_view=8,
                            triangle_data=triangle_data)
                        for choice in global_candidates:
                            choice['id']=len(surface_candidates)+1
                            choice['source_point_names']=failed_names
                            choice['candidate_roles']=['global_surface_search',
                                'measurement_marker','registration_reference']
                            choice['feature']='global CAD ray-hit recovery option'
                            surface_candidates.append(choice)
                failed_marker=next((marker for marker in markers
                    if marker.get('view_id')==failed_view and marker.get('uv')==failed_uv),None)
                focus_markers=[failed_marker] if failed_marker else markers
                focus_registration=[row for row in registration_points
                    if not failed_view or row.get('view_id')==failed_view]
                focused=render_focused_view_retry(atlas_path,focused_path,focus_markers,
                    focus_registration if rigid_group_possible else [],
                    surface_candidates=surface_candidates)
                if focused:
                    # Retain the complete six-view atlas so the model can
                    # change orientation if its previous view was wrong; the
                    # focused panel adds readable CAD detail and nearby
                    # verified ray-hit choices.
                    retry_images=list(dict.fromkeys([*images,str(atlas_path),str(focused)]))
                if surface_candidates and any(
                        row.get('candidate_role')=='global_surface_search'
                        for row in surface_candidates):
                    candidate_atlas=store._dir(project['id'])/'automatic-learning'/(
                        f"cad-candidate-atlas-{mesh['source_sha256'][:16]}-{item['number']}-{attempt+1}.png")
                    render_surface_candidate_atlas(atlas_path,candidate_atlas,surface_candidates)
                    retry_images=list(dict.fromkeys([*images,str(candidate_atlas),str(atlas_path),
                                  *([str(focused)] if focused else [])]))
            except (OSError,ValueError,TypeError):
                pass
            surface_text=(f" Verified CAD ray-hit choices; source_point_names.group refers exactly to the named list in source_point_groups (no points or choices omitted). Failed measurement markers and failed registration references (local choices or compact whole-view recovery pool; search hints only; choose by MSOP feature semantics; replace every failed point, do not fix only one): "
                          f"{json.dumps(_candidate_prompt_payload(surface_candidates),ensure_ascii=False,separators=(',',':'))}."
                          if surface_candidates else
                          (' No nearby ray-hit candidate exists around the failed UV; do not reuse this view/pixel. '
                           'Select a different of the six views shown in the full atlas, then re-map the feature.'
                           if diagnostic.get('reason')=='marker_view_ray_missed_CAD' else ''))
            rigid_fit_instruction=(
                '本次失败原因是 MSOP 坐标组有足够非共线 XYZ，却没有通过源坐标到 CAD 的刚体拟合。必须重新选择同一坐标系内至少4个空间分布充分、非共线且各自落在正确 CAD 特征上的 registration_points；'
                '不得改用纯视线落点绕过拟合。拟合残差门限为0.75 mm，未通过就继续修正或弃答。'
                if diagnostic.get('reason')=='required_xyz_rigid_fit_not_established' else '')
            messages=[{'role':'user','content':question+
                       ' 上次返回的位置组未通过几何或实体对应复核。诊断：'+json.dumps(diagnostic,ensure_ascii=False)+
                       '。请修正模型选取的视图和像素位置，并完整重列 markers 与稀疏 registration_points；'
                          '不能删点或复用失败的对应。每个位置必须落在 CAD 实体上，完整组必须保持 MSOP 点间几何关系；不同 XYZ 的单点 marker 必须使用不同 UV 和不同 CAD 交点，不能把同一个 C 编号重复分配给多个点。若某视图把深度方向压扁或多个点重叠，改用侧视图。'
                       +surface_text+
                       ('本 FAI 的图纸位置数量清单已确认。若上述清单提供有效 CAD 候选，markers 每项必须用不同的 candidate_id（如 C7）明确选择；'
                        '不要手抄、四舍五入或猜测候选 UV，也不能重复一个候选来凑数量。'
                        if drawing_manifest_audit.get('status')=='confirmed' and surface_candidates else '')+
                       '重试图包含完整六视图图集，若上一轮局部位置无命中则另附带全视图绿色候选编号图；洋红色十字是旧 marker，青色十字是旧 registration point；'
                       '若旧视图没有可用实体候选，应切换到完整图集中另一个能看到该特征的视图，不能在零件外的视图里反复猜UV。'
                       '绿色 C 编号点是附近射线确实命中CAD的可选落点；候选包含 candidate_roles 和 source_point_names，measurement_marker 对应项应在 marker 中使用，registration_reference 对应项应在 registration_points 中使用。参考点必须在同一坐标系内空间分布充分、非共线；不得把多个不同 XYZ 点映射到相同 UV。只有候选确实对应 MSOP 特征时才能选择，并原样输出候选 UV。'
                       '绿色 C 候选图可直接通过 candidate_id（例如 C7）指定 markers 或 registration_points 的精确落点；此 ID 映射到真实 CAD 射线命中，优先输出 ID，不要抄写或估算 UV。每个不同 XYZ 点选择不同候选 ID。若能提出新的完整有效位置组，即使 confidence=low 也应 AUTO_MARK。'
                       +rigid_fit_instruction,
                       'attachments':[{'filename':'MSOP, drawing, and focused CAD retry view',
                           'parsed_text':context,'images':retry_images}]}]
    return dict(fai=item['number'],result='abstain_or_invalid',
        reason='model_selected_pixel_group_did_not_fully_hit_cad_surface',
        geometry_diagnostic=failure_diagnostic,
        registration_point_hit_count=len(registration_hits),
        missed_registration_points=failed_registration_points,
        rigid_fit_diagnostics={str(scope):{'max_residual_mm':fit.get('max_residual_mm'),
            'rms_residual_mm':fit.get('rms_residual_mm')} for scope,fit in rigid_fits.items()},
        marker_count=len(markers),raw_output=raw[:1200])


def mark_geometry_candidates(store, model, model_root: Path, project_id: str, generated_candidates=None):
    if not _local_model_runtime_ready(model):
        raise RuntimeError('共享本地 TALKER 模型尚未加载')
    if not _is_local_inference_model(model):
        raise RuntimeError('FAI 配准与标记只允许本机本地模型推理；已拒绝非本地模型端点')
    remote_inference = bool(getattr(model, 'openai_base_url', ''))
    adapter_bundle = None
    if not remote_inference:
        if model_root is None or not Path(model_root).exists():
            raise FileNotFoundError('本地 FAI 适配器目录不可用')
        adapter_bundle = discover_latest_adapter(model_root)
        if adapter_bundle.base_model_path.resolve() != model.bundle.base_model_path.resolve():
            raise ValueError('FAI 适配器与当前共享本地基座模型不一致')
    reconcile=getattr(store,'require_current_feature_integrity',None)
    if reconcile:reconcile(project_id)
    project = store.get(project_id)
    mesh = store.mesh(project_id)
    method_pages = store.active_msop(project)['method_pages']
    mesh_faces = {f['id']: f for f in mesh['faces']}
    results = []
    generated={c['fai']:c for c in (generated_candidates or [])}
    from .fai_strategy import strategy_state
    strategy_snapshot=(strategy_state(store,project_id) if callable(getattr(store,'_dir',None)) else {})
    strategies=({row['fai']:row for row in strategy_snapshot.get('items',[])}
                if not strategy_snapshot.get('source_stale') else {})
    adapter_name = 'viewer_fai_local'

    def discard_unaccepted_preview(item, old):
        if (old and old.get('origin') in ('geometry_candidate','msop_visual_candidate')
                and old.get('status')=='pending_review'):
            store.remove_annotation(project_id,item['number'])
        elif (old and old.get('origin')=='local_model_candidate'
              and old.get('status')=='stale'):
            store.remove_annotation(project_id,item['number'])

    # Native Strata/API backends cannot switch a Transformers PEFT adapter in-process.
    # In that mode, review candidates with the configured Strata model itself and make
    # the missing FAI LoRA explicit in provenance; never pretend the incompatible adapter ran.
    with model._lock:
        previous = None
        try:
            if adapter_bundle is not None:
                previous = model.model.active_adapter
                if isinstance(previous, list):
                    previous = previous[0]
                if adapter_name not in model.model.peft_config:
                    model.model.load_adapter(str(adapter_bundle.adapter_dir), adapter_name=adapter_name)
                model.model.set_adapter(adapter_name)
            for item in store.items(project):
                old = item.get('annotation')
                # Current local-model output is already the authoritative
                # automatic result. Everything else (manual binding, reviewed
                # pin, or a stale mark) is only historical input and cannot
                # veto a fresh model decision.
                if (old and old.get('origin') == 'local_model_candidate'
                        and old.get('status') == 'auto_marked'):
                    continue
                if old and old.get('origin') == 'local_model_candidate' and old.get('status') == 'reviewed':
                    old = None
                if old and old.get('origin') != 'local_model_candidate':
                    old = None
                if old and old.get('status')=='stale':
                    if item['number'] not in generated:
                        results.append({'fai':item['number'],'result':'stale_annotation_no_current_candidate'})
                        continue
                    old=None
                if not old and item['number'] in generated:
                    candidate = generated[item['number']]
                    old=dict(candidate,origin=candidate.get('origin','geometry_candidate'),status='pending_review',
                             automatic_xyz_hypothesis=candidate.get('candidate_kind') == 'xyz_display')
                if (not old or old.get('origin') not in ('geometry_candidate', 'msop_visual_candidate', 'local_model_candidate', 'same_source_stale_seed')
                        or old.get('status') == 'reviewed'):
                    continue
                store.update_local_model_marking(project_id,items=list(results),counts=dict(
                    accepted=sum(r['result']=='accepted' for r in results),
                    abstain=sum(r['result']=='abstain_or_invalid' for r in results),
                    skipped=sum(r['result'] not in ('accepted','abstain_or_invalid') for r in results),
                    processed=len(results)),
                    progress=dict(done=len(results),current_fai=item['number'],message=f"本地模型核对 FAI {item['number']} 的原文与候选面"))
                support = re.search(r'(\d+)/(\d+) 个显式点位', old.get('note', ''))
                support_ratio = (int(support.group(1)) / int(support.group(2))) if support and int(support.group(2)) else 1.0
                if old.get('automatic_xyz_hypothesis') and support_ratio < 0.65:
                    discard_unaccepted_preview(item,old)
                    results.append({'fai': item['number'], 'result': 'weak_geometric_support',
                                    'inlier_ratio': round(support_ratio, 3), 'threshold': 0.65})
                    continue
                detail = store.detail(project_id, item['number'])
                if old.get('candidate_kind') == 'model_visual_localization':
                    strategy=strategies.get(item['number']) or {}
                    if (strategy.get('next_route')=='assembly_applicability'
                            and strategy.get('evidence_class')=='explicit_absent_entity'):
                        results.append(dict(fai=item['number'],result='abstain_or_invalid',
                            reason='source_entity_absent_at_current_assembly_stage',
                            strategy_route='assembly_applicability',source_quotes=strategy.get('source_quotes'),
                            model_classification_reason=strategy.get('reason')))
                        continue
                    if strategy.get('next_route') in ('feature_topology_graph','drawing_pattern_to_part'):
                        from .part_identity import select_physical_parts
                        store.update_local_model_marking(project_id,progress=dict(done=len(results),
                            current_fai=item['number'],message=f"FAI {item['number']}：先核对 CAD 零部件身份与重复实例"))
                        source_images=_method_page_images(store,project,detail,model,item['number'])
                        parts,part_audit=select_physical_parts(store,project,item,detail,model,mesh,source_images)
                        if parts is None:
                            results.append(dict(fai=item['number'],result='abstain_or_invalid',
                                reason='local_part_identity_not_confirmed',strategy_route=strategy['next_route'],
                                local_model_audit=part_audit))
                            continue
                        context={**part_audit['decision'],'cad_sha256':mesh['source_sha256'],
                            'receipt':part_audit['receipt'],'strategy_route':strategy['next_route']}
                        old={**old,'evidence':{**old.get('evidence',{}),'part_identity_context':context}}
                    result=_mark_visual_fallback(store,project,item,detail,model,mesh,old)
                    if strategy:result['strategy_route']=strategy.get('next_route')
                    results.append(result)
                    continue
                conflicts = _source_scope_conflicts(old, detail, method_pages, item)
                if conflicts:
                    discard_unaccepted_preview(item,old)
                    results.append({'fai': item['number'], 'result': 'source_scope_conflict',
                                    'conflicting_point_names': conflicts, 'review_required': True})
                    continue
                question, context, candidate_ids, candidates, method = _prompt_data(
                    project, item, detail, old, mesh_faces, method_pages)
                if not candidate_ids:
                    discard_unaccepted_preview(item,old)
                    results.append({'fai': item['number'], 'result': 'no_valid_candidates'})
                    continue
                if len(context)+len(question)>48000:
                    results.append({'fai':item['number'],'result':'evidence_budget_exceeded','evidence_truncated':False})
                    continue
                messages = [{'role': 'user', 'content': question,
                             'attachments': [{'filename': 'Original MSOP and 2D drawing evidence', 'parsed_text': context,
                                              'images': _method_page_images(store, project, detail, model,
                                                                            item['number'])}]}]
                raw, truncated, token_count = model._generate_once(
                    messages, max_context_chars=len(context)+len(question)+512, max_new_tokens=4096,
                    max_images=5)
                parsed = _object(raw)
                decision = parsed.get('decision') if parsed else None
                selected = parsed.get('face_ids') if parsed else None
                decision_token = re.sub(r'[^A-Z_]', '', str(decision or '').upper())
                proposal_decisions = {
                    'AUTO_MARK', 'AUTOMARK', 'MARK_AUTOMATIC',
                    'PROPOSE_CANDIDATE', 'PROPOSE_REVIEW', 'PROPOSE_LOW_CONF',
                    'PENDING_REVIEW', 'REVIEW_REQUIRED',
                }
                valid = (isinstance(selected, list) and bool(selected)
                         and len(selected)<=12
                         and all(isinstance(face_id, str) and face_id in candidate_ids for face_id in selected))
                if old.get('automatic_xyz_hypothesis'):
                    valid = valid and _fai_number(parsed.get('fai'))==item['number']
                if decision_token not in proposal_decisions or not valid or truncated:
                    discard_unaccepted_preview(item,old)
                    results.append({'fai': item['number'], 'result': 'abstain_or_invalid',
                                    'decision': decision, 'truncated': truncated,
                                    'generated_tokens': token_count, 'raw_output': raw[:1000],
                                    'candidate_face_count': len(candidate_ids),
                                    'candidate_anchor_count': len(candidates)})
                    continue
                selected_ids = list(dict.fromkeys(selected))
                if old.get('automatic_xyz_hypothesis') and set(selected_ids)!=set(candidate_ids):
                    discard_unaccepted_preview(item,old)
                    results.append({'fai':item['number'],'result':'incomplete_model_selection',
                                    'selected_face_count':len(selected_ids),'required_face_count':len(candidate_ids)})
                    continue
                anchors = [a for a in candidates if a['face_id'] in selected_ids]
                if not anchors:
                    discard_unaccepted_preview(item,old)
                    results.append({'fai': item['number'], 'result': 'no_selected_anchors'})
                    continue
                confidence = _confidence(parsed.get('confidence'))
                model_confidence=confidence
                if old.get('automatic_xyz_hypothesis') or old.get('candidate_kind') == 'same_source_stale_seed':
                    confidence='low'
                model_source = (f"FAI LoRA：{adapter_bundle.adapter_dir}" if adapter_bundle else
                                f"Strata 模型：{getattr(model, 'openai_model', 'configured remote model')}；"
                                "未加载 Qwen3.5 专用 FAI LoRA（当前 Strata 模型不可热切换该适配器）")
                note = (f"本地模型自动标记。{model_source}. "
                        f"MSOP 第 {method.get('page', '?')} 页；模型依据：{parsed.get('evidence') or '未提供'}。"
                        f"模型决策：{decision_token}；模型自评置信度：{model_confidence or '未给出'}（未经校准）；"
                        f"显示置信度：{confidence or '未给出'}（未验证的自动 XYZ 假设上限为 low）。"
                        f"原几何候选说明：{old.get('note', '')}")
                entity_review=_review_visual_entity_group(store,project,item,detail,model,mesh,anchors,
                    [dict(feature='source feature') for _ in anchors],
                    _method_page_images(store,project,detail,model,item['number']))
                if entity_review.get('status')!='confirmed':
                    results.append(dict(fai=item['number'],result='abstain_or_invalid',
                        reason='local_entity_group_review_not_confirmed',entity_review=entity_review))
                    continue
                store.annotate(project_id, item['number'], anchors, note, 'auto_marked',
                               mesh['source_sha256'], item['document_id'], 'local_model_candidate', confidence, feature_verification=entity_review)
                results.append({'fai': item['number'], 'result': 'accepted', 'confidence': confidence,
                                'face_ids': selected_ids, 'anchor_count': len(anchors),
                                'decision': decision, 'truncated': truncated,
                                'source_pages': parsed.get('source_pages'),
                                'evidence': parsed.get('evidence'), 'raw_output': raw[:1000]})
        finally:
            if adapter_bundle is not None and previous is not None:
                model.model.set_adapter(previous)

    counts = {
        'accepted': sum(r['result'] == 'accepted' for r in results),
        'abstain': sum(r['result'] == 'abstain_or_invalid' for r in results),
        'skipped': sum(r['result'] not in ('accepted', 'abstain_or_invalid') for r in results),
        'processed': len(results),
    }
    return {'adapter': str(adapter_bundle.adapter_dir) if adapter_bundle else None,
            'model': getattr(model, 'openai_model', None) if remote_inference else None,
            'inference_backend': model.info().get('inference_backend'),
            'counts': counts, 'items': results}
