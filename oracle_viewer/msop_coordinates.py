"""Use explicit MSOP XYZ point tables as provenance-preserving CAD locations.

MSOP coordinates are never copied directly into the CAD frame. A drawing/rev
and source-hash-pinned XY calibration locates each named point on the FAI's
already selected CAD faces. Source Z remains in its original frame; equal-Z
measurement groups are jointly constrained to a common CAD face.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .candidate_mining import (
    COORDINATE_POINT, extract_feature_labels, extract_named_points,
    project_points_to_common_face, referenced_points,
)

LOCATION_VECTOR = re.compile(
    r"X,\s*Y,\s*Z\s*\(mm\)\s*([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([A-Z0-9_]+)"
    r"\s+I,\s*J,\s*K\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)", re.I
)
DIMENSION_LOCATION = re.compile(r'\bDimension\s+([XY])\s+location\b', re.I)
MEASURE_POINT_LIST = re.compile(
    r'\bMeasure\s+(\d+)\s+vector\s+points?\s+(.+?)'
    r'(?=\.|\band\s+construct\b|\bunder\b|\bDimension\b|$)',
    re.I | re.S,
)
MEASURED_GEOMETRY_LIST = re.compile(
    r'\bMeasure\s+(\d+)\s+(?:(?:vector|vision|surface)\s+)?(?:circles?|lines?|planes?)\s+(.+?)'
    r'(?=\.|\band\s+construct\b|\bunder\b|\bDimension\b|$)',
    re.I | re.S,
)


def _explicit_measure_point_names(methods, named_points):
    """Resolve complete, explicitly enumerated MSOP point groups.

    When a method names exactly N points, use only those names. Narrowing is
    allowed only when every operation resolves to its declared count.
    """
    resolved = set()
    found = False

    def aliases(name):
        values = {name.upper()}
        # MSOP text commonly shortens a table label such as DATUMB_PNT_P09
        # to DATUMB_P09 in its measurement instruction. Keep the source name
        # as the resolved identity; use this only to resolve the printed alias.
        if '_PNT_' in name.upper():
            prefix, suffix = name.upper().split('_PNT_', 1)
            values.add(f'{prefix}_{suffix}')
        return values

    alias_index = {}
    for name in named_points:
        for alias in aliases(name):
            alias_index.setdefault(alias, set()).add(name)

    # Build page-local groups once. A later MSOP result page can repeat a
    # Plane View instruction while the coordinates and point view appear only
    # on the first page. Resolve that repeat from a unique same-program,
    # same-frame coordinate group in the same FAI's methods instead of
    # returning an empty/partial point set.
    coordinate_groups = {}
    for name, point in named_points.items():
        for occurrence in point.get('occurrences', []):
            key = (occurrence.get('program'), occurrence.get('coordinate_system'),
                   occurrence.get('page'))
            if None not in key:
                coordinate_groups.setdefault(key, set()).add(name)

    def resolve(token):
        if token in named_points:
            return token
        matches = alias_index.get(token, set())
        return next(iter(matches)) if len(matches) == 1 else None

    for page in methods or []:
        text = page.get('text', '')
        from .msop_tables import table_point_manifest
        table_group=table_point_manifest(page)
        if table_group and all(resolve(name) for name in table_group['source_point_names']):
            found=True
            resolved.update(resolve(name) for name in table_group['source_point_names'])
        for match in MEASURE_POINT_LIST.finditer(text):
            found = True
            expected = int(match.group(1))
            clause = match.group(2)
            from .point_names import expand_point_clause
            literal_names=expand_point_clause(clause)
            if (literal_names and len(literal_names)==expected and len(set(literal_names))==expected
                    and all(resolve(name) for name in literal_names)):
                resolved.update(resolve(name) for name in literal_names)
                continue
            tokens = [token.strip(' \t\r\n,;:') for token in re.split(r',', clause)]
            names = set()
            previous_prefix = ''
            for token in tokens:
                token = re.sub(r'\s+', '', token).upper()
                if not token:
                    continue
                # MSOP measurement lists abbreviate a shared prefix with a
                # slash, e.g. F1_PNT_Z1/Z2 or DATUMC_P04/03/47/46. Expand
                # those aliases before resolving the exact source point names;
                # otherwise semantic retrieval can pull unrelated similarly
                # named points from other FAI method pages.
                slash_parts=token.split('/')
                if len(slash_parts)>1:
                    head=slash_parts[0]
                    head_match=(re.match(r'^(.*_)([A-Z]*)(\d+)$',head)
                                or re.match(r'^(.*?)([A-Z_]*)(\d+)$',head))
                    expanded_slash=[head]
                    if head_match:
                        prefix,tag,_number=head_match.groups()
                        for part in slash_parts[1:]:
                            if resolve(part):
                                expanded_slash.append(part)
                                continue
                            suffix_match=re.match(r'^([A-Z_]*)(\d+)$',part)
                            if not suffix_match:
                                expanded_slash=[]
                                break
                            suffix,number=suffix_match.groups()
                            expanded_slash.append(prefix+(suffix or tag)+number)
                    else:
                        expanded_slash=[]
                    if expanded_slash and all(resolve(value) for value in expanded_slash):
                        slash_expanded=expanded_slash
                    else:
                        slash_expanded=[token]
                else:
                    slash_expanded=[token]
                endpoints=[value for slash_token in slash_expanded
                           for value in re.split(r'~|–|—|\bTO\b',slash_token)]
                expanded = []
                for endpoint in endpoints:
                    endpoint = endpoint.strip()
                    if not endpoint:
                        continue
                    token_name = endpoint if resolve(endpoint) else (previous_prefix + endpoint if previous_prefix else endpoint)
                    expanded.append((resolve(token_name), token_name))
                if (len(expanded)==expected
                        and all(name is not None for name,_token in expanded)):
                    names.update(name for name,_token in expanded)
                    continue
                if len(expanded) == 1 and expanded[0][0] is not None:
                    name = expanded[0][0]
                    names.add(name)
                    suffix = re.match(r'^(.*_)[A-Z]+\d+[A-Z]?$', name)
                    if suffix:
                        previous_prefix = suffix.group(1)
                    continue
                if len(expanded) == 2:
                    left_match = re.match(r'^(.*?)(\d+)([A-Z_]*)$', expanded[0][1])
                    right_match = re.match(r'^(.*?)(\d+)([A-Z_]*)$', expanded[1][1])
                    if (left_match and right_match and left_match.group(1) == right_match.group(1)
                            and left_match.group(3) == right_match.group(3)):
                        lo, hi = sorted((int(left_match.group(2)), int(right_match.group(2))))
                        for name in named_points:
                            for alias in aliases(name):
                                m = re.match(r'^(.*?)(\d+)([A-Z_]*)$', alias)
                                if (m and m.group(1) == left_match.group(1)
                                        and m.group(3) == left_match.group(3)
                                        and lo <= int(m.group(2)) <= hi):
                                    names.add(name)
                                    break
            if len(names) != expected:
                # Some MSOP instructions say "as Plane View show" instead of
                # printing point names. Use the page-local XYZ table only when
                # it contains exactly the declared count in the measured frame.
                has_named_reference = any(
                    re.search(rf'(?<![A-Z0-9_]){re.escape(name)}(?![A-Z0-9_])', clause, re.I)
                    for name in named_points)
                measured_frame = page.get('measured_system')
                local_names = set()
                if not has_named_reference and measured_frame:
                    for name, point in named_points.items():
                        if any(o.get('page') == page.get('page')
                               and o.get('program') == page.get('program')
                               and o.get('coordinate_system') == measured_frame
                               for o in point.get('occurrences', [])):
                            local_names.add(name)
                    if len(local_names) != expected:
                        scoped_groups = {
                            frozenset(group) for (program, frame, _), group in coordinate_groups.items()
                            if program == page.get('program') and frame == measured_frame
                            and len(group) == expected
                        }
                        if len(scoped_groups) == 1:
                            local_names = set(next(iter(scoped_groups)))
                if len(local_names) == expected:
                    names = local_names
                else:
                    return None
            resolved.update(names)
        # Circle/line/plane instructions also identify measured geometry whose
        # center or construction point has an XYZ row. Include those exact
        # measured features alongside vector-point constructions; do not let a
        # supporting datum point list hide the actual FAI feature.
        for match in MEASURED_GEOMETRY_LIST.finditer(text):
            found = True
            expected = int(match.group(1))
            tokens = set(re.findall(r'[A-Z][A-Z0-9_]*', match.group(2).upper()))
            names = {resolve(token) for token in tokens if resolve(token) is not None}
            if len(names) != expected:
                return None
            resolved.update(names)
    return resolved if found and resolved else None


def _measure_point_basis(methods, named_points):
    """Describe how a complete point group was resolved for audit/UI context."""
    names = _explicit_measure_point_names(methods, named_points)
    if not names:
        return None
    for page in methods or []:
        for match in MEASURE_POINT_LIST.finditer(page.get('text', '')):
            clause = match.group(2)
            if not any(re.search(rf'(?<![A-Z0-9_]){re.escape(name)}(?![A-Z0-9_])', clause, re.I)
                       for name in named_points):
                return 'same method page; unique measured-frame XYZ row count matches Measure N'
    return 'explicit point names and Measure N count match'


def _nominal_specification(page):
    specification = page.get('specification') or {}
    try:
        nominal = float(specification['nominal'])
        upper = float(specification.get('upper') or 0)
        lower = float(specification.get('lower') or 0)
        return nominal, upper, lower
    except (KeyError, TypeError, ValueError):
        pass
    heading = page.get('heading') or page.get('text', '').splitlines()[0]
    patterns = (
        (re.compile(r'([-+]?\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)'), 'symmetric'),
        (re.compile(r'([-+]?\d+(?:\.\d+)?)\s+([+-]\d+(?:\.\d+)?)/([+-]\d+(?:\.\d+)?)'), 'bilateral'),
        (re.compile(r'([-+]?\d+(?:\.\d+)?)\s+(\+\d+(?:\.\d+)?)\b'), 'upper_only'),
    )
    for pattern, kind in patterns:
        match = pattern.search(heading)
        if not match:
            continue
        if kind == 'symmetric':
            nominal, tolerance = map(float, match.groups())
            return nominal, tolerance, -tolerance
        if kind == 'bilateral':
            return tuple(map(float, match.groups()))
        nominal, upper = map(float, match.groups())
        return nominal, upper, 0.0
    return None


def _rank_semantic_coordinate_pages(rows, pages):
    """Use sibling FAI nominal axes to disambiguate repeated point tables."""
    by_feature = {}
    for row in rows:
        if row.get('semantic_feature'):
            by_feature.setdefault(row['semantic_feature'], []).append(row)
    page_by_number = {p.get('page'): p for p in pages or []}
    for feature, feature_rows in by_feature.items():
        source_pages = sorted({p for row in feature_rows for p in row.get('semantic_source_pages', [])})
        constraints = []
        for page in pages or []:
            labels = set(page.get('features') or extract_feature_labels([page]))
            if feature not in labels:
                continue
            axis = DIMENSION_LOCATION.search(page.get('text', ''))
            spec = _nominal_specification(page)
            if spec is None:
                continue
            if not axis:
                continue
            target, upper, lower = spec
            tol = max(abs(upper), abs(lower), .01)
            constraints.append((axis.group(1).upper(), target, tol,
                                page.get('page'), page.get('fai_ids', [])))
        if len(source_pages) < 2 or not constraints:
            continue
        scored = []
        for source_page in source_pages:
            values = []
            for row in feature_rows:
                occurrence = next((o for o in row.get('occurrences', [])
                                   if o.get('page') == source_page), None)
                if occurrence is None:
                    break
                values.append(occurrence['xyz'])
            if len(values) != len(feature_rows):
                continue
            center = np.mean(np.asarray(values, float), axis=0)
            evidence = [(abs(center['XYZ'.index(axis)] - target) / tol,
                         {'axis': axis, 'target_mm': target, 'group_center_mm': float(center['XYZ'.index(axis)]),
                          'normalized_error': abs(center['XYZ'.index(axis)] - target) / tol,
                          'fai_ids': fai_ids, 'method_page': method_page})
                        for axis, target, tol, method_page, fai_ids in constraints]
            scored.append((sum(value for value, _ in evidence), source_page, evidence))
        if len(scored) < 2:
            continue
        scored.sort(key=lambda value: (value[0], value[1]))
        if scored[1][0] - scored[0][0] < .5:
            continue
        best_page = scored[0][1]
        for row in feature_rows:
            row['semantic_source_pages'] = [best_page]
            row['semantic_resolution'] = {
                'basis': 'sibling FAI nominal coordinate and dimension axis',
                'selected_source_page': best_page,
                'score': scored[0][0],
                'runner_up_score': scored[1][0],
                'constraints': [item for _, item in scored[0][2]],
            }


def project_calibration(project: dict, root: Path | None = None) -> dict | None:
    """Load only a calibration pinned to this exact drawing, revision and CAD/MSOP."""
    base = root or Path(__file__).resolve().parent.parent
    drawing, revision = str(project.get('drawing_no') or ''), str(project.get('revision') or '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', drawing) or not re.fullmatch(r'[A-Za-z0-9_-]+', revision):
        return None
    path = base / 'config' / 'cad_frame_calibrations' / f'{drawing}-{revision}.json'
    if not path.is_file():
        return None
    calibration = json.loads(path.read_text(encoding='utf-8'))
    msop = next((d for d in project.get('documents', []) if d.get('role') == 'msop'), None)
    model = project.get('model') or {}
    if (calibration.get('drawing') != project.get('drawing_no')
            or str(calibration.get('revision')) != str(project.get('revision'))
            or calibration.get('cad_sha256') != model.get('source_sha256')
            or not msop or calibration.get('msop_sha256') != msop.get('sha256')):
        return None
    return calibration


def _coordinate_rows(methods, dependencies, feature_labels, all_pages=None, named_points=None):
    from .point_context import contextual_pages,explicit_point_frames,resolve_scopes
    named = named_points if named_points is not None else extract_named_points(all_pages or methods or [])
    point_pages = contextual_pages(list(all_pages or methods or []), named)
    by_page = {p.get('page'): p for p in point_pages}
    pages = [by_page.get(p.get('page'), p) for p in methods or []]
    explicit_names = _explicit_measure_point_names(pages, named)
    if explicit_names:
        # The explicit list is authoritative and may use shorthand labels that
        # the general semantic retriever cannot match to the full XYZ table
        # names. Resolve the complete declared set directly from the corpus.
        selected = {name: named[name] for name in explicit_names if name in named}
    else:
        selected = referenced_points(None, pages, named, feature_labels)
    rows = []
    _rank_semantic_coordinate_pages(list(selected.values()), point_pages)
    for source in selected.values():
      for scoped_source in resolve_scopes(source, pages):
        row = dict(scoped_source, ijk=source.get('ijk'))
        vectors = []
        for page in point_pages:
            if (page.get('page') not in row.get('pages', [])
                    or page.get('program') != row.get('program')):
                continue
            frame=explicit_point_frames(page,named).get(row['name'],{}).get('coordinate_system',page.get('measured_system'))
            if frame!=row.get('coordinate_system'):continue
            for x,y,z,name,i,j,k in LOCATION_VECTOR.findall(page.get('text', '')):
                if name == row['name'] and np.allclose(row['xyz'],[float(x),float(y),float(z)],atol=1e-6,rtol=0):
                    vectors.append([float(i),float(j),float(k)])
        if vectors:
            if any(not np.allclose(v, vectors[0], atol=1e-6, rtol=0) for v in vectors[1:]):
                row['vector_conflict'] = True
                row['coordinate_conflict'] = True
            else:
                row['ijk'] = vectors[0]
        rows.append(row)
    return rows


def map_task_points(task: dict, mesh: dict, calibration: dict | None, all_pages=None, named_points=None) -> dict:
    """Project all explicit source points onto this FAI's selected CAD face set.

    The result distinguishes source XYZ from CAD display XYZ. Mapping uses the
    calibrated XY system and constrained CAD faces; source Z is preserved as a
    grouping datum, never treated as a registered CAD Z value.
    """
    rows = _coordinate_rows(task.get('methods'), task.get('dependencies'), task.get('features', []), all_pages, named_points)
    from .section_geometry import source_plane_constraints
    constraints=source_plane_constraints(rows,task.get('methods',[]))
    if not rows:
        return dict(status='no_explicit_points', source_frame_registered_to_cad=False,
                    source_points=[], source_point_constraints=[], mapped_points=[], common_source_z_groups=[])
    method_text='\n'.join(m.get('text','') for m in task.get('methods',[]))
    plane_feature=bool(re.search(r'construct\s+(?:(?:a|one|1)\s+)?plane\b',method_text,re.I) and re.search(r'\bflatness\b',method_text,re.I))
    anchor_faces = list(dict.fromkeys(a.get('face_id') for a in task.get('anchors', []) if a.get('face_id')))
    if not calibration or not anchor_faces:
        return dict(status='coordinates_preserved_unmapped', source_frame_registered_to_cad=False,
                    reason='No source-hash-matched project calibration or selected CAD face set.',
                    source_points=rows, source_point_constraints=constraints, mapped_points=[], common_source_z_groups=[])
    faces = {f['id']: f for f in mesh.get('faces', []) if f['id'] in anchor_faces and f.get('indices')}
    if not faces:
        return dict(status='coordinates_preserved_unmapped', source_frame_registered_to_cad=False,
                    reason='Selected CAD faces are unavailable.', source_points=rows, source_point_constraints=constraints,
                    mapped_points=[], common_source_z_groups=[])

    triangles, face_ids = [], []
    for face in faces.values():
        vertices = np.asarray(face['positions'], dtype=np.float32).reshape(-1, 3)
        indices = np.asarray(face['indices'], dtype=np.int64).reshape(-1, 3)
        triangles.extend(vertices[indices])
        face_ids.extend([face['id']] * len(indices))
    triangles = np.asarray(triangles, dtype=np.float32)
    face_ids = np.asarray(face_ids, dtype=object)
    hypotheses = calibration.get('hypotheses', [])
    from .candidate_mining import nearest_xy_triangle
    projected = []
    applied_rigids = {}
    unknown_rows = [r for r in rows if not r.get('coordinate_system')]
    inferred_hypothesis = None
    if unknown_rows:
        scores = []
        for h in hypotheses:
            distances = []
            for row in unknown_rows:
                x, y, _ = row['xyz']
                xy = [float(h.get('x_sign', 1))*x+float(h.get('x_offset_mm', 0)),
                      float(h.get('y_sign', 1))*y+float(h.get('y_offset_mm', 0))]
                distances.append(nearest_xy_triangle(triangles, face_ids, *xy)[0])
            good = [d for d in distances if d <= float(calibration.get('max_xy_residual_mm', 1.0))]
            scores.append((len(good), -float(np.median(good)) if good else -float('inf'), h))
        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if scores:
            best = scores[0];runner = scores[1] if len(scores)>1 else (0, -float('inf'), None)
            needed = max(int(calibration.get('min_inliers', 2)), int(np.ceil(len(unknown_rows)*float(calibration.get('min_inlier_ratio', .5)))))
            decisive = best[0]-runner[0] >= max(2, int(np.ceil(len(unknown_rows)*.1))) or best[1]-runner[1] >= .25
            if best[0] >= needed and decisive:
                inferred_hypothesis=best[2]

    named_index = named_points if named_points is not None else extract_named_points(all_pages or task.get('methods', []))
    programs_by_frame = {}
    for value in named_index.values():
        for occurrence in value.get('occurrences', []):
            programs_by_frame.setdefault(occurrence.get('coordinate_system'), set()).add(occurrence.get('program'))
    for row in rows:
        if row.get('coordinate_conflict'):continue
        systems = {str(row.get('coordinate_system') or '').upper()}
        source_pages = set(row.get('pages', []))
        for page in list(task.get('methods', [])) + list(task.get('dependencies', [])):
            if page.get('page') in source_pages and page.get('measured_system'):
                systems.add(str(page['measured_system']).upper())
        options = ([h for h in hypotheses if not h.get('systems') or systems.intersection(
            str(s).upper() for s in h['systems'])] if row.get('coordinate_system') else
            ([inferred_hypothesis] if inferred_hypothesis else []))
        frame_programs = programs_by_frame.get(row.get('coordinate_system'), set())
        # A legacy alignment-only calibration cannot apply to both flip
        # programs. Require an explicit program when the source reuses a frame.
        options = [h for h in options if (h.get('program') is not None and h.get('program') == row.get('program'))
                   or (not h.get('program') and len(frame_programs) <= 1)]
        rigid=next((r for r in calibration.get('rigid_frame_proposals',[])
                    if r.get('frame')==row.get('coordinate_system') and
                    ((r.get('program') is not None and r.get('program')==row.get('program')) or (not r.get('program') and len(frame_programs)<=1))
                    and r.get('status')=='geometrically_supported_proposal'),None)
        if len(options) != 1:
            if not rigid:continue
            options=[dict(name='source-hash-pinned display rigid fit: '+rigid['frame'])]
        h = options[0]
        x, y, _ = row['xyz']
        mapped_xy = [float(h.get('x_sign', 1)) * x + float(h.get('x_offset_mm', 0)),
                     float(h.get('y_sign', 1)) * y + float(h.get('y_offset_mm', 0))]
        spatial_residual=None
        if rigid:
            from .visibility import closest_surface
            from .section_geometry import closest_plane_section
            rotation=np.asarray(rigid['rotation'],float);translation=np.asarray(rigid['translation_mm'],float)
            if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1,atol=1e-6):
                continue
            cad_target=rotation@np.asarray(row['xyz'],float)+translation
            hits=[]
            for fid in anchor_faces:
                mask=face_ids==fid
                if not mask.any():continue
                if plane_feature:
                    hit=closest_surface(cad_target,triangles[mask])
                else:
                    plane_normal=rotation[:,2]
                    plane_origin=rotation@np.array([0.,0.,row['xyz'][2]])+translation
                    hit=closest_plane_section(cad_target,triangles[mask],plane_normal,float(plane_normal@plane_origin))
                if hit:hits.append((hit['distance_mm'],fid,hit['point']))
            if not hits:continue
            spatial_residual,face_id,position=min(hits,key=lambda p:p[0])
            mapped_xy=cad_target[:2].tolist();residual=float(np.linalg.norm(np.asarray(position[:2])-cad_target[:2]))
            if spatial_residual>float(rigid.get('threshold_mm',.1)):continue
            applied_rigids[(row.get('program'),row.get('coordinate_system'))]=rigid
        else:
            residual, face_id, position = nearest_xy_triangle(triangles, face_ids, *mapped_xy)
        projected.append(dict(name=row['name'], point_key=row.get('point_key'), program=row.get('program'), source_xyz=row['xyz'], source_ijk=row.get('ijk'),
                              source_pages=row.get('pages', []), coordinate_system=row.get('coordinate_system'),
                              calibration=h['name'], mapped_xy=mapped_xy, face_id=face_id,
                              cad_position_mm=position, xy_residual_mm=float(residual),
                              xyz_residual_mm=spatial_residual,source_z_display_registered=bool(rigid),
                              registration_authority='display_geometry_fitted_not_metrology' if rigid else 'xy_hypothesis_only'))

    tolerance = float(calibration.get('max_xy_residual_mm', 1.0))
    mapped = [p for p in projected if p['xy_residual_mm'] <= tolerance]
    # Uniform source-Z sets used to construct one measured feature must stay on
    # a single one of the already selected faces, even when layers overlap in XY.
    source_z_groups = []
    group_mapping_complete = True
    by_frame = {}
    for point in mapped:
        by_frame.setdefault((point.get('program'), point.get('coordinate_system'), point.get('calibration')), []).append(point)
    for (program, system, frame_name), frame_points in by_frame.items():
      ordered=sorted(frame_points,key=lambda p:p['source_xyz'][2]);z_groups=[]
      for point in ordered:
        if not z_groups or abs(point['source_xyz'][2]-z_groups[-1][0]['source_xyz'][2])>float(calibration.get('same_source_z_tolerance_mm',.001)):
          z_groups.append([point])
        else:z_groups[-1].append(point)
      for group in z_groups:
        zs = [p['source_xyz'][2] for p in group]
        if len(group) < 3:continue
        # An established 3D display fit must not be overwritten by XY-only
        # common-face retrieval. Verify the common-face agreement instead.
        if any(p.get('source_z_display_registered') for p in group):
            rigid=applied_rigids[(program,system)]
            normal=np.asarray(rigid['rotation'],float)[:,2]
            positions=np.asarray([p['cad_position_mm'] for p in group],float)
            span=float(np.ptp(positions@normal))
            consistent=span<=float(calibration.get('same_source_z_tolerance_mm',.001))
            if not consistent:group_mapping_complete=False
            source_z_groups.append(dict(program=program,coordinate_system=system,calibration=frame_name,source_z_mm=zs[0],point_names=[p['name'] for p in group],cad_face_ids=list(dict.fromkeys(p['face_id'] for p in group)),status='registered_measurement_plane' if consistent else 'registered_points_leave_measurement_plane',plane_normal_cad=normal.tolist(),plane_coordinate_span_mm=span,cad_z_span_mm=float(np.ptp(positions[:,2])),common_face_required=plane_feature))
            continue
        if not plane_feature:
            # Equal source Z defines a section plane, not one B-Rep face.
            # Without a registered plane, XY-only projections cannot choose
            # a physical layer or safely replace the existing markers.
            group_mapping_complete=False
            source_z_groups.append(dict(program=program,coordinate_system=system,calibration=frame_name,source_z_mm=zs[0],point_names=[p['name'] for p in group],status='measurement_plane_registration_required',common_face_required=False))
            continue
        candidates = [{'name': p['name'], 'xyz': p['source_xyz'], 'pages': p['source_pages'],
                       'mapped_xy': p['mapped_xy']} for p in group]
        common = project_points_to_common_face(candidates, triangles, face_ids, anchor_faces,
                                               max_xy_residual_mm=tolerance,
                                               min_coverage=float(calibration.get('common_face_min_coverage', .9)))
        if not common:
            group_mapping_complete = False
            source_z_groups.append(dict(program=program, coordinate_system=system, calibration=frame_name, source_z_mm=zs[0],
                                        point_names=[p['name'] for p in group], status='ambiguous'))
            continue
        if common['coverage'] < 1.0:
            group_mapping_complete = False
            source_z_groups.append(dict(program=program, coordinate_system=system, calibration=frame_name, source_z_mm=zs[0],
                                        point_names=[p['name'] for p in group], status='common_cad_face_partial',
                                        cad_face_id=common['face_id'], coverage=common['coverage']))
            continue
        replacements = {p['name']: p for p in common['inliers']}
        mapped = [dict(p, face_id=replacements[p['name']]['face_id'],
                       cad_position_mm=replacements[p['name']]['position'],
                       xy_residual_mm=float(np.linalg.norm(np.asarray(replacements[p['name']]['position'][:2])-p['mapped_xy']))) if p in group and p['name'] in replacements else p
                  for p in mapped]
        source_z_groups.append(dict(program=program, coordinate_system=system, calibration=frame_name, source_z_mm=zs[0],
                                    point_names=[p['name'] for p in group],
                                    cad_face_id=common['face_id'], status='common_cad_face',
                                    cad_z_span_mm=common['z_span_mm'],
                                    alternatives=common['alternatives']))

    anchors = task.get('anchors', [])
    complete = group_mapping_complete and len(mapped) == len(rows) and len(rows) == len(anchors)
    anchor_match = []
    if complete:
        # Preserve the existing marker/atom index only when a one-to-one spatial
        # match exists. This avoids silently swapping labels on similar points.
        distances = np.linalg.norm(np.asarray([p['cad_position_mm'] for p in mapped])[:,None,:]-
                                   np.asarray([a['position'] for a in anchors])[None,:,:],axis=2)
        limit = float(calibration.get('max_anchor_correspondence_mm', 2.0))
        choices = [sorted((i for i in range(len(anchors)) if d[i]<=limit),key=lambda i:d[i]) for d in distances]
        owners = {}
        def assign(point_index, seen):
            for index in choices[point_index]:
                if index in seen: continue
                seen.add(index)
                if index not in owners or assign(owners[index],seen):
                    owners[index] = point_index; return True
            return False
        complete = all(assign(i,set()) for i in sorted(range(len(mapped)),key=lambda i:len(choices[i])))
        if complete:
            anchor_match = [dict(point_name=mapped[p]['name'],anchor_index=i,
                                 match_distance_mm=float(distances[p,i])) for i,p in sorted(owners.items())]
    accepted = {(p.get('point_key') or p['name'],p.get('coordinate_system'),p.get('calibration')) for p in mapped}
    return dict(status='mapped_complete' if complete else 'mapped_partial',
                source_frame_registered_to_cad=bool(mapped) and len(mapped)==len(rows) and all(p.get('source_z_display_registered') for p in mapped),
                registration_scope='display_geometry_only_not_metrology',
                mapping='Project-calibrated XY onto selected CAD faces; source Z retained only in MSOP frame.',
                inferred_frame=({"calibration":inferred_hypothesis['name'],"basis":"unique geometric support among project-pinned transforms"} if inferred_hypothesis else None),
                source_points=rows, source_point_constraints=constraints, mapped_points=mapped, rejected_points=[p for p in projected if (p.get('point_key') or p['name'],p.get('coordinate_system'),p.get('calibration')) not in accepted],
                common_source_z_groups=source_z_groups, anchor_correspondence=anchor_match,
                can_replace_visual_anchors=complete)
