"""Reusable, provenance-preserving retrieval of MSOP coordinates on CAD meshes.

This module deliberately knows nothing about a drawing number or FAI range. A
project supplies its coordinate-frame calibration hypotheses; all point parsing,
cross-page feature retrieval, geometric screening, and provenance are shared.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from .point_context import contextual_pages, resolve_scopes, scope_rows, explicit_point_frames, point_key

COORDINATE_POINT = re.compile(
    r"X,\s*Y,\s*Z\s*\(mm\)\s*([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([A-Z0-9_]+)", re.I
)
IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", re.I)
GENERIC_FEATURE_TOKENS = {"CIR", "LIN", "PLN", "PNT", "POINT", "VISION", "FEATURE", "DATUM"}
GEOMETRY_TOKENS = {"CIR", "LIN", "PLN", "PNT"}
POINT_ORDINAL = re.compile(r"^(?:P|PT|POINT)0*\d+[A-Z]?$", re.I)
MEASURE_POINT_OPERATION = re.compile(
    r"\bMeasure\s+(\d+)\s+vector\s+points?\b(.*?)(?=\bMeasure\b|$)", re.I | re.S
)
POINT_REFERENCE = re.compile(r"\b([A-Z][A-Z0-9_]*PNT[A-Z0-9_]*)\b", re.I)
CONSTRUCTED_GEOMETRY = re.compile(
    r"\bconstruct(?:\s+(?:a|an|the|\d+))?\s+(plane|line)\s+([A-Z][A-Z0-9_]*)", re.I
)


def _point_family(feature_name):
    match = re.match(r"^(.+?)_(?:(?:[XYZ]LIN)|(?:PLN|LIN|CIR))(?:_|\d|$)", feature_name.upper())
    return match.group(1) if match else ""


def _point_number(name):
    match = re.search(r"(?:PNT|POINT)(?:_[A-Z]+)?_?0*(\d+)$", name.upper())
    return int(match.group(1)) if match else None


def _constructed_point_sets(pages, named_points):
    """Read MSOP's point-to-constructed-plane/line relationships.

    These associations let a vision feature reuse the matching measured point
    set from another page without sweeping every XYZ point sharing a prefix
    into one candidate.
    """
    result = []
    for page in pages:
        text = page.get("text", "")
        narrative = re.split(r"(?:\bLocation Definitions\b|X,\s*Y,\s*Z\s*\(mm\))",
                             text, maxsplit=1, flags=re.I)[0]
        local_names = sorted(
            (name for name, row in named_points.items()
             if any(o.get("page") == page.get("page") for o in row.get("occurrences", []))),
            key=lambda name: (_point_number(name) or 10**9, name))
        offsets = {}
        for operation in MEASURE_POINT_OPERATION.finditer(narrative):
            count = int(operation.group(1))
            clause = operation.group(2)
            construct = CONSTRUCTED_GEOMETRY.search(clause)
            if not construct:
                continue
            kind, feature = construct.group(1).lower(), construct.group(2).upper()
            if feature in {'PROBE', 'MASTER_P5', 'TSTAR_15MM_Y', 'TIP1', 'METHOD', 'PROGRAM', 'DIMENSION'}:
                method_feature = re.search(r'\bMethod\s+([A-Z][A-Z0-9_]+)', clause, re.I)
                if method_feature:
                    feature = method_feature.group(1).upper()
            family = _point_family(feature)
            if not family:
                continue
            candidates = [name for name in named_points
                          if name.upper().startswith(family + "_PNT")]
            selected = []
            refs = list(POINT_REFERENCE.finditer(clause))
            if refs:
                for ref in refs:
                    start = ref.group(1).upper()
                    prefix_match = re.match(r"^(.*_PNT_?)0*(\d+)$", start)
                    if not prefix_match:
                        continue
                    prefix, first = prefix_match.group(1), int(prefix_match.group(2))
                    if not prefix.startswith(family + "_"):
                        continue
                    end = first
                    tail = clause[ref.end():]
                    range_match = re.match(r"\s*(?:~|–|—|TO)\s*([A-Z][A-Z0-9_]*PNT[A-Z0-9_]*|PNT_?\d+)", tail, re.I)
                    if range_match:
                        raw_end = range_match.group(1).upper()
                        end_match = re.search(r"(?:PNT|POINT)_?0*(\d+)$", raw_end)
                        if end_match:
                            end = int(end_match.group(1))
                    low, high = sorted((first, end))
                    selected.extend(name for name in candidates
                                    if _point_number(name) is not None
                                    and low <= _point_number(name) <= high)
            if not selected:
                family_names = [name for name in local_names
                                if name.upper().startswith(family + "_PNT")]
                offset = offsets.get(family, 0)
                selected = family_names[offset:offset + count]
                offsets[family] = offset + count
            selected = list(dict.fromkeys(selected))
            if selected:
                explicit_frames = explicit_point_frames(page, named_points)
                selected_frames = {explicit_frames[name].get('coordinate_system')
                                   for name in selected if name in explicit_frames
                                   and not explicit_frames[name].get('frame_assignment_conflict')
                                   and explicit_frames[name].get('coordinate_system')}
                coordinate_system = (next(iter(selected_frames)) if len(selected_frames) == 1
                                     else page.get('measured_system'))
                coordinate_pages = sorted({occurrence.get('page')
                    for name in selected for occurrence in named_points[name].get('occurrences', [])
                    if (not page.get('program') or occurrence.get('program') == page.get('program'))
                    and (not coordinate_system or occurrence.get('coordinate_system') == coordinate_system)
                    and occurrence.get('page') is not None})
                result.append(dict(feature=feature, family=family, kind=kind,
                                   point_names=selected, page=page.get("page"),
                                   coordinate_pages=coordinate_pages,
                                   program=page.get("program"),
                                   coordinate_system=coordinate_system))
    return result


def constructed_support_rows(rows, method_pages, named_points):
    """Return sibling point rows for a shared constructed feature family.

    These rows are calibration support only. Callers must continue to mark and
    validate the target FAI's own complete point set separately.
    """
    page_numbers = {page.get('page') for page in method_pages or []}
    families = set()
    scopes = {(row.get('program'), row.get('coordinate_system')) for row in rows}
    for row in rows:
        for relation in row.get('constructed_point_sets', []):
            if relation.get('page') in page_numbers and relation.get('family'):
                scope = (relation.get('program'), relation.get('coordinate_system'))
                if scope in scopes and relation.get('coordinate_system'):
                    families.add((relation['family'], relation.get('kind'), *scope))
    if not families:
        return []
    selected = {}
    conflicts = set()
    for name, point in named_points.items():
        for relation in point.get('constructed_point_sets', []):
            if name not in relation.get('point_names', []):
                continue
            key = (relation.get('family'), relation.get('kind'),
                   relation.get('program'), relation.get('coordinate_system'))
            if key not in families:
                continue
            pages = set(relation.get('coordinate_pages') or [])
            if not pages:
                continue
            for occurrence in point.get('occurrences', []):
                if (occurrence.get('page') not in pages
                        or occurrence.get('program') != relation.get('program')
                        or occurrence.get('coordinate_system') != relation.get('coordinate_system')):
                    continue
                scope = (occurrence.get('program'), occurrence.get('coordinate_system'))
                key_point = point_key(name, *scope)
                candidate = dict(point, xyz=occurrence['xyz'], pages=[occurrence['page']],
                    program=scope[0], coordinate_system=scope[1], occurrences=[occurrence],
                    point_key=key_point, coordinate_conflict=False, scope_ambiguous=False)
                old = selected.get(key_point)
                if old and not np.allclose(old['xyz'], candidate['xyz'], atol=1e-6, rtol=0):
                    conflicts.add(key_point)
                else:
                    selected[key_point] = candidate
    return [selected[key] for key in sorted(selected) if key not in conflicts]


def resolve_dominant_source_page_conflicts(rows, *, minimum_page_coverage=0.7):
    """Resolve repeated XYZ labels only when a coherent page defines the whole point set.

    A later consolidated MSOP coordinate table can repeat a few point labels
    from earlier FAI pages with slightly different coordinates. We may select
    its occurrence only when it is the unique page covering at least the
    required fraction of the same program/frame point group and it supplies an
    occurrence for every conflicting point. All variants remain in provenance.
    """
    import copy
    resolved=[copy.deepcopy(row) for row in rows]
    evidence=[]
    scopes={
        (row.get('program'),row.get('coordinate_system'))
        for row in resolved if row.get('coordinate_system')
    }
    for program,frame in sorted(scopes):
        scoped=[row for row in resolved if (row.get('program'),row.get('coordinate_system'))==(program,frame)]
        conflicts=[row for row in scoped if row.get('coordinate_conflict')]
        if not conflicts or any(any(o.get('frame_assignment_conflict') for o in row.get('occurrences',[]))
                                for row in conflicts):
            continue
        page_names={}
        for row in scoped:
            pages={o.get('page') for o in row.get('occurrences',[])
                   if o.get('page') is not None and o.get('program')==program
                   and o.get('coordinate_system')==frame and not o.get('frame_assignment_conflict')}
            for page in pages:page_names.setdefault(page,set()).add(row.get('point_key') or row['name'])
        ranked=sorted(((len(names),page) for page,names in page_names.items()),reverse=True)
        if not ranked:continue
        top_count,page=ranked[0]
        if (top_count/len(scoped)<minimum_page_coverage
                or len(ranked)>1 and ranked[1][0]==top_count):
            continue
        choices={}
        for row in conflicts:
            values=[o for o in row.get('occurrences',[])
                    if o.get('page')==page and o.get('program')==program
                    and o.get('coordinate_system')==frame and not o.get('frame_assignment_conflict')]
            if len(values)!=1:
                choices={};break
            choices[row.get('point_key') or row['name']]=values[0]
        if len(choices)!=len(conflicts):continue
        coverage=top_count/len(scoped)
        for row in conflicts:
            key=row.get('point_key') or row['name'];chosen=choices[key]
            variants=[dict(o) for o in row.get('occurrences',[])]
            row['source_coordinate_variants']=variants
            row['xyz']=list(chosen['xyz']);row['pages']=[page];row['occurrences']=[dict(chosen)]
            row['coordinate_conflict']=False
            row['semantic_resolution']='dominant_source_coordinate_definition_page'
            row['coordinate_page_resolution']=dict(selected_page=page,program=program,
                coordinate_system=frame,page_group_coverage=coverage,
                alternative_occurrences=variants)
        evidence.append(dict(selected_page=page,program=program,coordinate_system=frame,
            covered_point_count=top_count,source_point_count=len(scoped),page_group_coverage=coverage,
            resolved_point_names=sorted(r['name'] for r in conflicts),
            source_variants_preserved=True))
    return resolved,evidence


def extract_named_points(pages):
    """Return named XYZ points with exact source page/FAI provenance."""
    result = {}
    from .msop_tables import feature_coordinate_rows
    def entries(page):
        values=[(x,y,z,name,None) for x,y,z,name in COORDINATE_POINT.findall(page.get('text',''))]
        values.extend((*row['values'],row['name'],row) for row in feature_coordinate_rows(page)
                      if row['axes']==['X','Y','Z'])
        return values
    names={v[3] for page in pages for v in entries(page)}
    scoped_pages=contextual_pages(pages,names)
    for page in scoped_pages:
        assigned=explicit_point_frames(page,names)
        for x, y, z, name, table_row in entries(page):
            resolution_xyz = [10.0 ** -len(token.split('.', 1)[1]) if '.' in token else 1.0
                              for token in (x, y, z)]
            result.setdefault(name, {
                "name": name,
                "xyz": [float(x), float(y), float(z)],
                "pages": [],
                "fai_ids": [],
                "occurrences": [],
            })
            row = result[name]
            inherited = page.get('inherited_context') or {}
            coordinate_system = page.get('measured_system')
            if 'measured_system' in inherited and page.get('dimension_system'):
                coordinate_system = page['dimension_system']
            if name in assigned:
                # Explicit point-scoped system blocks outrank page-wide context.
                # A conflict remains unresolved; never fall back to the page's
                # first/global frame and silently make it appear certain.
                coordinate_system = assigned[name].get('coordinate_system')
            occurrence={"xyz":[float(x),float(y),float(z)],"page":page.get("page"),"coordinate_system":coordinate_system,"program":page.get('program'),"coordinate_resolution_xyz_mm":resolution_xyz}
            if name in assigned:occurrence.update(assigned[name])
            if table_row:
                occurrence.update(coordinate_system=table_row.get('coordinate_system') or coordinate_system,
                    coordinate_frame_source=table_row.get('coordinate_frame_source'),
                    source_table_axes=table_row['axes'],source_table_row=table_row['original_row'])
            if occurrence not in row['occurrences']:row['occurrences'].append(occurrence)
            row['coordinate_conflict']=any(v.get('frame_assignment_conflict') for v in row['occurrences']) or any(a['coordinate_system']==b['coordinate_system'] and a.get('program')==b.get('program') and not np.allclose(a['xyz'],b['xyz'],atol=1e-6,rtol=0)
                                           for a in row['occurrences'] for b in row['occurrences'])
            row['scope_ambiguous']=len(scope_rows(row))>1
            if page.get("page") not in row["pages"]:
                row["pages"].append(page.get("page"))
            for number in page.get("fai_ids", []):
                if number not in row["fai_ids"]:
                    row["fai_ids"].append(number)
    for point_set in _constructed_point_sets(scoped_pages, result):
        for name in point_set["point_names"]:
            result[name].setdefault("constructed_point_sets", []).append(point_set)
    return result


def semantic_point_references(features, named_points):
    """Find cross-page point labels sharing a specific non-generic token.

    Example: ``CIR_N3_VISION`` links to ``N3_PNT1`` ... ``N3_PNT6``. Ambiguous
    generic tokens are ignored; only the strongest overlap is returned.
    """
    refs = {}
    for feature in features:
        tokens = set(re.split(r"[_\W]+", feature.upper())) - {"", "VISION", "FEATURE", "DATUM"}
        feature_parts = feature.upper().split("_")
        feature_group = feature_parts[0] if feature_parts else ""
        feature_kind = next((part for part in feature_parts if part in GEOMETRY_TOKENS), None)
        scored = []
        for name, row in named_points.items():
            point_parts = name.upper().split("_")
            point_tokens = set(re.split(r"[_\W]+", name.upper())) - {"", "VISION", "FEATURE", "DATUM"}
            score = 0
            if name.upper() == feature.upper():
                score += 100
            if feature_group and point_parts and feature_group == point_parts[0]:
                score += 20
            if feature_kind and feature_kind in point_tokens and feature_group and point_parts and feature_group == point_parts[0]:
                score += 4
            # A shared point ordinal (for example P3) is not evidence that
            # R1_PNT_P3 and R3_PNT_P3 are the same feature. This previously
            # attached cross-family XYZ tables to the wrong FAI.
            shared = (tokens & point_tokens) - GENERIC_FEATURE_TOKENS
            shared = {token for token in shared if not POINT_ORDINAL.fullmatch(token)}
            score += 2 * len(shared)
            if score:
                scored.append((score, name, row))
        if scored:
            best = max(score for score, _, _ in scored)
            refs[feature] = [row for score, _, row in scored if score == best]
    return refs


def extract_feature_labels(pages, existing=()):
    """Extract geometric identifiers even when their type token is not first.

    Handles common forms such as CIR_N3_VISION, F1_CIR_L, F2_PNT_Z1,
    DATUMC_PNT_P03, and STANDOFF_U18 without drawing-specific vocabularies.
    """
    labels = set(existing or ())
    for page in pages:
        for identifier in IDENTIFIER.findall(page.get("text", "")):
            tokens = set(identifier.upper().split("_"))
            if tokens & GEOMETRY_TOKENS or re.fullmatch(r"(?:STANDOFF|CHASSIS)_U\d+", identifier.upper()):
                labels.add(identifier.upper())
    return sorted(labels)


def referenced_points(pages, method_pages, named_points, feature_labels=()):
    """Resolve exact and abbreviated point names against the full MSOP corpus."""
    method_text = "\n".join(page.get("text", "") for page in method_pages).upper()
    selected = {}
    for name, row in named_points.items():
        if re.search(rf"(?<![A-Z0-9_]){re.escape(name)}(?![A-Z0-9_])", method_text):
            selected[name] = row

    # Printed point ranges and comma lists use the last numeric component,
    # including suffixes: E04_OUT~E07_OUT, W04_PNT_Z1,Z3, DATUMB_E09,E10.
    # Expand only existing definitions in the same family, never invent points.
    numeric = re.compile(r'^(.*?)(\d+)([A-Z_]*)$')
    for base_name in named_points:
        family = numeric.fullmatch(base_name)
        if not family: continue
        prefix, start, suffix = family.groups()
        for match in re.finditer(rf'(?<![A-Z0-9_]){re.escape(base_name)}(?![A-Z0-9_])', method_text):
            tail = method_text[match.end():]
            current = int(start)
            for _ in range(128):
                part = re.match(r'\s*(~|–|—|TO|,|/)\s*([A-Z0-9_]+)', tail)
                if not part: break
                operator, token = part.groups()
                candidate = numeric.fullmatch(token)
                if not candidate: break
                token_prefix, end, token_suffix = candidate.groups()
                if token_prefix not in (prefix, prefix.rsplit('_', 1)[-1], '') or token_suffix not in (suffix, ''):
                    break
                end = int(end)
                numbers = range(min(current, end), max(current, end)+1) if operator in ('~','–','—','TO') and abs(end-current)<=1000 else [end]
                for name, row in named_points.items():
                    value = numeric.fullmatch(name)
                    if value and value[1] == prefix and value[3] == suffix and int(value[2]) in numbers:
                        selected[name] = row
                current = end
                tail = tail[part.end():]

    # Resolve common MSOP shorthand such as DATUMC_P04/03/47/46, P22/23/27/28.
    families = {name.split("_PNT_", 1)[0] for name in named_points if "_PNT_" in name}
    for family in families:
        family_pattern = re.compile(rf"\b{re.escape(family)}_P(\d+(?:/\d+)*)", re.I)
        for match in family_pattern.finditer(method_text):
            codes = match.group(1).split("/")
            tail = method_text[match.end():match.end() + 100]
            shorthand = re.match(r"\s*,\s*P(\d+(?:/\d+)*)", tail)
            if shorthand:
                codes.extend(shorthand.group(1).split("/"))
            normalized = {f"P{int(code):02d}" for code in codes}
            for name, row in named_points.items():
                if name.startswith(family + "_PNT_") and name.rsplit("_", 1)[-1] in normalized:
                    selected[name] = row

    method_programs = {str(page.get('program')).upper() for page in method_pages
                      if page.get('program')}
    for feature, rows in semantic_point_references(feature_labels, named_points).items():
        # Semantic fallback may cross MSOP pages, but it may not cross UP/DOWN
        # programs. A shared part token alone cannot identify a measurement
        # point from the opposite setup.
        if method_programs:
            rows = [row for row in rows if any(
                str(occurrence.get('program') or '').upper() in method_programs
                for occurrence in row.get('occurrences', []))]
            if not rows:
                continue
        # Explicit point names/ranges already define this feature's local set.
        # A generic construction such as EDGE_LIN must not expand three
        # explicitly measured EDGE_Pxx points into every EDGE point in MSOP.
        # Semantic family retrieval is only a fallback for otherwise unlocated
        # features (e.g. CIR_N3_VISION with definitions on another page).
        if any(row['name'] in selected for row in rows):
            continue
        feature_parts = set(feature.upper().split('_'))
        feature_kind = next((part for part in feature.upper().split('_')
                             if part in GEOMETRY_TOKENS), None)
        wanted_kind = 'line' if feature_kind == 'LIN' else 'plane' if feature_kind in ('CIR', 'PLN') else None
        associated_sets = {}
        if wanted_kind:
            for point_name, point_row in named_points.items():
                if point_name not in {row['name'] for row in rows}:
                    continue
                for point_set in point_row.get('constructed_point_sets', []):
                    if (point_set.get('family') in feature_parts
                            and point_set.get('kind') == wanted_kind):
                        key = tuple(sorted(point_set['point_names']))
                        associated_sets[key] = associated_sets.get(key, 0) + 1
        if associated_sets:
            # Repeated declarations across method pages strengthen a point-set
            # link. A circle uses the plane-construction set for its family;
            # it must not absorb neighboring line or unrelated point groups.
            best_set = min(associated_sets, key=lambda key: (-associated_sets[key], -len(key), key))
            rows = [row for row in rows if row['name'] in best_set]
            association_pages = sorted({source_page
                for point_name, point_row in named_points.items() if point_name in best_set
                for point_set in point_row.get('constructed_point_sets', [])
                if tuple(sorted(point_set.get('point_names', []))) == best_set
                and point_set.get('family') in feature_parts
                and point_set.get('kind') == wanted_kind
                for source_page in point_set.get('coordinate_pages', [])})
            rows = [dict(row, semantic_source_pages=association_pages,
                         semantic_feature=feature) for row in rows]
        for row in rows:
            selected[row["name"]] = row
    resolved = {}
    for row in selected.values():
        scoped = resolve_scopes(row, method_pages)
        for value in scoped:
            resolved[value['name'] if len(scoped)==1 else value['point_key']] = value
    return resolved


def nearest_xy(vertices, face_ids, x, y, chunk_size=12000):
    vertices = np.asarray(vertices, dtype=np.float32)
    face_ids = np.asarray(face_ids, dtype=object)
    target = np.asarray([x, y], dtype=np.float32)
    best_distance, best_index = float("inf"), -1
    for start in range(0, len(vertices), chunk_size):
        block = vertices[start:start + chunk_size, :2]
        d2 = np.sum((block - target) ** 2, axis=1)
        offset = int(np.argmin(d2))
        distance = float(np.sqrt(d2[offset]))
        if distance < best_distance:
            best_distance, best_index = distance, start + offset
    return best_distance, str(face_ids[best_index]), vertices[best_index].tolist()


def nearest_xy_triangle(triangles, face_ids, x, y, chunk_size=12000):
    """Project an XY coordinate onto the closest tessellated triangle.

    Unlike nearest_xy(), this does not snap the FAI point to a mesh vertex. The
    returned XYZ is interpolated on the selected triangle, so it remains a
    valid surface anchor while removing tessellation-dependent marker offsets.
    """
    tris = np.asarray(triangles, dtype=np.float32).reshape(-1, 3, 3)
    ids = np.asarray(face_ids, dtype=object)
    target = np.asarray([x, y], dtype=np.float32)
    best_d2, best_i, best_pos = float("inf"), -1, None
    for start in range(0, len(tris), chunk_size):
        block = tris[start:start + chunk_size]
        a, b, c = block[:, 0], block[:, 1], block[:, 2]
        ab, ac = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        ap = target - a[:, :2]
        d00 = np.einsum("ij,ij->i", ab, ab)
        d01 = np.einsum("ij,ij->i", ab, ac)
        d11 = np.einsum("ij,ij->i", ac, ac)
        d20 = np.einsum("ij,ij->i", ap, ab)
        d21 = np.einsum("ij,ij->i", ap, ac)
        denom = d00 * d11 - d01 * d01
        safe = np.where(np.abs(denom) > 1e-20, denom, 1.0)
        u = (d11 * d20 - d01 * d21) / safe
        v = (d00 * d21 - d01 * d20) / safe
        inside = (np.abs(denom) > 1e-20) & (u >= 0) & (v >= 0) & (u + v <= 1)
        projected = a + u[:, None] * (b - a) + v[:, None] * (c - a)
        d2 = np.sum((projected[:, :2] - target) ** 2, axis=1)
        d2[~inside] = np.inf
        # For exterior and XY-degenerate triangles, use closest point on each
        # projected edge, retaining its 3D interpolation parameter.
        for p0, p1 in ((a, b), (b, c), (c, a)):
            edge = p1[:, :2] - p0[:, :2]
            length2 = np.einsum("ij,ij->i", edge, edge)
            t = np.clip(np.einsum("ij,ij->i", target - p0[:, :2], edge) /
                        np.maximum(length2, 1e-20), 0, 1)
            edge_pos = p0 + t[:, None] * (p1 - p0)
            edge_d2 = np.sum((edge_pos[:, :2] - target) ** 2, axis=1)
            improve = edge_d2 < d2
            d2[improve] = edge_d2[improve]
            projected[improve] = edge_pos[improve]
        local = int(np.argmin(d2))
        if float(d2[local]) < best_d2:
            best_d2 = float(d2[local])
            best_i = start + local
            best_pos = projected[local].astype(float).tolist()
    if best_i < 0:
        return float("inf"), "", []
    return float(np.sqrt(best_d2)), str(ids[best_i]), best_pos


def screen_point_references(points, vertices, face_ids, transform,
                           max_xy_residual_mm=1.0, min_inlier_ratio=0.5,
                           min_inliers=2, triangles=None, triangle_face_ids=None):
    """Project coordinate points through a calibrated frame and retain inliers."""
    x_sign = float(transform.get("x_sign", 1.0))
    y_sign = float(transform.get("y_sign", 1.0))
    x_offset = float(transform.get("x_offset_mm", 0.0))
    y_offset = float(transform.get("y_offset_mm", 0.0))
    projected = []
    for point in points:
        x, y = point["xyz"][:2]
        mapped_xy = [x_sign * x + x_offset, y_sign * y + y_offset]
        if triangles is not None:
            distance, face_id, position = nearest_xy_triangle(
                triangles, triangle_face_ids if triangle_face_ids is not None else face_ids,
                mapped_xy[0], mapped_xy[1])
        else:
            distance, face_id, position = nearest_xy(
                vertices, face_ids, mapped_xy[0], mapped_xy[1])
        projected.append({"name": point["name"], "source_pages": point["pages"],
                          "source_xyz": point["xyz"], "mapped_xy": mapped_xy,
                          "residual_mm": distance, "face_id": face_id, "position": position})
    inliers = [row for row in projected if row["residual_mm"] <= max_xy_residual_mm]
    valid = len(inliers) >= min_inliers and len(inliers) / max(1, len(projected)) >= min_inlier_ratio
    return {"valid": valid, "projected": projected, "inliers": inliers,
            "inlier_count": len(inliers), "point_count": len(projected),
            "inlier_ratio": len(inliers) / max(1, len(projected)),
            "max_inlier_residual_mm": max((r["residual_mm"] for r in inliers), default=None)}


def project_points_to_common_face(points, triangles, triangle_face_ids, candidate_face_ids,
                                   max_xy_residual_mm=1.0, min_coverage=0.9):
    """Jointly place one measured surface-point set on a shared CAD face.

    MSOP points used to construct one plane are a group, even when their XY
    projections overlap multiple assembly layers. Score each face from the
    whole point set, then return anchors projected onto the best-supported
    common face. This is intentionally opt-in at the call site: independent
    features (for example a plane and a line) must remain separate groups.
    """
    points = list(points)
    face_ids = [str(face_id) for face_id in dict.fromkeys(candidate_face_ids)]
    if not points or not face_ids:
        return None
    triangles = np.asarray(triangles, dtype=np.float32).reshape(-1, 3, 3)
    triangle_face_ids = np.asarray(triangle_face_ids, dtype=object)
    candidates = []
    for face_id in face_ids:
        mask = triangle_face_ids == face_id
        if not np.any(mask):
            continue
        face_triangles = triangles[mask]
        face_triangle_ids = triangle_face_ids[mask]
        projected = []
        for point in points:
            distance, selected_face, position = nearest_xy_triangle(
                face_triangles, face_triangle_ids,
                point["mapped_xy"][0], point["mapped_xy"][1])
            projected.append({**point, "face_id": selected_face,
                              "position": position, "residual_mm": distance})
        inliers = [row for row in projected if row["residual_mm"] <= max_xy_residual_mm]
        coverage = len(inliers) / len(points)
        if coverage < min_coverage:
            continue
        residuals = [row["residual_mm"] for row in inliers]
        z_median = float(np.median([row["position"][2] for row in inliers]))
        candidates.append((coverage, float(np.median(residuals)),
                           float(np.mean(residuals)), face_id, inliers, z_median))
    if not candidates:
        return None
    # Prefer full-group support first; residual then resolves overlapping
    # coplanar/stacked candidates without favoring arbitrary mesh face order.
    coverage, median_residual, mean_residual, face_id, inliers, z_median = min(
        candidates, key=lambda row: (-row[0], row[1], row[2], row[3]))
    z_values = [float(row["position"][2]) for row in inliers]
    alternatives = [
        {"face_id": row[3], "coverage": row[0], "median_residual_mm": row[1],
         "mean_residual_mm": row[2], "median_z_mm": row[5]}
        for row in candidates if row[3] != face_id and row[0] == coverage
        and abs(row[1] - median_residual) <= 0.03
        and abs(row[2] - mean_residual) <= 0.05
        and abs(row[5] - z_median) > 0.1
    ]
    return {"face_id": face_id, "inliers": inliers,
            "point_count": len(points), "inlier_count": len(inliers),
            "coverage": coverage, "median_residual_mm": median_residual,
            "mean_residual_mm": mean_residual, "median_z_mm": z_median,
            "z_span_mm": max(z_values) - min(z_values),
            "ambiguous": bool(alternatives), "alternatives": alternatives}


def group_points_by_page_system(pages, named_points):
    """Collect explicit point names used by each MSOP method page."""
    by_page = defaultdict(list)
    for page in pages:
        names = {match[3] for match in COORDINATE_POINT.findall(page.get("text", ""))}
        for name in names:
            if name in named_points:
                row = dict(named_points[name])
                row["method_page"] = page.get("page")
                by_page[page.get("page")].append(row)
    return by_page
