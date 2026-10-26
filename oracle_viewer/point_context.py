"""Resolve MSOP point identities without crossing program or alignment scopes."""
from __future__ import annotations

import json
import re

import numpy as np


def explicit_point_frames(page, names):
    """Use point-local frame clauses rather than a page's first frame.

    Some MSOP pages switch between Measured System and Dimension System in the
    middle of one FAI. The following Location Definitions page has no frame
    header, so its point-to-frame assignments must be carried forward by point
    name. A page-level measured/dimension pair is not enough to scope every
    point on such a page.
    """
    found={}
    authority={}
    def split_frames(value):
        return [part.strip().upper() for part in re.split(r'\s*&\s*', value) if part.strip()]
    pattern=r'\bmeasure\b(?:(?!\bmeasure\b).)*?\bunder\s+(?:alignment\s+)?(ALI_[A-Z0-9_]+(?:\s*&\s*ALI_[A-Z0-9_]+)*)'
    from .candidate_mining import referenced_points
    flat={name:dict(name=name,xyz=[0,0,0],pages=[]) for name in names}
    for clause in re.finditer(pattern,page.get('text',''),re.I|re.S):
        frames=split_frames(clause.group(1))
        if not any('_' in frame for frame in frames):continue
        refs=referenced_points(None,[dict(text=clause.group(0))],flat)
        for name in refs:
            found.setdefault(name,set()).update(frames)
            authority.setdefault(name,set()).add('explicit_method_under_alignment_clause')

    # A method page can contain several explicit system blocks. Resolve point
    # labels only inside their own block; the next block changes the scope for
    # the operations that follow it. This handles pages where N2 points are
    # measured in one frame while N3/N4 are measured in another.
    system_pattern=re.compile(r'\b(Measured|Dimension)\s+System\s+(ALI_[A-Z0-9_]+(?:\s*&\s*ALI_[A-Z0-9_]+)*)',re.I)
    text=page.get('text','')
    blocks=list(system_pattern.finditer(text))
    for index,match in enumerate(blocks):
        end=blocks[index+1].start() if index+1<len(blocks) else len(text)
        block=text[match.end():end]
        block=re.split(r'(?:\bLocation Definitions\b|X,\s*Y,\s*Z\s*\(mm\))',
                       block,maxsplit=1,flags=re.I)[0]
        frames=split_frames(match.group(2))
        kind=match.group(1).lower()
        # Only points named by a Measure operation establish the point's
        # coordinate scope. A later dimension or location-definition mention
        # may refer to a point measured in a different coordinate system.
        for operation in re.finditer(r'\bMeasure\b[^.]*',block,re.I):
            refs=referenced_points(None,[dict(text=operation.group(0))],flat)
            for name in refs:
                found.setdefault(name,set()).update(frames)
                authority.setdefault(name,set()).add(f'explicit_{kind.lower()}_system_block')

    # Continuation/location pages inherit the preceding method page's explicit
    # per-point assignments, not one page-wide frame.
    inherited=page.get('inherited_point_frames') or {}
    for name,value in inherited.items():
        if name in flat and name not in found:
            frame=value.get('coordinate_system')
            if frame:
                found.setdefault(name,set()).add(str(frame).upper())
                authority.setdefault(name,set()).add(value.get('frame_authority','inherited_point_system_block'))
            elif value.get('frame_assignment_conflict'):
                found.setdefault(name,set()).update(value.get('coordinate_system_candidates') or {'__CONFLICT__'})
                authority.setdefault(name,set()).add(value.get('frame_authority','inherited_point_system_block'))

    result={}
    for name,frames in found.items():
        conflict=len(frames)!=1 or '__CONFLICT__' in frames
        result[name]=dict(coordinate_system=next(iter(frames)) if not conflict else None,
                          coordinate_system_candidates=sorted(frames),
                          frame_assignment_conflict=conflict,
                          frame_authority=';'.join(sorted(authority.get(name,()))))
    return result


def contextual_pages(pages, names=None):
    """Inherit page and point-local context on adjacent continuation pages."""
    result = []
    previous = None
    for original in pages:
        page = dict(original)
        same_fai = bool(set(page.get('fai_ids') or []) & set(previous.get('fai_ids') or [])) if previous else False
        same_heading = bool(page.get('heading') and page.get('heading') == previous.get('heading')) if previous else False
        if (previous and page.get('page') == previous.get('page', -2) + 1
                and (same_heading or same_fai)):
            for field in ('program', 'measured_system', 'dimension_system'):
                if not page.get(field) and previous.get(field):
                    page[field] = previous[field]
                    page.setdefault('inherited_context', {})[field] = previous['page']
            if names:
                frames=explicit_point_frames(previous,names)
                if frames:
                    page['inherited_point_frames']=frames
        result.append(page)
        previous = page
    return result


def point_key(name, program, frame):
    return json.dumps([name, program, frame], ensure_ascii=False, separators=(',', ':'))


def scope_rows(row):
    occurrences = row.get('occurrences', [])
    if not occurrences:
        return [row]
    buckets = {}
    for occurrence in occurrences:
        scope = (occurrence.get('program'), occurrence.get('coordinate_system'))
        buckets.setdefault(scope, []).append(occurrence)
    rows = []
    for (program, frame), values in buckets.items():
        deltas = [float(np.max(np.abs(np.asarray(v['xyz'], float) - np.asarray(values[0]['xyz'], float))))
                  for v in values[1:]]
        precisions = [v.get('coordinate_resolution_xyz_mm') for v in values]
        # When the only difference is no greater than the source's printed
        # coordinate resolution, retain every occurrence as provenance and
        # choose a deterministic displayed value. Larger differences remain
        # genuine conflicts and are never averaged away.
        component_deltas = [np.abs(np.asarray(v['xyz'], float) - np.asarray(values[0]['xyz'], float))
                            for v in values[1:]]
        precision_equivalent = (bool(component_deltas) and all(p is not None for p in precisions)
                                and all(np.all(np.asarray(p, float) <= .1 + 1e-9)
                                        for p in precisions)
                                and all(np.all(delta <= np.maximum(precisions[0], p) + 1e-9)
                                        for delta, p in zip(component_deltas, precisions[1:])))
        conflict = any(v.get('frame_assignment_conflict') for v in values) or any(
            np.any(delta > (np.maximum(precisions[0], p) + 1e-9
                            if precision_equivalent else 1e-6))
            for delta, p in zip(component_deltas, precisions[1:]))
        rows.append(dict(row, xyz=values[0]['xyz'], pages=list(dict.fromkeys(v['page'] for v in values)),
                         program=program, coordinate_system=frame, occurrences=values,
                         point_key=point_key(row['name'], program, frame),
                         coordinate_conflict=conflict, scope_ambiguous=False,
                         coordinate_precision_equivalent=precision_equivalent,
                         coordinate_variant_span_mm=max(deltas, default=0.0)))
    return rows


def resolve_scopes(row, methods):
    """Prefer explicit local definitions, then a matching program/alignment.

    An unresolved scope remains evidence but is blocked from geometric mapping.
    The first XYZ occurrence is never accepted simply because it came first.
    """
    candidates = scope_rows(row)
    page_numbers = {p.get('page') for p in methods}

    # A location definition printed on the same method page is a stronger
    # identity link than another occurrence of the same short point name in a
    # later setup. Use it only when that method page explicitly names the
    # point, so an unrelated coordinate table on a shared/continuation page
    # cannot silently erase a real conflict.
    explicit_local_pages = set()
    for page in methods:
        if page.get('page') not in page_numbers:
            continue
        text = page.get('text', '')
        if re.search(rf'(?<![A-Z0-9_]){re.escape(row["name"])}(?![A-Z0-9_])',
                     re.split(r'(?:\bLocation Definitions\b|X,\s*Y,\s*Z\s*\(mm\))',
                              text, maxsplit=1, flags=re.I)[0], re.I):
            explicit_local_pages.add(page.get('page'))
            continue
        # MSOP method pages often say only “measure N vector points” and put
        # the actual names in the page's Location Definitions table. That table
        # is authoritative for this FAI page; unrelated copies elsewhere are
        # retained only when no local definition exists.
        sections = re.split(r'\bLocation Definitions\b', text, maxsplit=1, flags=re.I)
        if len(sections) == 2 and re.search(
                rf'(?<![A-Z0-9_]){re.escape(row["name"])}(?![A-Z0-9_])', sections[1], re.I):
            explicit_local_pages.add(page.get('page'))
    if explicit_local_pages:
        local = []
        for candidate in candidates:
            occurrences = [o for o in candidate.get('occurrences', [])
                           if o.get('page') in explicit_local_pages]
            if not occurrences:
                continue
            # Multiple local definitions remain a conflict unless they agree
            # exactly; scope_rows applies that check to the filtered evidence.
            narrowed = dict(candidate, occurrences=occurrences,
                            pages=list(dict.fromkeys(o.get('page') for o in occurrences)),
                            xyz=occurrences[0]['xyz'])
            narrowed['coordinate_conflict'] = any(
                o.get('frame_assignment_conflict') for o in occurrences
            ) or any(not np.allclose(o['xyz'], occurrences[0]['xyz'], atol=1e-6, rtol=0)
                     for o in occurrences[1:])
            local.append(narrowed)
        if local:
            return local

    semantic_pages = set(row.get('semantic_source_pages', []))
    if semantic_pages:
        associated = []
        for candidate in candidates:
            occurrences = [o for o in candidate.get('occurrences', [])
                           if o.get('page') in semantic_pages]
            if not occurrences:
                continue
            xyz = np.asarray(occurrences[0]['xyz'], float)
            deltas = [np.abs(np.asarray(o['xyz'], float) - xyz) for o in occurrences[1:]]
            precisions = [o.get('coordinate_resolution_xyz_mm') for o in occurrences]
            precision_equivalent = (bool(deltas) and all(p is not None for p in precisions)
                and all(np.all(np.asarray(p, float) <= .1 + 1e-9) for p in precisions)
                and all(np.all(delta <= np.maximum(precisions[0], p) + 1e-9)
                        for delta, p in zip(deltas, precisions[1:])))
            conflict = any(o.get('frame_assignment_conflict') for o in occurrences) or any(
                np.any(delta > (np.maximum(precisions[0], p) + 1e-9
                                if precision_equivalent else 1e-6))
                for delta, p in zip(deltas, precisions[1:]))
            associated.append(dict(candidate, xyz=occurrences[0]['xyz'],
                pages=list(dict.fromkeys(o.get('page') for o in occurrences)),
                occurrences=occurrences, coordinate_conflict=conflict,
                coordinate_precision_equivalent=precision_equivalent,
                coordinate_variant_span_mm=max((float(np.max(d)) for d in deltas), default=0.0)))
        if associated:
            return associated

    own = [r for r in candidates if page_numbers.intersection(r.get('pages', []))]
    if own:
        return own
    programs = {p['program'] for p in methods if p.get('program')}
    frames = {s for p in methods for s in re.split(r'\s*[&,]\s*', p.get('measured_system') or '') if s}
    selected = [r for r in candidates if (not programs or r.get('program') in programs)
                and (not frames or r.get('coordinate_system') in frames)]
    if len(selected) == 1:
        return selected
    # A reference may deliberately use a different alignment within the same
    # program; preserve it only if that program has one unambiguous definition.
    same_program = [r for r in candidates if not programs or r.get('program') in programs]
    if not selected and len(same_program) == 1:
        return same_program
    if not programs and not frames and len(candidates) == 1:
        return candidates
    return [dict(row, scope_ambiguous=True, coordinate_conflict=True,
                 unresolved_scopes=[dict(program=r.get('program'), coordinate_system=r.get('coordinate_system'),
                                         pages=r.get('pages'), xyz=r.get('xyz')) for r in candidates],
                 point_key=point_key(row['name'], None, 'unresolved'))]
