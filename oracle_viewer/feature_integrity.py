"""Source-scoped feature identities and cross-FAI checks; never choose CAD points."""
from __future__ import annotations
import hashlib
import json
import math
import re

VERSION = 1
# Display-anchor consistency only; this is not a measurement tolerance.
DISPLAY_ANCHOR_GATE_MM = .75
_NAME = re.compile(r'\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b')


def printed_feature_names(methods):
    from .local_matching import _declared_marker_manifest
    groups=_declared_marker_manifest(methods) or []
    expanded={name for group in groups for name in group['source_point_names']}
    return sorted(expanded|{name for page in methods for name in _NAME.findall(page.get('text','').upper())
                   if not name.startswith(('ALI_', 'MASTER_'))})


def feature_links(item, detail, anchors, locations, assignments, mesh):
    """Validate exact model-selected source names, scope and physical instance."""
    methods=detail.get('methods') or []
    allowed=set(printed_feature_names(methods))
    faces={f['id']:f for f in mesh['faces']}
    if not isinstance(assignments,list) or len(assignments)!=len(anchors):
        return [],['every_marker_needs_source_entity_assignment']
    by_id={}
    for row in assignments:
        if not isinstance(row,dict) or type(row.get('marker_id')) is not int:return [],['invalid_marker_assignment']
        if row['marker_id'] in by_id:return [],['duplicate_marker_assignment']
        by_id[row['marker_id']]=row
    if set(by_id)!=set(range(1,len(anchors)+1)):return [],['incomplete_marker_assignments']
    from .local_matching import _declared_marker_manifest
    groups=_declared_marker_manifest(methods) or []
    links=[];errors=[]
    for index,(anchor,location) in enumerate(zip(anchors,locations),1):
        row=by_id[index];names=row.get('source_names');instance=faces[anchor['face_id']].get('instance_path',[])
        if (not isinstance(names,list) or not all(isinstance(n,str) and n in allowed for n in names)
                or len(set(names))!=len(names) or (allowed and not names)):
            errors.append(f'{index}:source_names_not_exact_printed_features');continue
        expected=set(location.get('source_point_names') or []) & allowed
        if location.get('feature') in allowed:expected.add(location['feature'])
        if expected and set(names)!=expected:
            errors.append(f'{index}:review_cannot_rename_proposed_source_feature');continue
        if row.get('instance_path')!=instance or not row.get('evidence'):
            errors.append(f'{index}:instance_or_source_evidence_unconfirmed');continue
        for name in names:
            pages=[p for p in methods if name in _NAME.findall(p.get('text','').upper()) or any(
                name in group['source_point_names'] and group['source_page']==p.get('page') for group in groups)]
            scopes=set()
            for page in pages:
                frames=re.findall(r'\bOutput\s+system\s*:?\s*(ALI_[A-Z0-9_]+(?:\s*&\s*ALI_[A-Z0-9_]+)*)',page.get('text',''),re.I)
                frame=page.get('measured_system') or (re.sub(r'\s*&\s*','&',frames[0].upper()) if len(set(frames))==1 else None)
                scopes.add((page.get('program'),frame))
            # Ambiguous scopes cannot share a global identity by name alone.
            if len(scopes)!=1:
                errors.append(f'{index}:source_feature_scope_ambiguous');continue
            scope=next(iter(scopes))
            key=json.dumps([item.get('document_id'),*scope,([] if scope[1] else sorted(p['page'] for p in pages)),name],ensure_ascii=False,separators=(',',':'))
            kind=('point' if re.search(r'(?:^|_)(?:P\d+|PNT\d*|POINT\d*)$',name) else
                  'circle' if re.search(r'(?:^|_)(?:CIR|CIRCLE)(?:_|\d|$)',name) else 'entity')
            links.append(dict(feature_id=hashlib.sha256(key.encode()).hexdigest(),source_name=name,
                program=scope[0],coordinate_system=scope[1],source_pages=[p['page'] for p in pages],
                marker_id=index,kind=kind,position=anchor['position'],face_id=anchor['face_id'],
                instance_path=instance,source_evidence=row['evidence']))
    return links,errors


def consistency_conflicts(project, fai, links):
    """Reject incompatible peers symmetrically; no peer is a preferred location."""
    conflicts=[]
    for key,annotation in project.get('annotations',{}).items():
        if key.rsplit(':',1)[-1]==str(fai) or annotation.get('status')!='auto_marked':continue
        verification=annotation.get('feature_verification') or {}
        if (verification.get('integrity_version')!=VERSION or annotation.get('source_sha256')!=(project.get('model') or {}).get('source_sha256')):continue
        for current in links:
            for peer in verification.get('feature_links',[]):
                if current['feature_id']!=peer['feature_id']:continue
                distance=math.dist(current['position'],peer['position'])
                different_instance=current['instance_path']!=peer['instance_path']
                # Lines/planes may have several valid sampling points. Only
                # unique named points/circle anchors use the display distance gate.
                unique=sum(x['feature_id']==current['feature_id'] for x in links)==1 and sum(
                    x['feature_id']==peer['feature_id'] for x in verification.get('feature_links',[]))==1
                if different_instance or (unique and current['kind'] in ('point','circle') and distance>DISPLAY_ANCHOR_GATE_MM):
                    conflicts.append(dict(source_name=current['source_name'],peer_fai_key=key,
                        marker_id=current['marker_id'],distance_mm=distance,different_instance=different_instance,
                        proposed_position=current['position'],peer_position=peer['position'],
                        peer_instance_path=peer['instance_path'],
                        reason='shared_source_feature_has_incompatible_locations'))
    return conflicts
