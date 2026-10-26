"""Closest display-mesh point on an explicitly registered measurement plane.

A section can cross several CAD faces. It never implies a common face, or
supplies a missing source-to-CAD transform.
"""
from __future__ import annotations
import numpy as np
from .visibility import closest_surface

# The point's nearest CAD anchor is already gated to <=0.1 mm. Keep the
# source-section residual tighter than that surface-distance bound while
# allowing the small CAD/source-frame tilt seen in rounded MSOP XYZ tables.
# This is a pending-review display limit, never a metrology tolerance.
MAX_DISPLAY_SECTION_DEVIATION_MM = 0.08


def closest_plane_section(target, triangles, normal, offset, tolerance=1e-7):
    point=np.asarray(target,float);tri=np.asarray(triangles,float).reshape(-1,3,3)
    n=np.asarray(normal,float)
    if point.shape!=(3,) or n.shape!=(3,) or not np.isfinite(point).all() or not np.isfinite(tri).all() or not np.isfinite(n).all() or not np.isfinite(offset) or np.linalg.norm(n)<1e-12:
        raise ValueError('Finite XYZ, triangles and a nonzero plane normal are required')
    length=np.linalg.norm(n);n=n/length;offset=float(offset)/length
    if not len(tri):return None
    signed=tri@n-offset
    candidates=np.flatnonzero((signed.min(1)<=tolerance)&(signed.max(1)>=-tolerance))
    best=None
    for index in candidates:
        triangle=tri[index];dist=signed[index]
        if np.all(np.abs(dist)<=tolerance):
            hit=closest_surface(point,triangle[None,:,:])
            if hit is None:continue
            candidate=np.asarray(hit['point'])
        else:
            cuts=[]
            for a,b in ((0,1),(1,2),(2,0)):
                if abs(dist[a])<=tolerance:cuts.append(triangle[a]-dist[a]*n)
                if dist[a]*dist[b]<0:
                    cuts.append(triangle[a]+(triangle[b]-triangle[a])*dist[a]/(dist[a]-dist[b]))
            if not cuts:continue
            if len(cuts)==1:candidate=cuts[0]
            else:
                # Coincident vertex cuts are harmless; choose the longest
                # segment so a vertex-on-plane triangle retains its full cut.
                a,b=max(((a,b) for a in cuts for b in cuts),key=lambda pair:np.linalg.norm(pair[1]-pair[0]))
                v=b-a;den=float(v@v);candidate=a if den<1e-20 else a+v*np.clip(float((point-a)@v)/den,0,1)
        distance=float(np.linalg.norm(candidate-point))
        if best is None or distance<best['distance_mm']:
            best=dict(point=candidate.tolist(),distance_mm=distance,triangle_index=int(index),plane_residual_mm=float(abs(candidate@n-offset)))
    return best


def prefer_near_surface_projection(nearest, section, normal, offset, *,
                                   max_surface_distance_mm, max_section_deviation_mm=MAX_DISPLAY_SECTION_DEVIATION_MM):
    """Use the local surface hit when an exact triangle section is a distant artifact.

    Tessellated CAD surfaces can be slightly tilted relative to a source-Z
    plane. A long triangle may intersect that plane far from the point's local
    nearest hit. Permit the local hit only when its CAD surface distance and
    its signed distance from the transformed source plane are both tightly
    bounded. This changes display-anchor selection only; source XYZ and its
    shared plane remain untouched.
    """
    if not nearest or not np.isfinite(nearest.get('distance_mm', np.inf)):
        return section
    if nearest['distance_mm'] > max_surface_distance_mm:
        return section
    n=np.asarray(normal,float);length=float(np.linalg.norm(n))
    if not np.isfinite(n).all() or length<1e-12 or not np.isfinite(offset):
        raise ValueError('Finite plane normal and offset are required')
    residual=abs(float(np.asarray(nearest['point'],float)@n-float(offset)))/length
    if residual > max_section_deviation_mm:
        return section
    if section is not None and section.get('distance_mm',np.inf) <= nearest['distance_mm']:
        return section
    return dict(nearest,plane_residual_mm=residual,
                projection_method='nearest_surface_within_source_section_tolerance')


def source_plane_constraints(rows, methods, z_tolerance=.001):
    """Preserve constant-Z sampling sections without assuming one CAD face."""
    import re
    text='\n'.join(m.get('text','') for m in methods or [])
    flatness=bool(re.search(r'construct\s+(?:(?:a|one|1)\s+)?plane\b',text,re.I) and re.search(r'\bflatness\b',text,re.I))
    buckets={}
    for row in rows:
        if len(row.get('xyz',[]))==3 and np.isfinite(row['xyz']).all():
            buckets.setdefault((row.get('program'),row.get('coordinate_system')),[]).append(row)
    result=[]
    for (program,frame),points in buckets.items():
        groups=[]
        for row in sorted(points,key=lambda p:p['xyz'][2]):
            if not groups or abs(row['xyz'][2]-groups[-1][0]['xyz'][2])>z_tolerance:groups.append([row])
            else:groups[-1].append(row)
        for group in groups:
            # XYZ defines the source Z section directly. It remains valid for
            # a single point or a two-point line; requiring three samples would
            # silently drop the coplanarity constraint for those FAI groups.
            result.append(dict(kind='constructed_plane' if flatness else 'constant_source_z_section',program=program,coordinate_system=frame,normal_source=[0,0,1],offset_source_mm=group[0]['xyz'][2],point_keys=[p.get('point_key') or p['name'] for p in group],point_names=[p['name'] for p in group],source_pages=sorted({page for p in group for page in p.get('pages',[])}),scope_conflicting=any(p.get('coordinate_conflict') for p in group),common_cad_face_required=False,requires_source_to_cad_registration=True,authority='explicit_source_coordinate_table',metrology_accepted=False))
    return result
