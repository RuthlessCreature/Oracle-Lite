"""Geometry-only visual search options for thin CAD features.

No FAI identity or source-to-CAD correspondence is inferred here. Every
returned option is an exact visible ray hit awaiting local-model selection.
"""
from __future__ import annotations

import numpy as np

from .cad_view_atlas import (VIEWS, cached_surface_geometry,
    _view_projection_cache, _ray_hit_view_triangles)


def feature_view_candidates(mesh, view_id, uv, *, max_candidates=8,
                            max_uv_distance=.20, triangle_data=None, panel_size=480):
    """Probe projected triangle centers near a model's failed image point.

    Uniform image grids can miss an entire narrow rail. Triangle centers
    sample the actual surface instead of hoping a grid intersects it. The
    frontmost ray is rechecked, so hidden triangles cannot authorize hidden
    placements. Per-face seed diversity bounds large tessellated faces.
    """
    if view_id not in VIEWS or type(max_candidates) is not int or max_candidates<1:
        return []
    try:
        requested=np.asarray(uv,dtype=float)
    except (ValueError,TypeError):return []
    if requested.shape!=(2,) or not np.isfinite(requested).all() or np.any(requested<0) or np.any(requested>1):
        return []
    if not np.isfinite(max_uv_distance) or max_uv_distance<=0:return []
    triangles,face_ids=(triangle_data[:2] if triangle_data is not None
                        else cached_surface_geometry(mesh)[:2])
    projection=_view_projection_cache(triangles,view_id,panel_size,22)
    _normal,right,up,cr,cu,scale=projection[:6]
    centers=triangles.mean(axis=1)
    usable=panel_size-44
    coords=np.column_stack((((centers@right-cr)*scale+panel_size/2-22)/usable,
                            ((cu-centers@up)*scale+panel_size/2-22)/usable))
    distances=np.linalg.norm(coords-requested,axis=1)
    eligible=np.flatnonzero((distances<=max_uv_distance)&np.all((coords>=0)&(coords<=1),axis=1))
    if not len(eligible):return []
    eligible=eligible[np.argsort(distances[eligible],kind='stable')]
    # Keep the nearest triangle for every face first. Otherwise a large
    # nearby tessellation can exhaust the search before a tiny adjacent face.
    seeds=[];seen=set()
    for index in eligible:
        face=str(face_ids[index])
        if face in seen:continue
        seen.add(face);seeds.append(index)
        if len(seeds)>=64:break
    # More than one location on a long face can be needed by a multi-spot
    # group. Add a bounded spatially separated set from the nearby surface.
    nearest_spacing=np.full(len(eligible),np.inf)
    for old in seeds:
        nearest_spacing=np.minimum(nearest_spacing,np.sum((coords[eligible]-coords[old])**2,axis=1))
    while len(seeds)<96:
        separated=np.flatnonzero(nearest_spacing>=.012**2)
        if not len(separated):break
        index=eligible[separated[0]]
        seeds.append(index)
        nearest_spacing=np.minimum(nearest_spacing,np.sum((coords[eligible]-coords[index])**2,axis=1))
    hits=[]
    for index in seeds:
        u,v=map(float,coords[index])
        hit=_ray_hit_view_triangles(triangles,face_ids,view_id,u,v,
                                    panel_size=panel_size,projection=projection)
        if hit is None:continue
        if any(np.linalg.norm(np.asarray(hit['position'])-np.asarray(old['position']))<.1 for old in hits):
            continue
        hits.append({**hit,'candidate_source':'projected_CAD_triangle_center',
            'seed_face_id':str(face_ids[index]),'distance_from_requested_uv':float(distances[index])})
    if not hits:return []
    # Nearest alternatives first, then maximize spatial variety among the
    # remaining options. No first option is accepted without a model choice.
    hits.sort(key=lambda row:row['distance_from_requested_uv'])
    selected=list(range(min(3,max_candidates,len(hits))))
    uv_array=np.asarray([row['uv'] for row in hits])
    while len(selected)<min(max_candidates,len(hits)):
        spacing=np.min(np.linalg.norm(uv_array[:,None]-uv_array[selected][None,:],axis=2),axis=1)
        spacing[selected]=-1
        selected.append(int(np.argmax(spacing)))
    return [{**hits[index],'id':number} for number,index in enumerate(selected,1)]
