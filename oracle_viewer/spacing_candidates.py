"""Complete spacing hypotheses derived from a local model's CAD picks.

No hypothesis is a placement until the local model reviews and selects it.
Source XYZ are never changed: this path is exclusively for no-XYZ locations.
"""
from __future__ import annotations

import numpy as np

from .cad_view_atlas import cached_surface_geometry,project_point_to_view_uv,ray_hit_view


def evenly_spaced_candidates(mesh,anchors,view_id,count,*,max_projection_mm=.15):
    if type(count) is not int or not 3<=count<=64 or len(anchors)<3:return []
    points=np.asarray([row['position'] for row in anchors],float)
    if not np.isfinite(points).all():return []
    distance=np.linalg.norm(points[:,None]-points[None,:],axis=2)
    i,j=np.unravel_index(int(np.argmax(distance)),distance.shape)
    if distance[i,j]<.2:return []
    line=np.linspace(points[i],points[j],count)
    proposals=[('LINE',line,{'basis':'farthest pair of model-selected CAD points'})]
    # A few near-surface pixel picks may hit different thickness levels of a
    # rail. Offer whole lines on planes that the model actually selected,
    # never relocate each point independently to its own layer.
    face_map={row['id']:row for row in mesh.get('faces',[])}
    planes=[]
    for face_id in dict.fromkeys(row['face_id'] for row in anchors):
        face=face_map.get(face_id,{})
        if face.get('surface')!='plane':continue
        vertices=np.asarray(face.get('positions',[]),float).reshape(-1,3)
        indices=np.asarray(face.get('indices',[]),int)
        if len(indices)<3:continue
        triangles=vertices[indices[:len(indices)//3*3].reshape(-1,3)]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        areas=np.linalg.norm(normals,axis=1);largest=int(np.argmax(areas))
        if areas[largest]<1e-10:continue
        normal=normals[largest]/areas[largest];origin=triangles[largest,0]
        if np.abs((vertices-origin)@normal).max()>.01:continue
        if any(abs(normal@old_n)>.99999 and abs((origin-old_o)@normal)<.01 for old_n,old_o in planes):continue
        planes.append((normal,origin))
        projected=line-((line-origin)@normal)[:,None]*normal
        if np.linalg.norm(projected-line,axis=1).max()>1.0:continue
        proposals.append(('PLANAR_LINE',projected,{'basis':'plane of a model-selected CAD face','face_id':face_id}))
        if len(planes)>=3:break
    # Closed circular layouts are another common interpretation of evenly
    # spaced spots. Only offer this when the model's picks define a stable
    # plane and circle; a straight rail cannot become an arbitrary huge arc.
    center=points.mean(axis=0);_u,singular,basis=np.linalg.svd(points-center,full_matrices=False)
    if singular[1]>.1 and singular[-1]<=.1*singular[1]:
        xy=(points-center)@basis[:2].T
        matrix=np.column_stack((2*xy,np.ones(len(xy))))
        fitted=np.linalg.lstsq(matrix,np.sum(xy**2,axis=1),rcond=None)[0]
        radius_sq=float(fitted[2]+np.sum(fitted[:2]**2))
        if radius_sq>0:
            radius=np.sqrt(radius_sq)
            radial=np.linalg.norm(xy-fitted[:2],axis=1)
            if np.max(np.abs(radial-radius))<=max(.15,.1*radius):
                phase=np.arctan2(xy[0,1]-fitted[1],xy[0,0]-fitted[0])
                angles=phase+np.arange(count)*2*np.pi/count
                ring_xy=fitted[:2]+radius*np.column_stack((np.cos(angles),np.sin(angles)))
                ring=center+ring_xy@basis[:2]
                proposals.append(('CIRCLE',ring,{'basis':'circle through model-selected CAD points'}))
    _triangles,face_ids,index=cached_surface_geometry(mesh)
    options=[]
    for kind,targets,provenance in proposals:
        result=[];errors=[]
        for point in targets:
            nearest=index.nearest(point)
            if nearest['distance_mm']>max_projection_mm:break
            position=np.asarray(nearest['point'],float)
            uv,_scale=project_point_to_view_uv(mesh,view_id,position)
            visible=ray_hit_view(mesh,view_id,*uv)
            if visible is None or np.linalg.norm(np.asarray(visible['position'])-position)>.05:break
            result.append({'face_id':str(face_ids[nearest['triangle_index']]),
                'position':position.tolist(),'label_offset':[32,-26]})
            errors.append(float(nearest['distance_mm']))
        if len(result)!=count:continue
        mapped=np.asarray([row['position'] for row in result])
        pairwise=np.linalg.norm(mapped[:,None]-mapped[None,:],axis=2)
        np.fill_diagonal(pairwise,np.inf);spacing=pairwise.min(axis=1)
        if spacing.min()<.1 or spacing.max()/spacing.min()>1.25:continue
        if any(np.allclose(mapped,np.asarray([row['position'] for row in old['anchors']]),atol=.01) for old in options):continue
        options.append({'candidate_id':f'SP{len(options)+1:02d}','pattern':kind,'anchors':result,
            'provenance':provenance,'count':count,'max_projection_mm':max(errors),
            'nearest_neighbor_ratio':float(spacing.max()/spacing.min()),
            'source_coordinates_available':False,'requires_local_model_selection':True})
    return options
