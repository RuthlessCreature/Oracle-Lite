"""Conservative swept-volume screening for model-assigned camera exposures.

Candidate order is a distance heuristic, not a controller program. A clear
envelope is conditional on supplied geometry and does not validate mechanics.
"""
from __future__ import annotations
import math
import numpy as np
from .physical import solids, triangles_hit_box


def pose_bounds(head, components=("lens", "camera")):
    low, high = [], []
    for solid in solids(head['scene']):
        if solid['id'] not in components:continue
        center = solid['origin']+solid['axis']*solid['height']/2
        half = np.r_[solid['half'],solid['height']/2] if solid['shape']=='box' else np.array([solid['outer'],solid['outer'],solid['height']/2])
        extent = np.abs(solid['basis']).T@half
        low.append(center-extent);high.append(center+extent)
    return np.min(low,axis=0),np.max(high,axis=0)


def candidate_order(head_ids,heads,home,center_key="camera_flange_mm"):
    remaining=set(head_ids);order=[];cursor=home
    while remaining:
        here=np.asarray(heads[cursor]['scene'][center_key])
        chosen=min(remaining,key=lambda h:(float(np.linalg.norm(np.asarray(heads[h]['scene'][center_key])-here)),h))
        remaining.remove(chosen);order.append(chosen);cursor=chosen
    return order


def screen_motion(camera_units,heads,mesh,design,light_units=()):
    limits=design.get('minimum_clearance_mm',2);rows=[]
    face_data=[]
    for f in (mesh or {}).get('faces',[]):
        points=np.asarray(f['positions'],float).reshape(-1,3)
        if len(points):face_data.append((f['id'],points,f.get('indices'),points.min(0),points.max(0)))
    moving_lights=[dict(id=l['id'],head_ids=l['head_ids'],home_head_id=l['home_head_id'],motion_mode='move_light',component_kind='light')
                   for l in light_units if l.get('mode')=='existing_moving_light' and l.get('head')]
    fixed_bodies=[(c['id'],pose_bounds(heads[c['home_head_id']])) for c in camera_units]
    fixed_bodies += [(l['id'],pose_bounds(l['head'],('light',))) for l in light_units if l.get('head')]
    for camera in [*camera_units,*moving_lights]:
        is_light=camera.get('component_kind')=='light'
        components=('light',) if is_light else ('lens','camera')
        center_key='light_face_mm' if is_light else 'camera_flange_mm'
        order=candidate_order(camera['head_ids'],heads,camera['home_head_id'],center_key)
        if camera.get('motion_mode')=='move_part':
            rows.append(dict(camera_id=camera['id'],candidate_order=order,legs=[],translation_total_mm=None,rotation_required=camera.get('rotation_required'),order_authority='relative_pose_order_only',status='requires_workpiece_and_fixture_transform',executable=False))
            continue
        pairs=list(zip([camera['home_head_id']]+order,order))
        legs=[]
        for start,end in pairs:
            if start==end:continue
            a,b=heads[start],heads[end];lo1,hi1=pose_bounds(a,components);lo2,hi2=pose_bounds(b,components)
            n1=np.asarray(a['scene']['light_direction' if is_light else 'normal']);n2=np.asarray(b['scene']['light_direction' if is_light else 'normal'])
            u1=np.asarray(a['scene'].get('basis_x',[1,0,0]));u2=np.asarray(b['scene'].get('basis_x',[1,0,0]))
            if is_light:u1=n1;u2=n2
            rotation=max(math.degrees(math.acos(float(np.clip(n1@n2,-1,1)))),math.degrees(math.acos(float(np.clip(u1@u2,-1,1)))))
            lo=np.minimum(lo1,lo2)-limits;hi=np.maximum(hi1,hi2)+limits
            # Rotation can sweep outside endpoint boxes. Enclose the entire
            # camera/lens in a sphere about the flange for all orientations.
            if rotation>1e-4:
                import itertools
                radius=max(max(np.linalg.norm(np.asarray(c)-np.asarray(h['scene'][center_key]))
                               for c in itertools.product(*zip(*pose_bounds(h,components)))) for h in (a,b))
                centers=np.asarray([a['scene'][center_key],b['scene'][center_key]])
                lo=centers.min(0)-radius-limits;hi=centers.max(0)+radius+limits
            box=dict(origin=np.array([*(lo[:2]+hi[:2])/2,lo[2]]),basis=np.eye(3),half=(hi[:2]-lo[:2])/2,height=float(hi[2]-lo[2]))
            conflicts=[]
            for fid,points,indices,flo,fhi in face_data:
                if np.any(fhi<lo) or np.any(flo>hi):continue
                if indices and triangles_hit_box(points,indices,box):
                    conflicts.append(dict(object='workpiece',face_id=fid));break
            for z in design.get('forbidden_zones',[]):
                if np.all(np.asarray(z['max_mm'])>lo) and np.all(np.asarray(z['min_mm'])<hi):conflicts.append(dict(object=z['name']))
            for other_id,(ol,oh) in fixed_bodies:
                if other_id==camera['id']:continue
                if np.all(oh>lo) and np.all(ol<hi):conflicts.append(dict(object=other_id))
            legs.append(dict(start_head_id=start,end_head_id=end,translation_mm=float(np.linalg.norm(np.asarray(a['scene'][center_key])-b['scene'][center_key])),
                             rotation_deg=rotation,swept_bounds_mm=[*lo.tolist(),*hi.tolist()],
                             status='potential_interference' if conflicts else 'clear_supplied_envelope',conflicts=conflicts))
        rows.append(dict(camera_id=camera['id'],component_kind='light' if is_light else 'camera',candidate_order=order,legs=legs,
                         translation_total_mm=sum(l['translation_mm'] for l in legs),
                         rotation_required=any(l['rotation_deg']>1e-4 for l in legs),
                         order_authority='distance_heuristic_within_local_model_owned_exposures',
                         executable=False))
    return dict(method='conservative_translation_AABB_and_rotation_sphere_sweep',cameras=[r for r in rows if r.get('component_kind')!='light'],lights=[r for r in rows if r.get('component_kind')=='light'],
                simultaneous_motion_validated=False,
                static_light_ids=[l['id'] for l in light_units if l.get('head') and l.get('mode')!='existing_moving_light'],
                potential_interference_count=sum(bool(l['conflicts']) for r in rows for l in r['legs']),
                unresolved_unit_ids=[r['camera_id'] for r in rows if r.get('status')=='requires_workpiece_and_fixture_transform'],
                path_validated=False, fixture_complete=False, controller_program_generated=False,
                limitations=['保守包络相交表示需要改路径，不等于精确实体碰撞','尚缺完整支架、夹具、线缆和运动轴约束','候选顺序未替代MSOP基准先后关系或设备控制程序','相机与光源分别筛查，尚未验证同步、机械联动或并行动作'])
