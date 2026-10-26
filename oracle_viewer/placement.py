"""Deterministic, evidence-tagged device pose proposals for station plans.

Pose proposals use CAD display coordinates and catalogue working-distance/FOV
fields. They are not installation acceptance: supplier mounting frames and
fixture geometry are often absent from the local catalogue.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _unit(v):
    length=math.sqrt(sum(float(x)*float(x) for x in v))
    return [float(x)/length for x in v] if length>1e-9 else None


def face_normals(mesh_file, wanted_face_ids):
    """Return robust averaged normals for referenced CAD faces in the mesh."""
    if not mesh_file or not wanted_face_ids or not Path(mesh_file).is_file():return {},None
    wanted=set(wanted_face_ids);out={};centers=[];bbox_center=None
    with Path(mesh_file).open() as f:data=json.load(f)
    bbox=data.get('bbox')
    if bbox and len(bbox)==6:bbox_center=[(bbox[i]+bbox[i+3])/2 for i in range(3)]
    for face in data.get('faces',[]):
        if face.get('id') not in wanted:continue
        import numpy as np
        p=np.asarray(face.get('positions',[]),dtype=float).reshape(-1,3)
        indices=np.asarray(face.get('indices',[]),dtype=int)
        if len(p)<3 or len(indices)<3:continue
        tri=p[indices.reshape(-1,3)];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1)
        valid=length>1e-10
        if not valid.any():continue
        normals=cross[valid]/length[valid,None]
        if face.get('surface')=='cylinder' and len(normals)>4:
            # Cylinder axis is the least-varying direction of local normals.
            _,singular,vh=np.linalg.svd(normals,full_matrices=False)
            n=vh[-1] if singular[-1]<singular[0]*.1 else normals.mean(axis=0)
        else:n=(normals*length[valid,None]).sum(axis=0)
        n=_unit(n)
        if n:
            center=face.get('center') or p.mean(axis=0).tolist()
            if bbox_center and sum(n[k]*(center[k]-bbox_center[k]) for k in range(3))<0:n=[-x for x in n]
            out[face['id']]=n
    return out,bbox_center


def _sensor_envelope(item):
    spec=item.get('spec') or {};kind=item.get('kind')
    if kind=='surface_sensor':
        near=spec.get('near_fov_mm');far=spec.get('far_fov_mm');wd=spec.get('recommended_working_distance_mm')
        if near and far and wd and len(near)==len(far)==len(wd)==2:
            return dict(working_distance_mm=float(sum(wd)/2),fov_mm=[(float(a)+float(b))/2 for a,b in zip(near,far)],basis='catalogue_near_far_FOV_and_working_distance')
    if kind=='lens' and spec.get('working_distance_mm'):
        return dict(working_distance_mm=float(spec['working_distance_mm']),basis='catalogue_lens_working_distance')
    if kind=='confocal_sensor' and spec.get('working_distance_mm'):
        return dict(working_distance_mm=float(spec['working_distance_mm']),basis='catalogue_confocal_working_distance')
    return None


def _cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def _dot(a,b):return sum(a[i]*b[i] for i in range(3))


def _quaternion_from_axes(local_axis,local_x,world_axis,world_x):
    """Quaternion mapping the verified CAD optical frame onto the target frame."""
    la=_unit(local_axis);lx=_unit([local_x[i]-la[i]*_dot(local_x,la) for i in range(3)])
    ly=_unit(_cross(la,lx));wa=_unit(world_axis);wx=_unit([world_x[i]-wa[i]*_dot(world_x,wa) for i in range(3)])
    wy=_unit(_cross(wa,wx))
    # R = [world basis] * [local basis]^T, stored row-major for matrix->quat.
    L=[lx,ly,la];W=[wx,wy,wa]
    m=[[sum(W[k][i]*L[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    tr=m[0][0]+m[1][1]+m[2][2]
    if tr>0:
        s=math.sqrt(tr+1)*2;q=[(m[2][1]-m[1][2])/s,(m[0][2]-m[2][0])/s,(m[1][0]-m[0][1])/s,.25*s]
    elif m[0][0]>m[1][1] and m[0][0]>m[2][2]:
        s=math.sqrt(1+m[0][0]-m[1][1]-m[2][2])*2;q=[.25*s,(m[0][1]+m[1][0])/s,(m[0][2]+m[2][0])/s,(m[2][1]-m[1][2])/s]
    elif m[1][1]>m[2][2]:
        s=math.sqrt(1+m[1][1]-m[0][0]-m[2][2])*2;q=[(m[0][1]+m[1][0])/s,.25*s,(m[1][2]+m[2][1])/s,(m[0][2]-m[2][0])/s]
    else:
        s=math.sqrt(1+m[2][2]-m[0][0]-m[1][1])*2;q=[(m[0][2]+m[2][0])/s,(m[1][2]+m[2][1])/s,.25*s,(m[1][0]-m[0][1])/s]
    norm=math.sqrt(sum(v*v for v in q));return [v/norm for v in q]


def _frame_pose(frame,world_axis,world_x,axis_point_world):
    q=_quaternion_from_axes(frame['optical_axis_local'],frame['image_x_axis_local'],world_axis,world_x)
    # Apply q to the source-CAD axis point, then translate so that it lands on
    # the specified physical interface/optical point in the assembly.
    x,y,z=frame['axis_point_local_mm'];qx,qy,qz,qw=q
    tx=2*(qy*z-qz*y);ty=2*(qz*x-qx*z);tz=2*(qx*y-qy*x)
    rotated=[x+qw*tx+(qy*tz-qz*ty),y+qw*ty+(qz*tx-qx*tz),z+qw*tz+(qx*ty-qy*tx)]
    return q,[axis_point_world[i]-rotated[i] for i in range(3)]


def plan_device_poses(setup, tasks, items, normals, bbox_center):
    """Create poses only where the device-local axis is tied to its exact STEP."""
    item_by_id={i['id']:i for i in items};anchors_by_fai={t['number']:t.get('anchors',[]) for t in tasks}
    anchors=[]
    for number in setup.get('fai_numbers',[]):
        for anchor in anchors_by_fai.get(number,[]):
            point=anchor.get('position');normal=normals.get(anchor.get('face_id'))
            if point and len(point)==3 and normal:anchors.append((number,[float(x) for x in point],normal))
    if not anchors:return dict(status='unresolved',poses=[],assemblies=[],unresolved_devices=list(setup.get('device_ids',[])),reason='FAI 点位没有可关联的有效 CAD 面法向，无法计算观察轴')
    clusters=[]
    for number,point,normal in anchors:
        cluster=next((c for c in clusters if sum(_unit(c['normal'])[k]*normal[k] for k in range(3))>=0.82),None)
        if cluster is None:cluster={'normal':[0.,0.,0.],'points':[],'fai_numbers':set()};clusters.append(cluster)
        cluster['points'].append(point);cluster['fai_numbers'].add(number)
        cluster['normal']=[cluster['normal'][k]+normal[k] for k in range(3)]
    candidates=[];assemblies=[];unresolved=[]
    for device_id in setup.get('device_ids',[]):
        item=item_by_id.get(device_id)
        if not item:unresolved.append(dict(device_id=device_id,reason='设备不在目录'));continue
        frame=item.get('cad_frame') or {};asset_hashes={a.get('sha256') for a in item.get('assets',[])}
        frame_valid=(frame.get('status','').startswith('verified_') and frame.get('source_sha256') in asset_hashes
                     and len(frame.get('optical_axis_local',[]))==3
                     and len(frame.get('axis_point_local_mm',[]))==3
                     and len(frame.get('image_x_axis_local',[]))==3)
        if item.get('kind')=='surface_sensor':
            envelope=_sensor_envelope(item)
            if not frame_valid or not envelope:
                unresolved.append(dict(device_id=device_id,reason='缺少与该STEP哈希绑定的光轴、轴上基准点、横滚基准或工作距/FOV依据'));continue
            for index,cluster in enumerate(clusters,1):
                n=_unit(cluster['normal']);points=cluster['points'];target=[sum(p[k] for p in points)/len(points) for k in range(3)]
                xaxis,yaxis,extents=_projected_extent(points,n)
                wd=envelope['working_distance_mm'];body=(item.get('spec') or {}).get('body_mm')
                depth=float(body[2]) if isinstance(body,list) and len(body)==3 else 0.
                axis_world=[-x for x in n];axis_point_world=[target[k]+n[k]*wd for k in range(3)]
                q,cad_origin=_frame_pose(frame,axis_world,xaxis,axis_point_world)
                candidates.append(dict(id=f"{setup['id']}:{device_id}:V{index}",device_id=device_id,
                    instance_id=(setup.get('device_instance_ids') or {}).get(device_id,setup['id']+':'+device_id),
                    fai_numbers=sorted(cluster['fai_numbers']),target_mm=target,position_mm=cad_origin,cad_origin_world_mm=cad_origin,
                    axis_point_world_mm=axis_point_world,quaternion_xyzw=q,
                    view_direction=axis_world,device_axis_local_mm=frame['optical_axis_local'],
                    working_distance_mm=wd,target_extent_mm=extents,fov_mm=envelope.get('fov_mm'),
                    basis_x_mm=xaxis,basis_y_mm=yaxis,pose_source=frame['status']+'; CAD face normals + catalogue WD/FOV',
                    cad_frame_source_sha256=frame['source_sha256'],pose_status='cad_axis_aligned_proposed',
                    coverage_status='within_catalogue_fov' if envelope.get('fov_mm') and extents[0]<=envelope['fov_mm'][0] and extents[1]<=envelope['fov_mm'][1] else 'fov_unverified_or_exceeded',
                    physical_validation='unverified_fixture_clearance_cable_and_metrology'))
        elif item.get('kind')=='camera' and setup.get('optical_pair'):
            if not frame_valid:
                unresolved.append(dict(device_id=device_id,reason='相机STEP缺少可核验的光轴、接口轴上基准点或图像横滚基准'));continue
            lenses=[item_by_id[d] for d in setup.get('device_ids',[]) if d in item_by_id and item_by_id[d].get('kind')=='lens']
            lens=lenses[0] if len(lenses)==1 else None;lspec=(lens or {}).get('spec') or {};cspec=item.get('spec') or {}
            wd=float(setup['optical_pair']['working_distance_mm']);lens_length=float(lspec.get('length_mm',0))
            if not lens or not lspec.get('length_mm'):
                unresolved.append(dict(device_id=device_id,reason='相机有CAD光轴，但光学总成缺镜头长度/配合接口资料'));continue
            for index,cluster in enumerate(clusters,1):
                n=_unit(cluster['normal']);points=cluster['points'];target=[sum(p[k] for p in points)/len(points) for k in range(3)]
                xaxis,yaxis,extents=_projected_extent(points,n);direction=[-x for x in n]
                lens_front=[target[k]+n[k]*wd for k in range(3)]
                lens_center=[target[k]+n[k]*(wd+lens_length/2) for k in range(3)]
                flange=[target[k]+n[k]*(wd+lens_length) for k in range(3)]
                q,cad_origin=_frame_pose(frame,direction,xaxis,flange)
                candidates.append(dict(id=f"{setup['id']}:{device_id}:V{index}",device_id=device_id,
                    instance_id=(setup.get('device_instance_ids') or {}).get(device_id,setup['id']+':'+device_id),
                    fai_numbers=sorted(cluster['fai_numbers']),target_mm=target,position_mm=cad_origin,cad_origin_world_mm=cad_origin,
                    axis_point_world_mm=flange,quaternion_xyzw=q,
                    view_direction=direction,device_axis_local_mm=frame['optical_axis_local'],working_distance_mm=wd,
                    target_extent_mm=extents,fov_mm=setup['optical_pair'].get('fov_mm'),image_size_px=setup['optical_pair'].get('roi_px'),basis_x_mm=xaxis,basis_y_mm=yaxis,
                    pose_source=frame['status']+' + C-mount optical pair geometry',cad_frame_source_sha256=frame['source_sha256'],
                    pose_status='cad_axis_aligned_proposed',coverage_status='within_optical_pair_fov' if setup['optical_pair'].get('fov_mm') and extents[0]<=setup['optical_pair']['fov_mm'][0] and extents[1]<=setup['optical_pair']['fov_mm'][1] else 'fov_unverified_or_exceeded',
                    physical_validation='unverified_mount_flange_fixture_clearance_and_metrology'))
                assemblies.append(dict(id=f"{setup['id']}:OPTICAL:{index}",fai_numbers=sorted(cluster['fai_numbers']),
                    axis_world=direction,lens_id=lens['id'],camera_id=device_id,lens_center_mm=lens_center,
                    camera_flange_mm=flange,target_mm=target,working_distance_mm=wd,lens_length_mm=lens_length,
                    mount_pair='C-mount',coaxiality_source='catalogue C-mount + datasheet front-to-flange length; lens CAD unavailable',
                    lens_front_mm=lens_front,lens_back_mm=flange,lens_axis_world=direction,
                    lens_outer_diameter_mm=float(lspec.get('front_diameter_mm',44)),
                    lens_render_status='datasheet_envelope_only_no_vendor_STEP',camera_pose_id=candidates[-1]['id']))
        else:
            unresolved.append(dict(device_id=device_id,reason='器件局部安装/光学轴未经该型号原始STEP核实'))
    status='proposed_unvalidated' if candidates and not unresolved else 'partial' if candidates else 'unresolved'
    return dict(status=status,poses=candidates,assemblies=assemblies,unresolved_devices=unresolved,
        limitations=['CAD姿态使用与原始STEP哈希绑定的光轴、轴上基准点和横滚基准；缺任一项即不生成姿态。','镜头位置使用C-mount和厂家标称前端至法兰长度；无镜头STEP时只显示明确标注的规格包络，不冒充厂家实体模型。','夹具、支架、螺钉、线缆与整机加载空间缺少完整CAD，安装避碰仍未验证。','FOV采用目录边界投影筛查，不能替代实测可视性、景深与精度验收。'])


def _projected_extent(points,normal):
    n=_unit(normal);reference=[1.,0.,0.] if abs(n[0])<.9 else [0.,1.,0.]
    xaxis=_unit([reference[k]-n[k]*sum(reference[j]*n[j] for j in range(3)) for k in range(3)])
    yaxis=[n[1]*xaxis[2]-n[2]*xaxis[1],n[2]*xaxis[0]-n[0]*xaxis[2],n[0]*xaxis[1]-n[1]*xaxis[0]]
    extents=[max(sum(p[k]*axis[k] for k in range(3)) for p in points)-min(sum(p[k]*axis[k] for k in range(3)) for p in points) for axis in (xaxis,yaxis)]
    return xaxis,yaxis,extents
