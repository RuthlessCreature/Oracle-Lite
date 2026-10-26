import pytest

from oracle_viewer.placement import _frame_pose, plan_device_poses


def _rotate(q, vector):
    x,y,z,w=q;vx,vy,vz=vector
    t=[2*(y*vz-z*vy),2*(z*vx-x*vz),2*(x*vy-y*vx)]
    c=[y*t[2]-z*t[1],z*t[0]-x*t[2],x*t[1]-y*t[0]]
    return [vector[i]+w*t[i]+c[i] for i in range(3)]


def test_cad_frame_transform_maps_optical_axis_and_interface_point_exactly():
    frame=dict(optical_axis_local=[0,0,-1],image_x_axis_local=[1,0,0],axis_point_local_mm=[0,0,-18.15])
    axis=[0,-1,0];point=[10,20,30]
    q,origin=_frame_pose(frame,axis,[1,0,0],point)
    assert _rotate(q,frame['optical_axis_local'])==pytest.approx(axis)
    transformed=[origin[i]+_rotate(q,frame['axis_point_local_mm'])[i] for i in range(3)]
    assert transformed==pytest.approx(point)


def test_pose_refuses_unverified_or_incomplete_vendor_axis_frame():
    task=dict(number=1,anchors=[dict(face_id='face',position=[1,2,3])])
    item=dict(id='CAM',kind='camera',cad_frame=dict(status='verified_test',source_sha256='abc',
        optical_axis_local=[0,0,-1]),assets=[dict(sha256='abc')])
    result=plan_device_poses(dict(id='U1',fai_numbers=[1],device_ids=['CAM'],optical_pair={'working_distance_mm':50}),[task],[item],{'face':[0,0,1]},[0,0,0])
    assert result['poses']==[]
    assert '基准点' in result['unresolved_devices'][0]['reason']


def test_camera_setup_returns_coaxial_datasheet_lens_envelope_not_fake_step():
    camera=dict(id='CAM',kind='camera',cad_frame=dict(status='verified_test',source_sha256='abc',
        optical_axis_local=[0,0,-1],image_x_axis_local=[1,0,0],axis_point_local_mm=[0,0,-18]),
        assets=[dict(sha256='abc')],spec=dict(body_mm=[30,30,50]))
    lens=dict(id='LENS',kind='lens',spec=dict(length_mm=100,front_diameter_mm=40))
    setup=dict(id='U1',fai_numbers=[1],device_ids=['CAM','LENS'],
        optical_pair=dict(working_distance_mm=60,fov_mm=[20,15]))
    task=dict(number=1,anchors=[dict(face_id='face',position=[0,0,0])])
    result=plan_device_poses(setup,[task],[camera,lens],{'face':[0,0,1]},[0,0,0])
    assert len(result['poses'])==1 and len(result['assemblies'])==1
    assembly=result['assemblies'][0]
    assert assembly['mount_pair']=='C-mount'
    assert assembly['lens_render_status']=='datasheet_envelope_only_no_vendor_STEP'
    assert assembly['lens_front_mm']==pytest.approx([0,0,60])
    assert assembly['lens_back_mm']==pytest.approx([0,0,160])
    assert result['poses'][0]['axis_point_world_mm']==pytest.approx(assembly['lens_back_mm'])
