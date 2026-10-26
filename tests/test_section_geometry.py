import numpy as np
import pytest
from oracle_viewer.section_geometry import closest_plane_section, prefer_near_surface_projection
from oracle_viewer.surface_index import SurfaceIndex
from oracle_viewer.msop_coordinates import map_task_points
from oracle_viewer.msop import measurement_indicators


def test_section_crosses_triangle_interior_without_mesh_vertex_on_plane():
    tri=np.array([[[0,-2,-1],[0,2,-1],[0,0,1]]],float)
    hit=closest_plane_section([.05,.4,0],tri,[0,0,1],0)
    assert np.allclose(hit['point'],[0,.4,0])
    assert hit['distance_mm']==pytest.approx(.05)
    assert hit['plane_residual_mm']<1e-10
    assert closest_plane_section([0,0,3],tri,[0,0,1],3) is None


def test_section_handles_coplanar_triangle_and_invalid_normal():
    tri=[[[0,0,2],[2,0,2],[0,2,2]]]
    hit=closest_plane_section([.2,.3,2],tri,[0,0,2],4)
    assert np.allclose(hit['point'],[.2,.3,2])
    with pytest.raises(ValueError):closest_plane_section([0,0,0],tri,[0,0,0],0)


def test_near_surface_anchor_beats_distant_triangle_section_artifact():
    tri=np.array([[[0,0,0],[4,0,0],[4,20,.03]],[[0,0,0],[4,20,.03],[0,20,.03]]],float)
    target=np.array([2,10,0.])
    nearest=SurfaceIndex(tri).nearest(target)
    section=closest_plane_section(target,tri,[0,0,1],0)
    assert nearest['distance_mm']==pytest.approx(.015,abs=1e-6)
    assert section['distance_mm']>9
    chosen=prefer_near_surface_projection(nearest,section,[0,0,1],0,max_surface_distance_mm=.1)
    assert chosen['projection_method']=='nearest_surface_within_source_section_tolerance'
    assert chosen['distance_mm']==pytest.approx(.015,abs=1e-6)
    assert chosen['plane_residual_mm']==pytest.approx(.015,abs=1e-6)
    rejected=prefer_near_surface_projection(nearest,section,[0,0,1],0,max_surface_distance_mm=.1,
                                            max_section_deviation_mm=.01)
    assert rejected is section


def section_fixture(rotation=None,registered=True):
    r=np.eye(3) if rotation is None else np.asarray(rotation,float)
    points=np.array([[.2,0,0],[.8,0,0],[.2,2,0],[.8,2,0]])
    faces=[]
    for index,y in enumerate([0,2]):
        v=np.array([[0,y,-1],[1,y,-1],[1,y,1],[0,y,1]])@r.T
        faces.append(dict(id=f'F{index}',positions=v.ravel().tolist(),indices=[0,1,2,0,2,3]))
    page=dict(page=1,program='DOWN',measured_system='FRAME',text='Measure 4 vector points; construct a set; Dimension line profile with datum.\n'+'\n'.join(f'X, Y, Z (mm) {x} {y} {z} OUT_PNT_P{i:02}' for i,(x,y,z) in enumerate(points,1)))
    anchors=[dict(face_id=f'F{i//2}',position=(v@r.T).tolist()) for i,v in enumerate(points)]
    task=dict(methods=[page],anchors=anchors,features=[])
    calibration=dict(hypotheses=[dict(name='legacy XY',systems=['FRAME'],program='DOWN')],rigid_frame_proposals=[dict(frame='FRAME',program='DOWN',status='geometrically_supported_proposal',rotation=r.tolist(),translation_mm=[0,0,0],threshold_mm=.1)] if registered else [],max_xy_residual_mm=.1)
    return task,dict(faces=faces),calibration,[page]


def test_registered_profile_keeps_one_plane_across_two_cad_faces():
    task,mesh,calibration,pages=section_fixture()
    result=map_task_points(task,mesh,calibration,pages)
    assert result['status']=='mapped_complete' and result['can_replace_visual_anchors']
    group=result['common_source_z_groups'][0]
    assert group['status']=='registered_measurement_plane' and len(group['cad_face_ids'])==2
    assert group['plane_coordinate_span_mm']<1e-9 and not group['common_face_required']


def test_tilted_source_plane_does_not_force_constant_cad_z():
    a=np.pi/4;r=[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]
    task,mesh,calibration,pages=section_fixture(r)
    result=map_task_points(task,mesh,calibration,pages)
    group=result['common_source_z_groups'][0]
    assert result['can_replace_visual_anchors'] and group['plane_coordinate_span_mm']<1e-9
    assert group['cad_z_span_mm']>1


def test_xy_profile_without_plane_registration_cannot_replace_markers():
    task,mesh,calibration,pages=section_fixture(registered=False)
    result=map_task_points(task,mesh,calibration,pages)
    assert not result['can_replace_visual_anchors']
    assert result['common_source_z_groups'][0]['status']=='measurement_plane_registration_required'


def test_msop_line_profile_word_order_emits_correct_symbol():
    assert any(i['symbol']=='⌒' for i in measurement_indicators('Dimension line profile with datum.'))
    assert any(i['symbol']=='⌓' for i in measurement_indicators('Dimension surface profile with datum.'))


def test_contextless_program_does_not_match_contextless_legacy_calibration():
    task,mesh,calibration,pages=section_fixture(registered=False)
    pages[0]['program']=None;calibration['hypotheses'][0].pop('program')
    pages += [dict(page=2,program='UP',measured_system='FRAME',text='X, Y, Z (mm) 0 0 1 OTHER_UP'),dict(page=3,program='DOWN',measured_system='FRAME',text='X, Y, Z (mm) 0 0 2 OTHER_DOWN')]
    result=map_task_points(task,mesh,calibration,pages)
    assert len(result['source_point_constraints'])==1
    assert result['mapped_points']==[] and not result['can_replace_visual_anchors']


def test_explicit_xyz_z_sections_are_preserved_for_singletons_and_lines():
    from oracle_viewer.section_geometry import source_plane_constraints
    rows = [
        dict(name='PNT_A', point_key='a', xyz=[1, 2, 0.0], program='UP', coordinate_system='FRAME'),
        dict(name='PNT_B', point_key='b', xyz=[3, 4, 0.0], program='UP', coordinate_system='FRAME'),
        dict(name='PNT_C', point_key='c', xyz=[5, 6, 1.0], program='UP', coordinate_system='FRAME'),
    ]
    groups = source_plane_constraints(rows, [])
    assert len(groups) == 2
    assert groups[0]['point_keys'] == ['a', 'b'] and groups[0]['offset_source_mm'] == 0.0
    assert groups[1]['point_keys'] == ['c'] and groups[1]['offset_source_mm'] == 1.0
