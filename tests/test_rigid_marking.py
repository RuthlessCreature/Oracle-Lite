import numpy as np
import pytest

from oracle_viewer.local_matching import (_project_model_selected_xyz_group,
                                          _rigid_correspondence, _rigid_source_anchors,
                                          _registration_reference_selection_valid,
                                          _refine_model_selected_xyz_frames)


def _plane_mesh():
    return {'faces':[{'id':'plane','positions':[
        -20,-20,2, 20,-20,2, 20,20,2, -20,20,2],
        'indices':[0,1,2,0,2,3]}]}


def _two_level_mesh():
    return {'faces':[
        {'id':'lower','positions':[-20,-20,2, 0,-20,2, 0,20,2, -20,20,2],
         'indices':[0,1,2,0,2,3]},
        {'id':'upper','positions':[0,-20,2.2, 20,-20,2.2, 20,20,2.2, 0,20,2.2],
         'indices':[0,1,2,0,2,3]},
    ]}


def test_sparse_registration_references_can_cover_a_complete_fai_point_group():
    points=[{'name':f'P{i+1}','xyz':xyz,'program':'UP','coordinate_system':'ALI_A',
             'requires_variant_choice':False}
            for i,xyz in enumerate(([-8,-4,0],[0,-4,0],[8,-4,0],[-8,4,0],[0,4,0],[8,4,0]))]
    refs=[{'source_point_name':name,'view_id':'+Z','uv':uv} for name,uv in (
        ('P1',[.2,.3]),('P3',[.8,.3]),('P4',[.2,.7]),('P6',[.8,.7]))]
    markers=[{'source_point_names':[point['name']]} for point in points]
    valid,audit=_registration_reference_selection_valid(points,refs,[],markers)
    assert valid
    assert audit['reference_count']==4
    assert audit['source_point_count']==6


def test_sparse_registration_references_reject_collinear_reference_points():
    points=[{'name':f'P{i+1}','xyz':xyz,'program':'UP','coordinate_system':'ALI_A',
             'requires_variant_choice':False}
            for i,xyz in enumerate(([-8,0,0],[-4,0,0],[0,0,0],[4,0,0],[8,0,0]))]
    refs=[{'source_point_name':name,'view_id':'+Z','uv':[.1*i,.2]} for i,name in enumerate(
        ('P1','P2','P3','P5'))]
    markers=[{'source_point_names':[point['name']]} for point in points]
    valid,audit=_registration_reference_selection_valid(points,refs,[],markers)
    assert not valid
    assert audit['reason']=='insufficient_or_collinear_registration_references'


def test_sparse_registration_references_still_require_complete_marker_group_coverage():
    points=[{'name':f'P{i+1}','xyz':xyz,'program':'UP','coordinate_system':'ALI_A',
             'requires_variant_choice':False}
            for i,xyz in enumerate(([-8,-4,0],[8,-4,0],[-8,4,0],[8,4,0]))]
    refs=[{'source_point_name':point['name'],'view_id':'+Z','uv':[(point['xyz'][0]+8)/16,
        (point['xyz'][1]+4)/8]} for point in points]
    markers=[{'source_point_names':[point['name']]} for point in points[:-1]]
    valid,audit=_registration_reference_selection_valid(points,refs,[],markers)
    assert not valid
    assert audit['reason']=='marker_groups_do_not_cover_every_source_point'


def test_complete_xyz_group_keeps_common_plane_and_relative_distances():
    source=np.array([[-5,-4,0],[5,-4,0],[5,4,0],[-5,4,0]],float)
    angle=np.deg2rad(23)
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],
                       [np.sin(angle),np.cos(angle),0],[0,0,1]])
    target=source@rotation.T+np.array([1,2,2])

    result=_rigid_source_anchors(_plane_mesh(),source.tolist(),target.tolist())

    assert result is not None
    anchors,fit=result
    positions=np.asarray([anchor['position'] for anchor in anchors])
    assert np.ptp(positions[:,2])<1e-9
    assert np.allclose(np.linalg.norm(positions[:,None]-positions[None,:],axis=2),
                       np.linalg.norm(source[:,None]-source[None,:],axis=2),atol=1e-8)
    assert fit['max_residual_mm']<1e-8
    assert fit['max_surface_residual_mm']<1e-8


def test_rigid_fit_rejects_nonrigid_or_wrong_correspondence():
    source=np.array([[0,0,0],[10,0,0],[10,10,0],[0,10,0]],float)
    target=source+np.array([0,0,2])
    target[2,0]+=2

    assert _rigid_correspondence(source,target,max_residual_mm=.75) is None
    assert _rigid_source_anchors(_plane_mesh(),source.tolist(),target.tolist()) is None


def test_rigid_fit_rejects_collinear_points():
    source=np.array([[0,0,0],[1,0,0],[2,0,0]],float)
    target=source+np.array([0,0,2])
    assert _rigid_correspondence(source,target) is None


def test_model_selected_frame_projects_whole_group_without_breaking_source_plane():
    source=[[0,0,0],[4,0,0],[4,4,0],[0,4,0]]
    points=[dict(name=f'P{i+1}',xyz=xyz,pages=[12],program='UP',coordinate_system='ALI_A')
            for i,xyz in enumerate(source)]
    frame=dict(program='UP',coordinate_system='ALI_A',hypothesis_id='H00',
               rotation=np.eye(3).tolist(),translation_mm=[0,0,2],model_confidence='medium')

    projected,error=_project_model_selected_xyz_group(_plane_mesh(),points,[frame],{'methods':[]},[])

    assert error is None
    anchors,audit,scopes=projected
    positions=np.asarray([row['position'] for row in anchors])
    assert len(anchors)==len(source)
    assert np.ptp(positions[:,2])<1e-9
    assert np.allclose(np.linalg.norm(positions[:,None]-positions[None,:],axis=2),
                       np.linalg.norm(np.asarray(source)[:,None]-np.asarray(source)[None,:],axis=2))
    assert scopes[0]['source_to_cad_registered'] is False
    assert scopes[0]['point_count']==4
    assert all(row['surface_residual_mm']==0 for row in audit)


def test_model_selected_frame_gets_bounded_local_refinement_then_exact_source_section():
    points=[dict(name=f'PNT_A{i}',xyz=xyz,pages=[12],program='UP',coordinate_system='ALI_A')
            for i,xyz in enumerate([[-4,-2,0],[4,-2,0],[4,2,0],[-4,2,0]],1)]
    frame=dict(program='UP',coordinate_system='ALI_A',hypothesis_id='H00',
        rotation=np.eye(3).tolist(),translation_mm=[0,0,1.7],model_confidence='medium')
    frames,audit=_refine_model_selected_xyz_frames(_plane_mesh(),points,[frame])
    assert audit and frames[0]['section_refined'] is True
    assert frames[0]['display_refinement']['source_to_cad_registered'] is False
    assert frames[0]['display_refinement']['max_section_residual_mm']<1e-7
    projected,error=_project_model_selected_xyz_group(
        _plane_mesh(),points,frames,{'methods':[]},[])
    assert error is None
    anchors,point_audit,_scopes=projected
    positions=np.asarray([row['position'] for row in anchors])
    assert np.ptp(positions[:,2])<1e-9
    assert all(row['surface_residual_mm']<1e-7 for row in point_audit)


def test_named_feature_families_at_same_source_z_are_checked_as_separate_planes():
    source=[[-10,-2,0],[-10,2,0],[10,-2,0],[10,2,0]]
    names=['PLASTIC_X01','PLASTIC_X02','CHASSIS_X01','CHASSIS_X02']
    points=[dict(name=name,xyz=xyz,pages=[96],program='UP',coordinate_system='ALI_A')
            for name,xyz in zip(names,source)]
    frame=dict(program='UP',coordinate_system='ALI_A',hypothesis_id='H00',
               rotation=np.eye(3).tolist(),translation_mm=[0,0,2],model_confidence='medium')

    projected,error=_project_model_selected_xyz_group(
        _two_level_mesh(),points,[frame],{'methods':[]},[])

    assert error is None
    anchors,audit,scopes=projected
    assert len(anchors)==4
    sections=scopes[0]['source_sections']
    assert {row['feature_group'] for row in sections}=={'PLASTIC','CHASSIS'}
    offsets={row['feature_group']:row['common_offset_mm'] for row in sections}
    assert offsets['PLASTIC']==pytest.approx(0)
    assert offsets['CHASSIS']==pytest.approx(.2)


def test_same_named_feature_family_still_rejects_non_coplanar_source_section():
    source=[[-10,-2,0],[10,-2,0]]
    points=[dict(name=name,xyz=xyz,pages=[96],program='UP',coordinate_system='ALI_A')
            for name,xyz in zip(['DATUMA_U01','DATUMA_U02'],source)]
    frame=dict(program='UP',coordinate_system='ALI_A',hypothesis_id='H00',
               rotation=np.eye(3).tolist(),translation_mm=[0,0,2],model_confidence='medium')

    projected,error=_project_model_selected_xyz_group(
        _two_level_mesh(),points,[frame],{'methods':[]},[])

    assert projected is None
    assert error['reason']=='source_section_coplanarity_not_preserved'
    assert error['feature_group']=='DATUMA'


def test_bare_numbered_point_rows_cannot_bypass_common_plane_gate():
    points=[dict(name=name,xyz=xyz,pages=[12],program='UP',coordinate_system='ALI_A')
        for name,xyz in zip(['Q38','Q39'],[[-10,-2,0],[10,-2,0]])]
    frame=dict(program='UP',coordinate_system='ALI_A',rotation=np.eye(3).tolist(),translation_mm=[0,0,2])
    projected,error=_project_model_selected_xyz_group(_two_level_mesh(),points,[frame],{'methods':[]},[])
    assert projected is None and error['reason']=='source_section_coplanarity_not_preserved'
    assert error['feature_group']=='Q'
