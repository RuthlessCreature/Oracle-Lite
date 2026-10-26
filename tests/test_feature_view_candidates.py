import numpy as np

from oracle_viewer.cad_view_atlas import project_point_to_view_uv,ray_hit_view
from oracle_viewer.feature_view_candidates import feature_view_candidates


def rail_mesh():
    return {'source_sha256':'thin-feature-test','faces':[
        {'id':'thin-rail','positions':[0,0,0,.02,0,0,.02,10,0,0,10,0],
         'indices':[0,1,2,0,2,3]},
        {'id':'other-part','positions':[3,0,0,4,0,0,4,1,0,3,1,0],
         'indices':[0,1,2,0,2,3]}]}


def test_thin_surface_recovery_uses_real_geometry_not_uniform_grid():
    mesh=rail_mesh()
    uv=project_point_to_view_uv(mesh,'+Z',[.01,5,0])[0]
    missed=[uv[0]-.04,uv[1]]
    assert ray_hit_view(mesh,'+Z',*missed) is None
    choices=feature_view_candidates(mesh,'+Z',missed)
    assert choices
    assert any(choice['face_id']=='thin-rail' for choice in choices)
    for choice in choices:
        hit=ray_hit_view(mesh,choice['view_id'],*choice['uv'])
        assert hit['face_id']==choice['face_id']
        assert np.allclose(hit['position'],choice['position'])
    assert feature_view_candidates(mesh,'+Z',missed,max_uv_distance=.001)==[]
    assert feature_view_candidates(mesh,'bad-view',missed)==[]


def test_hidden_triangle_seed_returns_only_frontmost_visible_surface():
    mesh=rail_mesh()
    top={**mesh['faces'][0],'id':'cover',
         'positions':[v+2 if i%3==2 else v for i,v in enumerate(mesh['faces'][0]['positions'])]}
    mesh['faces'].append(top)
    uv=project_point_to_view_uv(mesh,'+Z',[.01,5,0])[0]
    choices=feature_view_candidates(mesh,'+Z',[uv[0]-.04,uv[1]])
    assert choices and all(row['face_id']=='cover' for row in choices)
    assert all(np.isclose(row['position'][2],2) for row in choices)
