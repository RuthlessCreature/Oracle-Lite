import numpy as np
from oracle_viewer.auto_registration import automatic_fit,axis_rotations,refine_surface_fit,surface_seed_candidates
from oracle_viewer.surface_index import SurfaceIndex


def test_axis_search_contains_only_proper_rotations():
    rotations=list(axis_rotations())
    assert len(rotations)==24
    assert all(np.allclose(r.T@r,np.eye(3)) and np.isclose(np.linalg.det(r),1) for r in rotations)


def test_planar_source_runs_shape_fit_but_never_claims_registration():
    points=np.array([[x,y,0] for x in range(3) for y in range(3)])
    result=automatic_fit(points,points)
    assert result['source_rank']==2 and result['planar_source']
    assert result['best']['max_residual_mm']<1e-8
    assert not result['automatic_application']
    assert not result['source_to_cad_registered']
    assert not result['metrology_accepted']


def test_collinear_source_still_has_insufficient_spread():
    points=np.array([[x,0,0] for x in range(8)])
    result=automatic_fit(points,points)
    assert result['status']=='insufficient_source_spread'
    assert not result['automatic_application']


def test_five_point_group_generates_candidate_with_one_independent_holdout():
    points=np.array([[0,0,0],[2,0,0],[0,3,0],[2,3,0],[1,.5,0]],float)
    result=automatic_fit(points,points)
    assert result['best']['max_residual_mm']<1e-8
    assert result['fit_point_count']==4 and result['held_out_point_count']==1
    assert not result['automatic_application'] and not result['source_to_cad_registered']


def test_four_point_group_is_not_given_a_transform_without_a_heldout_point():
    points=np.array([[0,0,0],[2,0,0],[0,3,0],[2,3,0]],float)
    result=automatic_fit(points,points)
    assert result['status']=='insufficient_unambiguous_points'


def test_unpaired_fit_keeps_independent_holdout_and_unreviewed_authority():
    points=np.random.default_rng(3).normal(size=(30,3))*10
    result=automatic_fit(points,points+[2,3,4])
    assert result['best']['max_residual_mm']<1e-6
    assert result['held_out_point_count']==6
    assert not result['source_to_cad_registered'] and not result['metrology_accepted']


def test_triangle_refinement_uses_surface_not_vertex_distance():
    triangles=np.array([[[0,0,0],[10,0,0],[0,10,0]],
        [[0,0,5],[0,10,5],[10,0,5]]],float)
    source=np.array([[x,y,z] for z in (0,5) for x,y in ((1,1),(2,1),(1,2),(3,2),(2,3))])
    result=refine_surface_fit(source,triangles,[dict(rotation=np.eye(3).tolist(),translation_mm=[0,0,0])])
    assert result['best']['max_residual_mm']<1e-8
    assert not result['source_to_cad_registered']


def test_surface_index_does_not_miss_large_triangle_with_distant_centroid():
    rng=np.random.default_rng(42)
    small=[]
    for center in rng.normal(size=(100,3))*.05+[0,0,1]:
        small.append([center,center+[.01,0,0],center+[0,.01,0]])
    tri=np.array(small+[[[-100,-100,0],[100,-100,0],[0,100,0]]])
    index=SurfaceIndex(tri)
    assert index.nearest([0,0,.05],exact=False)['distance_mm']>.5
    hit=index.nearest([0,0,.05])
    assert abs(hit['distance_mm']-.05)<1e-9
    assert hit['triangle_index']==100
    assert np.allclose(hit['point'],[0,0,0])


def test_surface_seed_retrieval_does_not_read_held_out_points():
    triangles=np.array([[[0,0,0],[10,0,0],[0,10,0]],
                        [[0,0,5],[0,10,5],[10,0,5]]],float)
    source=np.array([[x,y,z] for z in (0,5) for x,y in ((1,1),(2,1),(1,2),(3,2),(2,3))],float)
    first=surface_seed_candidates(source,triangles,limit=2)
    source[np.arange(len(source))%5==0]+=[100,200,300]
    second=surface_seed_candidates(source,triangles,limit=2)
    assert first==second and all(row['fit_only'] for row in first)


def test_refinement_selection_cannot_be_changed_by_held_out_coordinates():
    triangles=np.array([[[0,0,0],[10,0,0],[0,10,0]],
                        [[0,0,5],[0,10,5],[10,0,5]]],float)
    source=np.array([[x,y,z] for z in (0,5) for x,y in ((1,1),(2,1),(1,2),(3,2),(2,3))],float)
    candidates=[dict(rotation=np.eye(3).tolist(),translation_mm=t) for t in ([0,0,0],[.2,.2,.1])]
    first=refine_surface_fit(source,triangles,candidates)
    source[np.arange(len(source))%5==0]+=[100,200,300]
    second=refine_surface_fit(source,triangles,candidates)
    assert first['selection_uses_fit_points_only']
    assert np.allclose(first['best']['rotation'],second['best']['rotation'])
    assert np.allclose(first['best']['translation_mm'],second['best']['translation_mm'])
    assert second['best']['held_out_max_mm']>100
