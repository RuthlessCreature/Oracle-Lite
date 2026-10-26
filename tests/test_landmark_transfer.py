import json
from types import SimpleNamespace
from oracle_viewer.landmark_transfer import peer_landmark_frame_options


def fixture(source,target,*,extra=None,cad='cad'):
    points=[{'name':f'P{i}','xyz':p,'program':'UP','coordinate_system':'A'} for i,p in enumerate(source)]
    if extra:points.append({'name':'OTHER','xyz':extra,'program':'UP','coordinate_system':'A'})
    rows=[{'name':p['name'],'source_xyz':p['xyz'],'transformed_xyz':t,'program':'UP','coordinate_system':'A'} for p,t in zip(points,target)]
    peer={'number':1,'annotation':{'status':'auto_marked','origin':'local_model_candidate',
        'source_sha256':cad,'document_sha256':'msop','note':'逐点原文 XYZ 与 CAD 投影：'+json.dumps(rows)}}
    store=SimpleNamespace(active_msop=lambda _:{'sha256':'msop'},items=lambda _:[peer])
    frame={'program':'UP','coordinate_system':'A','rotation':[[1,0,0],[0,1,0],[0,0,1]],'translation_mm':[0,0,0]}
    return store,{'model':{'source_sha256':'cad'}},points,[frame]


def test_exact_shared_landmarks_propose_a_frame_but_never_authorize_it():
    store,project,points,frames=fixture([[0,0,0],[1,0,0],[0,1,0]],[[4,5,6],[5,5,6],[4,6,6]])
    choices=peer_landmark_frame_options(store,project,{'number':2},points,frames)
    assert len(choices)==1 and choices[0][0]['translation_mm']==[4,5,6]
    assert choices[0][0]['peer_landmark_graph']['requires_fresh_local_model_selection']
    assert not choices[0][0]['peer_landmark_graph']['prior_annotation_is_ground_truth']
    assert frames[0]['translation_mm']==[0,0,0]


def test_collinear_landmarks_cannot_authorize_off_line_points_or_cross_cad_sources():
    store,project,points,frames=fixture([[0,0,0],[1,0,0]],[[4,5,6],[5,5,6]],extra=[0,1,0])
    assert not peer_landmark_frame_options(store,project,{'number':2},points,frames)
    store,project,points,frames=fixture([[0,0,0],[1,0,0]],[[4,5,6],[5,5,6]],cad='different')
    assert not peer_landmark_frame_options(store,project,{'number':2},points,frames)


def test_collinear_target_group_can_be_proposed_without_claiming_a_full_frame():
    store,project,points,frames=fixture([[0,0,0],[1,0,0]],[[4,5,6],[5,5,6]])
    choices=peer_landmark_frame_options(store,project,{'number':2},points,frames)
    assert len(choices)==1 and choices[0][0]['peer_landmark_graph']['basis'].startswith('collinear_source_group_only')
