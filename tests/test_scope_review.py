import json
from types import SimpleNamespace
from oracle_viewer import scope_review,local_matching


def fixture(tmp_path,monkeypatch,response):
    monkeypatch.setattr(local_matching,'_is_local_inference_model',lambda _:True)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *_:[])
    store=SimpleNamespace(_dir=lambda _:tmp_path,active_msop=lambda _:{'sha256':'msop','method_pages':[
        {'page':1,'program':'DOWN','text':'Measured System ALI_ITER. Dimension System ALI_ABD.'},
        {'page':2,'program':'DOWN','text':'FAI 7 DATUMA_U01 XYZ (1,2,0) Measured System ALI_ITER.'}]})
    class Model:
        loaded=True
        def info(self):return {'inference_backend':'strata-native'}
        def _generate_once(self,*args,**kwargs):return json.dumps(response),False,100
    project={'id':'p','model':{'source_sha256':'cad'}}
    detail={'methods':[{'page':1,'program':'DOWN','text':'Measured System ALI_ITER. Dimension System ALI_ABD.'}]}
    points=[{'name':'DATUMA_U01','xyz':[1,2,0],'pages':[2],'program':'DOWN','coordinate_system':'ALI_ABD'}]
    return store,project,{'number':7},detail,Model(),points


def decision():
    return {'decision':'ASSIGN_SCOPES','fai':7,'groups':[{'group_id':'S1','program':'DOWN','coordinate_system':'ALI_ITER','source_page':1,'source_quote':'Measured System ALI_ITER','reason':'The reporting alignment is distinct.'}],'evidence':'Original measuring alignment.'}


def test_native_scope_review_preserves_xyz_and_does_not_authorize_geometry(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch,decision());points,audit=scope_review.review_source_scopes(*args)
    assert points[0]['coordinate_system']=='ALI_ITER' and points[0]['xyz']==[1,2,0]
    assert args[-1][0]['coordinate_system']=='ALI_ABD'
    assert audit['status']=='selected' and audit['does_not_authorize_anchors']
    assert audit['source_xyz_modified'] is False


def test_scope_review_rejects_unprinted_frame_and_invented_quote(tmp_path,monkeypatch):
    for index,(field,value) in enumerate([('coordinate_system','ALI_FAKE'),('source_quote','Measured System ALI_ITER with invented text')]):
        response=decision();response['groups'][0][field]=value
        args=fixture(tmp_path/str(index),monkeypatch,response)
        points,audit=scope_review.review_source_scopes(*args)
        assert points is None and audit['status']=='unconfirmed'


def test_scope_review_never_calls_hosted_model(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch,decision())
    monkeypatch.setattr(local_matching,'_is_local_inference_model',lambda _:False)
    assert scope_review.review_source_scopes(*args)[0] is None


def test_scope_contract_feedback_is_redecided_by_local_model_and_cached(tmp_path,monkeypatch):
    bad=decision();bad['groups'][0]['coordinate_system']='ALI_ABD'
    args=fixture(tmp_path,monkeypatch,bad)
    model=args[4];calls=[]
    def generate(messages,**kwargs):
        calls.append(messages[0]['content'])
        return json.dumps(bad if len(calls)==1 else decision()),False,100
    model._generate_once=generate
    points,audit=scope_review.review_source_scopes(*args)
    assert points[0]['coordinate_system']=='ALI_ITER'
    assert points[0]['xyz']==args[-1][0]['xyz']
    assert audit['attempt_count']==2 and not audit['validation_errors']
    assert 'selected_frame_not_present_as_complete_token_in_quote' in calls[1]
    receipt=json.loads(open(audit['receipt']).read())
    assert receipt['attempts'][0]['decision']==bad
    assert receipt['attempts'][1]['decision']['groups'][0]['coordinate_system']=='ALI_ITER'
    scope_review.review_source_scopes(*args)
    assert len(calls)==2


def test_scope_explicit_abstention_is_preserved_without_retry(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch,{'decision':'ABSTAIN','fai':7,'evidence':'Conflicting source.'})
    points,audit=scope_review.review_source_scopes(*args)
    assert points is None and audit['attempt_count']==1
    assert audit['decision']['decision']=='ABSTAIN'


def test_scope_can_recover_frame_from_fai_linked_coordinate_page(tmp_path,monkeypatch):
    response=decision();args=fixture(tmp_path,monkeypatch,response)
    # The feature method omits its frame. The cited source XYZ page carries it.
    mutable=list(args)
    mutable[3]={'methods':[{'page':3,'program':'UP','text':'Measure 4 vector points DATUMA_U01.'}]}
    mutable[-1]=[{'name':'DATUMA_U01','xyz':[1,2,0],'pages':[2],'program':'DOWN','coordinate_system':None}]
    response['groups'][0]['program']='DOWN';response['groups'][0]['source_page']=2
    response['groups'][0]['source_quote']='FAI 7 DATUMA_U01 XYZ (1,2,0) Measured System ALI_ITER'
    model=mutable[4];model._generate_once=lambda *a,**k:(json.dumps(response),False,100)
    points,audit=scope_review.review_source_scopes(*mutable)
    assert audit['status']=='selected'
    assert points[0]['coordinate_system']=='ALI_ITER'
    assert points[0]['xyz']==[1,2,0]


def test_scope_quote_requires_whole_alignment_token():
    response=decision();response['groups'][0].update(coordinate_system='ALI_ABC',source_quote_id='Q1')
    errors=scope_review._scope_assignment_errors(response,fai=7,expected={'S1'},programs=['DOWN'],
        frames=['ALI_ABC','ALI_ABC_A'],quotes={'Q1':{'source_page':1,'source_quote':'FAI 7 Measured System ALI_ABC_A'}},
        methods=[{'page':1,'text':'FAI 7 Measured System ALI_ABC_A','is_fai_method_page':True}])
    assert errors==['S1:selected_frame_not_present_as_complete_token_in_quote']


def test_unrelated_fai_coordinate_page_cannot_authorize_measurement_scope():
    response=decision();response['fai']=66;response['groups'][0].update(program='UP',source_page=52,
        source_quote='FAI 38 (SPC Z) Measured System ALI_ABD',coordinate_system='ALI_ABD')
    errors=scope_review._scope_assignment_errors(response,fai=66,expected={'S1'},programs=['UP'],
        frames=['ALI_ABD'],quotes={},methods=[
            {'page':52,'text':'D52 Frame Welding Assy | MSOP: FAI 38 (SPC Z) Measured System ALI_ABD.',
             'is_fai_method_page':False}])
    assert errors==['S1:cited_page_does_not_identify_this_fai']


def test_range_header_can_authorize_linked_page_for_fai():
    assert scope_review._page_explicitly_covers_fai('MSOP: FAI 61 to FAI 79',66)
    assert not scope_review._page_explicitly_covers_fai('MSOP: FAI 38',66)


def test_scope_failed_repair_is_bounded_and_never_rewrites_source(tmp_path,monkeypatch):
    response=decision();response['groups'][0]['coordinate_system']='ALI_FAKE'
    args=fixture(tmp_path,monkeypatch,response)
    points,audit=scope_review.review_source_scopes(*args)
    assert points is None and audit['attempt_count']==3
    assert args[-1][0]['coordinate_system']=='ALI_ABD'
    assert audit['source_xyz_modified'] is False


def test_plane_minimax_keeps_rigid_distances_and_the_exact_plane():
    import numpy as np
    from oracle_viewer.surface_index import SurfaceIndex
    source=np.asarray([[.2,.2,0],[.8,.2,0],[.2,.8,0],[.8,.8,0],[.5,.5,0]])
    triangles=np.asarray([[[0,0,0],[2,0,0],[0,2,0]],[[2,0,0],[2,2,0],[0,2,0]]])
    seed={'rotation':np.eye(3).tolist(),'translation_mm':[-.3,0,0]}
    result=scope_review._plane_minimax_candidates(source,SurfaceIndex(triangles),[seed])
    assert len(result)==1
    r=np.asarray(result[0]['rotation']);t=np.asarray(result[0]['translation_mm'])
    transformed=source@r.T+t
    assert np.allclose(transformed[:,2],0) and np.allclose(r.T@r,np.eye(3))
    assert np.allclose(np.linalg.norm(transformed[0]-transformed[1]),np.linalg.norm(source[0]-source[1]))
    assert result[0]['whole_plane_refinement']['points_omitted']==0


def test_two_point_group_can_search_origin_without_claiming_frame_solution():
    import numpy as np
    mesh={'faces':[{'id':'bottom','positions':[0,0,0,4,0,0,0,4,0],'indices':[0,1,2]},
                   {'id':'top','positions':[0,0,2,4,0,2,0,4,2],'indices':[0,1,2]}]}
    points=[{'name':f'P{i}','xyz':xyz,'program':'UP','coordinate_system':'A','pages':[1]}
        for i,xyz in enumerate([[1.,1.,0.],[2.,1.,0.]])]
    frame={'program':'UP','coordinate_system':'A','rotation':np.eye(3).tolist(),'translation_mm':[0,0,1.]}
    trials=local_matching._normal_origin_frame_hypotheses(mesh,points,[frame])
    assert trials
    for trial in trials:
        assert trial[0]['normal_origin_search']['point_count']==2
        assert trial[0]['normal_origin_search']['source_coordinates_modified'] is False
        assert trial[0]['normal_origin_search']['metrology_accepted'] is False
        projection,error=local_matching._project_model_selected_xyz_group(mesh,points,trial,{},[])
        assert error is None and len(projection[0])==2


def test_small_group_translation_preserves_existing_orientation_and_all_source_points():
    import numpy as np
    from oracle_viewer.surface_index import SurfaceIndex
    rows=[{'xyz':[x,.2,0]} for x in [.2,.8]]
    triangles=np.asarray([[[0,0,0],[2,0,0],[0,2,0]],[[2,0,0],[2,2,0],[0,2,0]]])
    frame={'rotation':np.eye(3).tolist(),'translation_mm':[-.3,0,0]}
    candidates=scope_review.low_cardinality_translation_candidates(rows,frame,SurfaceIndex(triangles))
    assert len(candidates)==1 and candidates[0]['rotation']==frame['rotation']
    assert candidates[0]['low_cardinality_translation']['point_count']==2
    assert candidates[0]['low_cardinality_translation']['source_xyz_modified'] is False
    assert np.max(np.abs(candidates[0]['low_cardinality_translation']['delta_translation_mm']))<=.8
    assert rows==[{'xyz':[.2,.2,0]},{'xyz':[.8,.2,0]}]


def test_output_table_frame_requires_exact_xyz_and_keeps_absent_program_null(tmp_path,monkeypatch):
    response=decision();row=response['groups'][0]
    row.update(program=None,coordinate_system='ALI_ABC',source_page=1,
        source_quote='Output system ALI_ABC',scope_kind='coordinate_table_output_frame')
    args=list(fixture(tmp_path,monkeypatch,response))
    page={'page':1,'program':None,'text':'FAI 7\nOutput system ALI_ABC\nMeasure 1 points.\nFeatures X Y Z\nDATUMA_U01 1 2 0'}
    args[0].active_msop=lambda _:{'sha256':'msop','method_pages':[page]}
    args[3]={'methods':[page]};args[-1][0].update(program=None,pages=[1])
    points,audit=scope_review.review_source_scopes(*args)
    assert audit['status']=='selected'
    assert points[0]['xyz']==[1,2,0] and points[0]['program'] is None
    assert points[0]['coordinate_system']=='ALI_ABC'
    args[-1][0]['xyz']=[9,2,0]
    assert scope_review.review_source_scopes(*args)[0] is None
