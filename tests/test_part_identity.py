import json
from types import SimpleNamespace
from oracle_viewer.part_identity import cad_parts,select_physical_parts,_candidate_geometry_support
from oracle_viewer.local_matching import (_source_visual_search_candidates,
    _source_marker_host_mismatches,_exact_surface_candidate_hit,
    _candidate_host_selection_audit)


def mesh():
    return {'source_sha256':'test','faces':[
        {'id':'left','body':1,'prototype_body':4,'positions':[0,0,0,1,0,0,0,1,0],'indices':[0,1,2]},
        {'id':'right','body':2,'prototype_body':4,'positions':[10,0,0,11,0,0,10,1,0],'indices':[0,1,2]}]}


def test_catalog_keeps_repeated_original_bodies_and_assembly_positions():
    parts=cad_parts(mesh())
    assert [p['part_id'] for p in parts]==['B1','B2']
    assert parts[0]['prototype_body']==parts[1]['prototype_body']
    assert parts[0]['bbox_mm']['min']==[0,0,0] and parts[1]['bbox_mm']['min']==[10,0,0]
    assert parts[0]['face_ids']==['left']


def test_part_screening_geometry_hints_keep_only_direct_fai_point_names():
    candidate={'required_marker_groups':[{'source_point_names':['P_DIRECT']}],
        'evidence':{'point_correspondences':[
            {'name':'P_DIRECT','source_xyz':[1,2,3],'transformed_xyz':[10,0,0],
             'position':[10,0,0],'face_id':'right','residual_mm':0.01},
            {'name':'P_SIMILAR','source_xyz':[1,2,3],'transformed_xyz':[0,0,0],
             'position':[0,0,0],'face_id':'left','residual_mm':0.01}]}}
    support=_candidate_geometry_support(mesh(),candidate)
    assert len(support)==1 and support[0]['name']=='P_DIRECT'
    assert support[0]['body_id']=='B2'
    assert support[0]['authority']=='prior_local_model_display_hypothesis_only'


def test_xyz_missing_direct_point_gets_only_real_numbered_cad_ray_hits():
    options=_source_visual_search_candidates(mesh(),['CIR_N1_VISION'])
    assert options
    assert [row['id'] for row in options]==list(range(1,len(options)+1))
    assert all(row['candidate_role']=='global_surface_search'
        and row['source_point_names']==['CIR_N1_VISION']
        and row.get('face_id') is not None and row.get('position') is not None
        for row in options)


def test_xyz_missing_point_search_bounds_candidates_per_view_for_complete_fai_context():
    options=_source_visual_search_candidates(mesh(),['POINT_A','POINT_B'])
    counts={}
    for row in options:
        key=(row['source_point_names'][0],row['view_id'])
        counts[key]=counts.get(key,0)+1
    assert counts and max(counts.values())<=6


def test_visual_surface_candidates_are_scoped_to_each_source_point_host():
    options=_source_visual_search_candidates(mesh(),['P_LEFT','P_RIGHT'],
        allowed_face_ids_by_source_name={'P_LEFT':{'left'},'P_RIGHT':{'right'}})
    assert options
    assert all(row['source_point_names'] in (['P_LEFT'],['P_RIGHT']) for row in options)
    assert all(row['face_id']=='left' for row in options if 'P_LEFT' in row['source_point_names'])
    assert all(row['face_id']=='right' for row in options if 'P_RIGHT' in row['source_point_names'])
    assert {tuple(row['source_point_names']) for row in options}=={('P_LEFT',),('P_RIGHT',)}


def test_candidate_sampling_filters_to_host_before_limiting_each_view(monkeypatch):
    import oracle_viewer.local_matching as matching
    calls=[]
    def candidates(_mesh,**kwargs):
        allowed=kwargs.get('allowed_face_ids');calls.append(allowed)
        face='large-frame' if allowed is None else next(iter(allowed))
        return [{'id':index,'view_id':'+Z','uv':[index/10,.5],
            'face_id':face,'position':[index,0,0]} for index in range(1,7)]
    monkeypatch.setattr(matching,'distributed_surface_view_candidates',candidates)
    options=matching._source_visual_search_candidates(mesh(),['P_SMALL'],
        allowed_face_ids={'large-frame','small-rail'},
        allowed_face_ids_by_source_name={'P_SMALL':{'small-rail'}})
    assert calls==[{'small-rail'}]
    assert len(options)==6 and all(row['face_id']=='small-rail' for row in options)


def test_shared_host_candidates_keep_source_specific_ids_for_distinct_points():
    options=_source_visual_search_candidates(mesh(),['P_RAIL_A','P_RAIL_B'],
        allowed_face_ids_by_source_name={'P_RAIL_A':{'right'},'P_RAIL_B':{'right'}})
    assert options
    assert all(row['source_point_names'] in (['P_RAIL_A'],['P_RAIL_B']) for row in options)
    assert {tuple(row['source_point_names']) for row in options}=={('P_RAIL_A',),('P_RAIL_B',)}
    left=[row for row in options if row['source_point_names']==['P_RAIL_A']]
    right=[row for row in options if row['source_point_names']==['P_RAIL_B']]
    signature=lambda row:(row['view_id'],tuple(row['uv']),row['face_id'],tuple(row['position']))
    assert [signature(row) for row in left]==[signature(row) for row in right]
    assert {row['id'] for row in left}.isdisjoint({row['id'] for row in right})


def test_resolved_marker_face_must_match_source_points_selected_host():
    mismatches=_source_marker_host_mismatches([
        ({'source_point_names':['P_FRAME'],'feature':'FRAME'}, {'face_id':'rail-face'}),
        ({'source_point_names':['P_RAIL'],'feature':'RAIL'}, {'face_id':'rail-face'})],
        {'P_FRAME':{'frame-face'},'P_RAIL':{'rail-face'}},
        {'P_FRAME':['B10'],'P_RAIL':['B24']})
    assert mismatches==[{'source_point_name':'P_FRAME','expected_part_ids':['B10'],
        'actual_face_id':'rail-face','marker_feature':'FRAME'}]


def test_candidate_id_keeps_exact_surface_hit_under_assembly_occlusion():
    candidates=[{'id':7,'view_id':'+Y','uv':[.4,.5],'face_id':'frame-face',
        'position':[1.25,2.5,3.75],'source_point_names':['P_FRAME']}]
    hit=_exact_surface_candidate_hit({'candidate_id':'C7'},candidates)
    assert hit=={'face_id':'frame-face','position':[1.25,2.5,3.75],
        'label_offset':[32,-26]}


def test_numbered_candidate_must_match_the_named_points_selected_host():
    candidates=[
        {'id':1,'face_id':'rail-face','source_point_names':['P_FRAME']},
        {'id':2,'face_id':'frame-face','source_point_names':['P_FRAME']},
        {'id':3,'face_id':'rail-face','source_point_names':['P_RAIL']}]
    parsed={'markers':[{'candidate_id':'C1','source_point_names':['P_FRAME']},
        {'candidate_id':'C3','source_point_names':['P_RAIL']} ]}
    issues=_candidate_host_selection_audit(parsed,candidates,
        {'P_FRAME':{'frame-face'},'P_RAIL':{'rail-face'}})
    assert issues==[{'field':'markers','source_point_name':'P_FRAME',
        'selected_candidate_id':'C1','valid_candidate_ids':['C2'],
        'reason':'candidate_face_outside_selected_source_host'}]


def test_only_native_exact_body_selection_is_accepted_and_never_marks_points(tmp_path):
    class Model:
        ids=['B1','B2']
        def _generate_once(self,messages,**kwargs):
            assert messages[0]['attachments'][0]['images'][0]=='original-msop.png'
            return json.dumps({'decision':'SELECT_PARTS','fai':4,'part_ids':self.ids,
                'source_feature':'FANG','evidence':'Two repeated parts match the original leader.',
                'confidence':'low'}),False,80
    model=Model();store=SimpleNamespace(_dir=lambda _:tmp_path)
    selected,audit=select_physical_parts(store,{'id':'p'}, {'number':4}, {'methods':[]},model,mesh(),['original-msop.png'])
    assert len(selected)==2 and audit['does_not_authorize_measurement_anchors']
    model.ids=['B3']
    selected,audit=select_physical_parts(store,{'id':'p'}, {'number':4}, {'methods':[]},model,mesh(),['original-msop.png'])
    assert selected is None and audit['status']=='unconfirmed'


def test_part_selection_requires_complete_per_point_host_mapping(tmp_path):
    hypothesis={'required_marker_groups':[{'source_point_names':['P_FRAME','P_RAIL']}],
        'evidence':{'point_correspondences':[
            {'name':'P_FRAME','source_xyz':[0,0,0],'position':[0,0,0],'face_id':'left'}]}}
    class Model:
        valid=True
        def _generate_once(self,messages,**kwargs):
            return json.dumps({'decision':'SELECT_PARTS','fai':4,'part_ids':['B1','B2'],
                'source_feature':'FRAME TO RAIL','evidence':'Two measured ends map to separate bodies.',
                'source_point_host_assignments':([
                    {'source_point_names':['P_FRAME'],'part_ids':['B1'],'evidence':'Frame endpoint.'},
                    {'source_point_names':['P_RAIL'],'part_ids':['B2'],'evidence':'Rail endpoint.'}]
                    if self.valid else
                    [{'source_point_names':['P_FRAME'],'part_ids':['B1'],'evidence':'Frame endpoint.'}])}),False,80
    model=Model();store=SimpleNamespace(_dir=lambda _:tmp_path)
    parts,audit=select_physical_parts(store,{'id':'p','documents':[]},{'number':4},
        {'methods':[{'text':'Measure point P_FRAME to rail point P_RAIL.'}]},model,mesh(),
        ['original-msop.png'],geometry_hypothesis=hypothesis)
    assert [part['part_id'] for part in parts]==['B1','B2']
    assert audit['decision']['source_point_host_part_ids']=={'P_FRAME':['B1'],'P_RAIL':['B2']}
    model.valid=False
    parts,audit=select_physical_parts(store,{'id':'p','documents':[]},{'number':4},
        {'methods':[{'text':'Measure point P_FRAME to rail point P_RAIL.'}]},model,mesh(),
        ['original-msop.png'],geometry_hypothesis=hypothesis)
    assert parts is None and audit['status']=='unconfirmed'


def test_part_selection_host_map_includes_all_msop_measure_groups(tmp_path):
    hypothesis={'required_marker_groups':[{'source_point_names':['R1_PNT_P2']}],
        'evidence':{'point_correspondences':[
            {'name':'R1_PNT_P2','source_xyz':[0,0,0],'position':[0,0,0],'face_id':'left'}]}}
    class Model:
        def _generate_once(self,messages,**kwargs):
            prompt=messages[0]['content']
            assert 'S1_PNT_Z1' in prompt and 'R1_PNT_Z3' in prompt
            assert '同一X/Y位置的点按Z方向成对测距' in prompt
            return json.dumps({'decision':'SELECT_PARTS','fai':4,'part_ids':['B1','B2'],
                'source_feature':'FRAME PLASTIC TO RAIL','evidence':'MSOP identifies frame and rail ends.',
                'source_point_host_assignments':[
                    {'source_point_names':['R1_PNT_P2'],'part_ids':['B1'],'evidence':'Frame point.'},
                    {'source_point_names':['S1_PNT_Z1','R1_PNT_Z3'],'part_ids':['B2'],'evidence':'Rail line points.'}]}),False,80
    parts,audit=select_physical_parts(SimpleNamespace(_dir=lambda _:tmp_path),{'id':'p','documents':[]},
        {'number':4},{'methods':[{'page':25,'text':'Measure 1 vector point R1_PNT_P2. '
            'Measure 2 vector points S1_PNT_Z1, R1_PNT_Z3 and construct a line S1_ZLIN.'}]},
        Model(),mesh(),['original-msop.png'],geometry_hypothesis=hypothesis)
    assert [part['part_id'] for part in parts]==['B1','B2']
    assert audit['decision']['source_point_host_part_ids']=={
        'R1_PNT_P2':['B1'],'S1_PNT_Z1':['B2'],'R1_PNT_Z3':['B2']}


def test_part_selection_repairs_model_renaming_a_measurement_point(tmp_path):
    hypothesis={'required_marker_groups':[{'source_point_names':['R1_PNT_P2']}],
        'evidence':{'point_correspondences':[
            {'name':'R1_PNT_P2','source_xyz':[0,0,0],'position':[0,0,0],'face_id':'left'}]}}
    class Model:
        calls=0
        def _generate_once(self,messages,**kwargs):
            self.calls+=1
            rail_names=(['S1_PNT_Z1','S1_PNT_Z3'] if self.calls==1 else
                        ['S1_PNT_Z1','R1_PNT_Z3'])
            return json.dumps({'decision':'SELECT_PARTS','fai':4,'part_ids':['B1','B2'],
                'source_feature':'FRAME PLASTIC TO RAIL','evidence':'The method defines plastic and rail endpoints.',
                'source_point_host_assignments':[
                    {'source_point_names':['R1_PNT_P2'],'part_ids':['B1'],'evidence':'Plastic point.'},
                    {'source_point_names':rail_names,'part_ids':['B2'],'evidence':'Rail construction points.'}]}),False,80
    model=Model()
    parts,audit=select_physical_parts(SimpleNamespace(_dir=lambda _:tmp_path),{'id':'p','documents':[]},
        {'number':4},{'methods':[{'page':25,'text':'Measure 1 vector point R1_PNT_P2. '
            'Measure 2 vector points S1_PNT_Z1, R1_PNT_Z3 and construct a line S1_ZLIN.'}]},
        model,mesh(),['original-msop.png'],geometry_hypothesis=hypothesis)
    assert model.calls==2 and [part['part_id'] for part in parts]==['B1','B2']
    assert audit['decision']['source_point_host_part_ids']['R1_PNT_Z3']==['B2']
    assert [row['host_map_valid'] for row in audit['host_assignment_attempts']]==[False,True]


def test_measurement_markers_cannot_escape_native_selected_body(tmp_path,monkeypatch):
    from oracle_viewer import local_matching
    from oracle_viewer.cad_view_atlas import project_point_to_view_uv
    m=mesh();store=SimpleNamespace(_dir=lambda _:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-1.png'])
    class Model:
        def info(self):return {'image_inference':'strata_native_vision'}
        def _generate_once(self,messages,**kwargs):
            return json.dumps({'decision':'AUTO_MARK','fai':1,'markers':[{
                'view_id':'+Z','uv':project_point_to_view_uv(m,'+Z',[10.2,.2,0])[0],
                'feature':'CIR','source_pages':[1]}],'evidence':'Model chose this feature.','confidence':'low'}),False,80
    result=local_matching._mark_visual_fallback(store,{'id':'p','documents':[]},
        {'number':1,'document_id':'doc','pages':[1]},
        {'methods':[{'page':1,'text':'FAI 1 circle location.'}],'dependencies':[]},Model(),m,
        {'evidence':{'part_identity_context':{'part_ids':['B1'],'cad_sha256':'test'}}})
    assert not store.annotations and result['result']=='abstain_or_invalid'
    assert result['geometry_diagnostic']['reason']=='marker_outside_local_model_selected_parts'


def test_large_body_catalog_is_screened_completely_without_truncation(tmp_path):
    m={'source_sha256':'large','faces':[{'id':f'f{i}','body':i,
        'positions':[i*2,0,0,i*2+1,0,0,i*2,1,0],'indices':[0,1,2]} for i in range(1,70)]}
    seen=[]
    prompts=[]
    class Model:
        def _generate_once(self,messages,**kwargs):
            prompts.append(messages[0]['content'])
            payload=json.loads(messages[0]['attachments'][0]['parsed_text'])
            if 'all_original_bodies' in payload:
                assert len(payload['all_original_bodies'])==69
                return json.dumps({'decision':'SELECT_HOST_PARTS','fai':4,'part_ids':['B1'],
                    'evidence':'Global source and host topology comparison.'}),False,40
            if 'visible_parts' in payload:
                ids=[p['part_id'] for p in payload['visible_parts']];seen.extend(ids)
                assert '主体零件' in messages[0]['content']
                return json.dumps({'decision':'SCREEN_PARTS','fai':4,
                    'part_ids':['B1'] if 'B1' in ids else [],'evidence':'Host geometry comparison.'}),False,40
            return json.dumps({'decision':'SELECT_PARTS','fai':4,'part_ids':['B1'],
                'source_feature':'Rail','evidence':'Original leader and host shape.','confidence':'low'}),False,40
    parts,audit=select_physical_parts(SimpleNamespace(_dir=lambda _:tmp_path),{'id':'p'},
        {'number':4},{'methods':[]},Model(),m,['original-msop.png'])
    assert parts[0]['part_id']=='B1' and len(seen)==69 and len(set(seen))==69
    assert len(audit['screening'])==4 and audit['does_not_authorize_measurement_anchors']
    assert all('A TO B' in prompt and '两端' in prompt for prompt in prompts)


def test_unconfirmed_part_screening_persists_failure_evidence_for_downstream_review(tmp_path):
    m={'source_sha256':'large','faces':[{'id':f'f{i}','body':i,
        'positions':[i*2,0,0,i*2+1,0,0,i*2,1,0],'indices':[0,1,2]} for i in range(1,70)]}
    class Model:
        def _generate_once(self,messages,**kwargs):
            return json.dumps({'decision':'ABSTAIN','fai':4,'part_ids':[]}),False,40
    parts,audit=select_physical_parts(SimpleNamespace(_dir=lambda _:tmp_path),{'id':'p'},
        {'number':4},{'methods':[]},Model(),m,['original-msop.png'])
    assert parts is None and audit['status']=='unconfirmed'
    assert audit['receipt'] and (tmp_path/'automatic-learning'/'part-selections').is_dir()
    saved=json.load(open(audit['receipt']))
    assert saved['screening'][0]['raw_output'] and saved['does_not_authorize_measurement_anchors']


def test_invalid_body_batch_is_recorded_but_does_not_abort_full_catalog_review(tmp_path):
    m={'source_sha256':'large','faces':[{'id':f'f{i}','body':i,
        'positions':[i*2,0,0,i*2+1,0,0,i*2,1,0],'indices':[0,1,2]} for i in range(1,70)]}
    batch_number=0
    class Model:
        def _generate_once(self,messages,**kwargs):
            nonlocal batch_number
            payload=json.loads(messages[0]['attachments'][0]['parsed_text'])
            if 'visible_parts' in payload:
                batch_number+=1
                ids=[part['part_id'] for part in payload['visible_parts']]
                # The second batch hallucinates B1, which it was not shown.
                selected=(['B1'] if batch_number==1 else
                    ['B1'] if batch_number==2 else [])
                return json.dumps({'decision':'SCREEN_PARTS','fai':4,
                    'part_ids':selected,'evidence':'Shape and source comparison.'}),False,40
            if 'all_original_bodies' in payload:
                assert len(payload['all_original_bodies'])==69
                return json.dumps({'decision':'SELECT_HOST_PARTS','fai':4,'part_ids':['B2'],
                    'evidence':'Full body table and source feature comparison.'}),False,40
            return json.dumps({'decision':'SELECT_PARTS','fai':4,'part_ids':['B2'],
                'source_feature':'Rail','evidence':'Selected body matches the source feature.',
                'confidence':'low'}),False,40
    parts,audit=select_physical_parts(SimpleNamespace(_dir=lambda _:tmp_path),{'id':'p'},
        {'number':4},{'methods':[]},Model(),m,['original-msop.png'])
    assert [part['part_id'] for part in parts]==['B2']
    assert audit['screening'][0]['valid'] is True
    assert audit['screening'][1]['valid'] is False
    assert audit['screening'][2]['valid'] is True
    assert audit['screening'][3]['stage']=='global_host_comparison'
