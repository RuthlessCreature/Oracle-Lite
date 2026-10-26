from oracle_viewer.fai_strategy import _validate_batch


def classification():
    return {'decision':'CLASSIFY','items':[{'fai':1,'measurement_family':'flatness',
        'evidence_class':'complete_xyz_single_frame','failure_class':'frame_registration',
        'next_route':'datum_graph_reconstruction','source_quotes':[{'page':3,'text':'Measure 4 vector points'}],
        'reason':'先核对原始基准关系，再校验完整点组。'}]}


def sources():
    return [{'fai':1,'methods':[{'page':3,'text':'Measure 4 vector points Q1~Q4. Flatness.'}],
        'outcome':{'status':'auto_marked','is_verified_ground_truth':False}}]


def test_exact_local_categories_retain_source_and_do_not_promote_marked_cases_to_truth():
    valid,items=_validate_batch(classification(),sources())
    assert valid and items[0]['next_route']=='datum_graph_reconstruction'
    assert not items[0]['outcome_at_classification']['is_verified_ground_truth']
    assert items[0]['source_pages']==[3]


def test_classification_refuses_invented_quote_wrong_fai_and_unsupported_routes():
    for key,value in [('fai',2),('next_route','force_nearest_face'),
                      ('source_quotes',[{'page':3,'text':'Measure 8 vector points'}])]:
        result=classification();result['items'][0][key]=value
        assert not _validate_batch(result,sources())[0]
    result=classification();result['items']*=2
    assert not _validate_batch(result,sources())[0]


def test_absent_entity_is_routed_to_applicability_never_synthetic_geometry():
    result=classification();row=result['items'][0]
    row['evidence_class']='explicit_absent_entity'
    assert not _validate_batch(result,sources())[0]
    row['next_route']='assembly_applicability'
    assert _validate_batch(result,sources())[0]


def test_classification_calls_native_interface_and_caches_only_identical_source_input(tmp_path,monkeypatch):
    import copy
    from types import SimpleNamespace
    from oracle_viewer import fai_strategy,local_matching
    project={'model':{'source_sha256':'cad'}};msop={'sha256':'msop'}
    monkeypatch.setattr(fai_strategy,'collect_strategy_sources',lambda *_:(project,msop,copy.deepcopy(sources())))
    monkeypatch.setattr(local_matching,'_is_local_inference_model',lambda _:True)
    store=SimpleNamespace(_dir=lambda _:tmp_path,get=lambda _:project,active_msop=lambda _:msop)
    calls=[]
    class Model:
        loaded=True
        def info(self):return {'inference_backend':'strata-native','model_alias':'local-test'}
        def _generate_once(self,messages,*,max_context_chars,max_new_tokens,max_images,temperature):
            import json
            calls.append(messages);assert max_images==0
            return json.dumps(classification()),False,80
    first=fai_strategy.classify_project(store,'p',Model())
    assert first['status']=='completed' and len(first['items'])==1 and len(calls)==1
    second=fai_strategy.classify_project(store,'p',Model())
    assert second['status']=='completed' and len(calls)==1


def test_similarly_named_coordinate_rows_cannot_make_missing_declared_points_complete():
    source=sources();source[0]['missing_xyz_for_declared_points']=['Q1']
    assert not _validate_batch(classification(),source)[0]
    result=classification();result['items'][0]['evidence_class']='named_features_no_xyz'
    result['items'][0]['next_route']='feature_topology_graph'
    assert _validate_batch(result,source)[0]


def test_independent_fai_classifications_are_saved_while_only_invalid_cases_are_reasked(tmp_path,monkeypatch):
    import copy,json
    from types import SimpleNamespace
    from oracle_viewer import fai_strategy,local_matching
    project={'model':{'source_sha256':'cad'}};msop={'sha256':'msop'}
    input_rows=sources()+[{**sources()[0],'fai':2}]
    monkeypatch.setattr(fai_strategy,'collect_strategy_sources',lambda *_:(project,msop,copy.deepcopy(input_rows)))
    monkeypatch.setattr(local_matching,'_is_local_inference_model',lambda _:True)
    store=SimpleNamespace(_dir=lambda _:tmp_path,get=lambda _:project,active_msop=lambda _:msop)
    calls=[]
    class Model:
        loaded=True
        def info(self):return {'inference_backend':'strata-native','model_alias':'test'}
        def _generate_once(self,messages,**kwargs):
            supplied=json.loads(messages[0]['attachments'][0]['parsed_text']);calls.append([r['fai'] for r in supplied])
            rows=[]
            for source in supplied:
                row=classification()['items'][0];row['fai']=source['fai']
                if len(calls)==1 and source['fai']==2:row['source_quotes'][0]['text']='Invented quote'
                rows.append(row)
            return json.dumps({'decision':'CLASSIFY','items':rows}),False,80
    result=fai_strategy.classify_project(store,'p',Model())
    assert result['status']=='completed' and len(result['items'])==2
    assert calls==[[1,2],[2]]


def test_failure_history_survives_counter_reset_but_is_never_reused_across_cad_sources(tmp_path):
    from oracle_viewer.store import ViewerStore
    from oracle_viewer.msop import parse_page
    from oracle_viewer.fai_strategy import archive_marking_state,collect_strategy_sources
    store=ViewerStore(tmp_path/'state');project=store.create('generic','drawing','A')
    project['documents']=[{'id':'doc','role':'msop','sha256':'msop','filename':'method.pdf',
        'page_count':1,'method_parser_version':2,
        'method_pages':[parse_page('MSOP: FAI 1\nFlatness. Measure 4 vector points P1~P4.',1)]}]
    project['model']={'source_sha256':'cad-old'};store._write(project);pid=project['id']
    archive_marking_state(store,pid,{'status':'partial','updated_at':'2026-10-16T00:00:00',
        'items':[{'fai':1,'reason':'model_visual_decision_invalid','validation_failures':[{'code':'source_point_coverage_incomplete'}]}]})
    _,_,rows=collect_strategy_sources(store,pid)
    assert rows[0]['outcome']['failure_reason']=='model_visual_decision_invalid'
    assert rows[0]['outcome']['historical_diagnostic']
    assert not rows[0]['outcome']['historical_diagnostic_is_geometry_authority']
    project=store.get(pid);project['model']['source_sha256']='cad-new';store._write(project)
    _,_,rows=collect_strategy_sources(store,pid)
    assert rows[0]['outcome']['failure_reason'] is None


def test_classified_complete_xyz_failure_does_not_reenter_pixel_guessing(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from oracle_viewer import local_matching
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-1.png'])
    monkeypatch.setattr(local_matching,'_mark_model_selected_xyz_group',lambda *args:{
        'fai':1,'result':'abstain_or_invalid','reason':'no_fai_local_frame_candidate_passed_geometry_gate'})
    store=SimpleNamespace(_dir=lambda _:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    class Model:
        def info(self):return {'image_inference':'strata_native_vision'}
        def _generate_once(self,*args,**kwargs):raise AssertionError('Classified XYZ must not fall back to pixel guesses')
    mesh={'source_sha256':'cad','faces':[{'id':'f','positions':[0,0,0,1,0,0,0,1,0],'indices':[0,1,2]}]}
    candidate={'required_marker_count':2,'known_source_points':[{'name':'P1','xyz':[0,0,0]},
        {'name':'P2','xyz':[1,0,0]}],'evidence':{'fai_strategy':{'evidence_class':'complete_xyz_single_frame',
            'next_route':'rigid_full_group'}}}
    result=local_matching._mark_visual_fallback(store,{'id':'p','documents':[]},{'number':1,'pages':[1]},
        {'methods':[{'page':1,'text':'FAI 1 Flatness.'}],'dependencies':[]},Model(),mesh,candidate)
    assert result['classified_route_no_pixel_guessing'] and not store.annotations


def test_cache_identity_is_order_independent_for_point_rows_but_preserves_xyz_axes():
    from oracle_viewer.fai_strategy import strategy_source_sha256
    first={'fai':1,'source_points':[{'name':'P1','xyz':[1,2,3],'pages':[2,1]},
        {'name':'P2','xyz':[4,5,6]}]}
    second={'fai':1,'source_points':[{'name':'P2','xyz':[4,5,6]},
        {'name':'P1','xyz':[1,2,3],'pages':[1,2]}]}
    assert strategy_source_sha256(first,{'cad':'c'})==strategy_source_sha256(second,{'cad':'c'})
    second['source_points'][1]['xyz']=[3,2,1]
    assert strategy_source_sha256(first,{'cad':'c'})!=strategy_source_sha256(second,{'cad':'c'})


def test_xyz_route_trials_require_same_cad_and_msop(tmp_path):
    import json
    from types import SimpleNamespace
    from oracle_viewer.fai_strategy import strategy_state
    root=tmp_path/'automatic-learning';root.mkdir()
    project={'model':{'source_sha256':'cad'}}
    store=SimpleNamespace(_dir=lambda _:tmp_path,get=lambda _:project,active_msop=lambda _:{'sha256':'msop'})
    (root/'fai-strategies.json').write_text(json.dumps({'cad_sha256':'cad','msop_sha256':'msop','items':[]}))
    trial={'cad_sha256':'cad','msop_sha256':'msop','results':[{'fai':59,'source_point_count':48}]}
    (root/'xyz-route-trials.json').write_text(json.dumps(trial))
    assert strategy_state(store,'p')['route_trials'][0]['source_point_count']==48
    trial['msop_sha256']='other'
    (root/'xyz-route-trials.json').write_text(json.dumps(trial))
    assert not strategy_state(store,'p').get('route_trials')
