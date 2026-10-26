import json

from oracle_viewer.local_matching import _request_visual_marker_chunk,_merge_visual_marker_chunks


def result(names,decision='AUTO_MARK'):
    return {'decision':decision,'fai':8,'confidence':'low','evidence':'Original point table.',
        'markers':[{'source_point_names':[name],'view_id':'+Z','uv':[.2,.3]} for name in names]}


def test_native_model_repairs_extra_point_without_service_trimming_or_neighbor_regeneration():
    calls=[]
    class Model:
        def _generate_once(self,messages,**kwargs):
            calls.append(messages)
            assert messages[0]['attachments'][0]['parsed_text'].endswith('LAST_SOURCE_ROW')
            if len(calls)==2:
                assert 'outside_chunk' in messages[0]['content']
                assert 'P9' in messages[0]['content']
            return json.dumps(result(['P1','P2','P9'] if len(calls)==1 else ['P1','P2'])),False,100
    parsed,raw,truncated,tokens,aliases,audit=_request_visual_marker_chunk(Model(),{
        'role':'user','content':'Complete original FAI instructions.',
        'attachments':[{'filename':'Original source','parsed_text':'Full uncut evidence LAST_SOURCE_ROW'}]},
        fai=8,expected_names=['P1','P2'],source_points=[{'name':n} for n in ['P1','P2','P9']],
        allowed_source_pages={1})
    assert len(calls)==2 and len(audit)==2 and tokens==200 and not truncated
    assert parsed['markers']==result(['P1','P2'])['markers']
    assert audit[0]['contract_issues'][0]['outside_chunk']==['P9']
    assert json.loads(audit[0]['raw_output'])['markers'][-1]['source_point_names']==['P9']
    merged=_merge_visual_marker_chunks([parsed,result(['P9'])],[['P1','P2'],['P9']],fai=8)
    assert merged['decision']=='AUTO_MARK' and not merged['_chunk_failures']


def test_native_abstention_is_not_retried_into_acceptance():
    calls=[]
    class Model:
        def _generate_once(self,*args,**kwargs):
            calls.append(1);return json.dumps(result([],decision='ABSTAIN')),False,20
    parsed,*rest=_request_visual_marker_chunk(Model(),{'role':'user','content':'Original source'},
        fai=8,expected_names=['P1'],source_points=[{'name':'P1'}],allowed_source_pages={1})
    assert len(calls)==1 and parsed['decision']=='ABSTAIN'


def test_exhausted_repairs_never_authorize_or_silently_remove_an_extra_point():
    class Model:
        def _generate_once(self,*args,**kwargs):
            return json.dumps(result(['P1','P2'])),False,20
    parsed,raw,truncated,tokens,aliases,audit=_request_visual_marker_chunk(Model(),
        {'role':'user','content':'Original source'},fai=8,expected_names=['P1'],
        source_points=[{'name':'P1'},{'name':'P2'}],allowed_source_pages={1})
    assert len(audit)==3 and len(parsed['markers'])==2
    merged=_merge_visual_marker_chunks([parsed],[['P1']],fai=8)
    assert merged['decision']=='ABSTAIN'


def test_invalid_candidate_id_is_returned_to_native_model_not_reassigned_by_service():
    calls=[]
    class Model:
        def _generate_once(self,messages,**kwargs):
            calls.append(messages)
            row=result(['P1']);row['markers'][0]['candidate_id']='C2' if len(calls)==1 else 'C1'
            return json.dumps(row),False,30
    parsed,_,_,_,_,audit=_request_visual_marker_chunk(Model(),{'role':'user','content':'Original source'},
        fai=8,expected_names=['P1'],source_points=[{'name':'P1'},{'name':'P2'}],allowed_source_pages={1},
        surface_candidates=[{'id':1,'source_point_names':['P1']},{'id':2,'source_point_names':['P2']}])
    assert len(calls)==2 and audit[0]['contract_issues'][0]['selected_candidate_id']=='C2'
    assert parsed['markers'][0]['candidate_id']=='C1'


def test_coordinate_variant_batches_do_not_import_other_names_from_complete_source_context():
    from oracle_viewer.local_matching import _select_coordinate_variants_with_local_model
    points=[{'name':f'P{i}','requires_variant_choice':True,'coordinate_variants':[
        {'page':1,'program':'UP','coordinate_system':'A','xyz':[i,0,0]},
        {'page':2,'program':'DOWN','coordinate_system':'B','xyz':[i,0,1]}]} for i in range(7)]
    calls=[]
    class Model:
        def _generate_once(self,messages,**kwargs):
            text=messages[0]['content'];calls.append(text)
            assert 'COMPLETE_SOURCE_LAST_ROW' in text
            expected=json.loads(text.split('CURRENT_OUTPUT_POINT_NAMES: ',1)[1].split('\n',1)[0])
            names=[row['name'] for row in points] if len(calls)==1 else expected
            if len(calls)==2:
                assert 'outside_current_subset' in text and 'P6' in text
            choices=[{'name':name,**next(p for p in points if p['name']==name)['coordinate_variants'][0]}
                for name in names]
            return json.dumps({'decision':'SELECT_VARIANTS','fai':8,'coordinate_choices':choices,
                'confidence':'low','evidence':'Original row definitions.'}),False,100
    choices,audit=_select_coordinate_variants_with_local_model(Model(),8,points,
        'All seven source points remain in context. COMPLETE_SOURCE_LAST_ROW')
    assert audit['status']=='selected' and len(choices)==7 and len(calls)==3
    assert len(audit['chunks'][0]['attempts'])==2
    assert json.loads(audit['chunks'][0]['attempts'][0]['raw_output'])['coordinate_choices'][-1]['name']=='P6'
