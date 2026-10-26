from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from oracle_viewer import local_matching
from oracle_viewer.local_matching import (_method_page_images, _compact_registration_diagnostic,
    _resolve_surface_candidate_ids, _surface_candidate_selection_valid,
    _surface_candidate_selection_audit,
    _visual_marker_validation_issues, _complete_marker_registration_references)
from oracle_viewer.local_matching import (_xyz_rigid_fit_required_scopes,
    _missing_required_xyz_fits, _normalize_model_source_aliases,
    _explicit_source_point_relations, _partition_visual_source_points, _merge_visual_marker_chunks,
    _select_coordinate_variants_with_local_model, _request_visual_marker_chunk, _attach_explicit_source_xy,
    _explicit_coincident_xy_z_distance_pairs)
from oracle_viewer.msop_coordinates import _explicit_measure_point_names


def test_viewer_location_prompt_distinguishes_geometric_measurement_points_from_test_objects():
    prompt=local_matching.VIEWER_LOCATION_TASK
    assert '平面度、直线度、轮廓度、位置度' in prompt
    assert '每个测量向量点都必须映射到 CAD' in prompt
    assert '这些测点不是焊点、螺母或力学试验对象' in prompt
    assert '原文XYZ、坐标系、实体与整组几何校验仍须一致' in prompt


def test_global_surface_hits_keep_one_id_and_list_all_eligible_source_points():
    candidates=[{'view_id':'+Z','uv':[.25,.4],'position':[1.0,2.0,3.0],
        'face_id':'face-1','source_point_names':[]}]
    scoped=local_matching._scope_global_surface_candidates(
        candidates,['P1','P2','P3'],first_id=40)
    assert [row['id'] for row in scoped]==[40]
    assert [row['source_point_names'] for row in scoped]==[['P1','P2','P3']]
    assert all(row['position']==[1.0,2.0,3.0] for row in scoped)


def test_large_joint_candidate_pool_stays_inside_strata_context_budget():
    names=[f'DATUMA_U{i:02d}' for i in range(1,11)]
    candidates=[]
    views=('+Z','-Z','+X','-X','+Y','-Y')
    for index in range(144):
        candidates.append({'id':index+1,'view_id':views[index%6],
            'uv':[.1+(index%12)*.065,.1+(index//12)*.065],
            'position':[float(index),float(index%9),float(index%5)],
            'face_id':f'face-{index}','source_point_names':names,
            'candidate_role':'global_surface_search',
            'blocked_uvs_by_source_name':{name:[{'view_id':'+Z','uv':[.5,.5]}]
                for name in names}})
    source_points=[{'name':name,'xyz':[float(i),float(i%7),2.5],
        'program':'UP','coordinate_system':'ALI_ABC'}
        for i,name in enumerate(names,1)]
    class Model:
        called=False
        def _generate_once(self,messages,**kwargs):
            self.called=True
            assert kwargs['max_context_chars']<58000
            assert 'candidate_options' in messages[0]['content']
            return json.dumps({'decision':'ABSTAIN','fai':7,'markers':[]}),False,32
    model=Model()
    parsed,_,truncated,_,_,attempts=local_matching._request_visual_marker_chunk(
        model,{'role':'user','content':'FAI 7 source evidence',
            'attachments':[{'filename':'MSOP and drawing','parsed_text':'source pages','images':['msop.png']}]},
        fai=7,expected_names=names,source_points=source_points,
        allowed_source_pages={26,88,89},surface_candidates=candidates,
        global_source_points=source_points)
    assert model.called and not truncated and parsed['decision']=='ABSTAIN'
    assert not any('Complete local visual evidence exceeds context budget' in row.get('error','')
        for row in attempts)


def test_retry_context_budget_preserves_all_source_text_or_refuses_the_call():
    from talker.model import TalkerModel
    messages=[{'role':'user','content':'Initial task. '+('Additional candidate diagnostics. '*200),
        'attachments':[{'filename':'Original source','parsed_text':
            'Complete point group. '*500+'LAST_ORIGINAL_SOURCE_POINT'}]}]
    budget=local_matching._complete_visual_context_budget(messages)
    prompt,_=TalkerModel.__new__(TalkerModel)._context(messages,budget,max_images=0)
    assert 'LAST_ORIGINAL_SOURCE_POINT' in prompt
    assert 'excerpt incomplete' not in prompt
    with pytest.raises(ValueError,match='no truncated decision accepted'):
        local_matching._complete_visual_context_budget(messages,limit=100)


def test_compact_visual_retry_keeps_complete_source_attachment_and_candidate_constraints():
    source='Complete MSOP XYZ evidence. '+('PNT [1,2,3] preserved. '*2100)+'LAST_SOURCE_POINT'
    original={'role':'user','content':'Verbose repeated prompt. '*900,
        'attachments':[{'filename':'Original MSOP and drawing','parsed_text':source,
                        'images':['msop.png','drawing.png']}]}
    compact=local_matching._compact_visual_evidence_message(original,fai=6,
        source_points=[{'name':'N2_PNT1','xyz':[1,2,3]}],source_pages=[45],
        required_marker_groups=[{'group_id':'N2','declared_count':1,'source_point_names':['N2_PNT1']}],
        surface_candidates=[{'id':7,'view_id':'-X','uv':[.4,.6],
            'source_point_names':['N2_PNT1'],
            'blocked_uvs_by_source_name':{'N2_PNT1':[{'view_id':'+Z','uv':[.2,.4]}]}}],
        failure_summary=[{'code':'collapsed_source_point_view_forbidden'}])
    assert local_matching._complete_visual_context_budget([compact]) < 58000
    assert compact['attachments'][0]['parsed_text']==source
    assert compact['attachments'][0]['images']==['msop.png','drawing.png']
    assert 'LAST_SOURCE_POINT' in compact['attachments'][0]['parsed_text']
    assert 'Deduplicated blocked exact ray hits by source point' in compact['content']
    assert '"N2_PNT1"' in compact['content'] and '"uv":[0.2,0.4]' in compact['content']


def test_fai_retry_compaction_moves_complete_xyz_into_prompt_and_drops_duplicate_pdf_transcript():
    original={'role':'user','content':'Verbose FAI prompt. '*1800,
        'attachments':[{'filename':'Original MSOP and drawing','parsed_text':
            'Duplicated full extraction. '*3000,'images':['method.png','drawing.png','cad-candidates.png']}]}
    points=[{'name':'DATUMA_U01','xyz':[1.0,2.0,3.0],'program':'UP','coordinate_system':'ALI_ABC',
             'pages':[88],'coordinate_variants':[{'page':88,'program':'UP',
                'coordinate_system':'ALI_ABC','xyz':[1.0,2.0,3.0]}]},
            {'name':'DATUMA_U02','xyz':[2.0,2.0,3.0],'program':'UP','coordinate_system':'ALI_ABC',
             'pages':[88]}]
    compact=local_matching._compact_visual_evidence_message(original,fai=7,
        source_points=[points[0]],source_pages=[88],layout_source_points=points,
        method_evidence='Measure two points on Datum A.',
        dependency_evidence='Datum A is established from the complete point group.',
        surface_candidates=[{'id':4,'view_id':'-Z','uv':[.4,.5],
            'source_point_names':['DATUMA_U01']}])
    attachment=compact['attachments'][0]
    assert 'parsed_text' not in attachment
    assert attachment['images']==['method.png','drawing.png','cad-candidates.png']
    assert 'DATUMA_U02' in compact['content'] and '[2.0,2.0,3.0]' in compact['content']
    assert 'coordinate_variants' not in compact['content'] and 'ALI_ABC' in compact['content']
    assert 'Measure two points on Datum A.' in compact['content']
    assert 'Datum A is established' in compact['content']
    assert local_matching._complete_visual_context_budget([compact]) < 58000


def test_fai_retry_keeps_genuinely_conflicting_xyz_variant():
    point={'name':'PNT_A','xyz':[1,2,3],'program':'UP','coordinate_system':'ALI_ABC',
        'coordinate_variants':[{'page':10,'program':'UP','coordinate_system':'ALI_ABC','xyz':[1,2,3]},
            {'page':11,'program':'DOWN','coordinate_system':'ALI_ABD','xyz':[1,2,4]}]}
    compact=local_matching._compact_visual_evidence_message({'role':'user','content':'verbose'},
        fai=1,source_points=[point],source_pages=[10,11],layout_source_points=[point])
    assert 'PNT_A' in compact['content'] and '[1,2,4]' in compact['content']
    assert 'ALI_ABD' in compact['content'] and 'coordinate_variants' in compact['content']


def test_fai_retry_does_not_expand_all_point_pairs_or_repeat_identical_coordinate_variants():
    points=[{'name':f'DATUMA_U{i:02d}','xyz':[float(i),float(i%7),2.5],
        'program':'UP','coordinate_system':'ALI_ABC','coordinate_variants':[
            {'page':26,'program':'UP','coordinate_system':'ALI_ABC',
             'xyz':[float(i),float(i%7),2.5]},
            {'page':27,'program':'UP','coordinate_system':'ALI_ABC',
             'xyz':[float(i),float(i%7),2.5]}]} for i in range(1,36)]
    original={'role':'user','content':'x'*61000,
        'attachments':[{'filename':'MSOP and drawing','parsed_text':'y'*30000,
                        'images':['method.png','drawing.png']}]}
    compact=local_matching._compact_visual_evidence_message(original,fai=7,
        source_points=points[:10],source_pages=[26,27],layout_source_points=points,
        method_evidence='datum and point method evidence',
        dependency_evidence='datum frame dependency evidence')
    encoded=compact['content']
    assert all(point['name'] in encoded for point in points)
    assert 'coordinate_variants' not in encoded
    assert 'pairwise_xy_distances' not in encoded
    assert local_matching._complete_visual_context_budget([compact]) < 58000


def test_chunk_retry_compaction_only_includes_candidates_for_its_source_points():
    source={'role':'user','content':'large repeated retry '+('x'*60000),
        'attachments':[{'filename':'MSOP','parsed_text':'full source text','images':['m.png']} ]}
    compact=local_matching._compact_visual_evidence_message(source,fai=10,
        source_points=[{'name':'CWS_S01'}],source_pages=[73],
        surface_candidates=[{'id':1,'view_id':'+Z','uv':[.2,.3],
            'source_point_names':['CWS_S01']},
            {'id':2,'view_id':'-Z','uv':[.6,.7],
             'source_point_names':['CWS_N14']}])
    assert 'CWS_S01' in compact['content'] and 'CWS_N14' not in compact['content']
    assert compact['attachments'][0]['parsed_text']=='full source text'


@pytest.mark.parametrize('decision',['success','pass','accepted','accept','approve','approved','proceed','placed'])
def test_positive_local_model_decision_alias_requires_exact_complete_point_coverage(decision):
    complete={'decision':decision,'markers':[
        {'point_name':'DATUMA_U01','candidate_id':'C1'},
        {'point_name':'DATUMA_U02','candidate_id':'C2'}],
        'registration_points':['DATUMA_U01','DATUMA_U02']}
    normalized=local_matching._normalize_positive_visual_chunk_decision(
        complete,['DATUMA_U01','DATUMA_U02'])
    assert normalized['decision']=='AUTO_MARK'
    assert normalized['registration_points']==[]
    assert normalized['confidence']=='low'
    assert all(isinstance(marker.get('source_point_names'),list) for marker in normalized['markers'])
    incomplete={'decision':decision,'markers':[{'point_name':'DATUMA_U01'}]}
    assert local_matching._normalize_positive_visual_chunk_decision(
        incomplete,['DATUMA_U01','DATUMA_U02'])['decision']==decision
    rejected={'decision':decision,'markers':[{'point_name':'DATUMA_U01','status':'rejected'}]}
    assert local_matching._normalize_positive_visual_chunk_decision(
        rejected,['DATUMA_U01'])['decision']==decision


def test_compact_whole_group_geometry_feedback_preserves_failure_fields_only():
    compact=local_matching._compact_visual_failure_feedback([
        {'code':'registration_reference_audit_failed','details':[
            {'source_point_name':'DATUMA_U02','reason':'rigid_fit_residual_exceeded',
             'residual_mm':1.25,'max_residual_mm':0.75,'private_dump':'x'*400},
            {'source_point_name':'DATUMA_U03','reason':'candidate_hit_mismatch','candidate_id':'C17'}]},
        {'code':'source_point_coverage_incomplete','missing_names':['DATUMA_U04'],'missing_count':1}])
    encoded=json.dumps(compact)
    assert 'registration_reference_audit_failed' in encoded
    assert 'DATUMA_U02' in encoded and '1.25' in encoded and 'C17' in encoded
    assert 'DATUMA_U04' in encoded and 'private_dump' not in encoded


def test_compact_geometry_feedback_keeps_names_of_every_missed_ray_point():
    compact=local_matching._compact_visual_failure_feedback([{
        'reason':'marker_view_ray_missed_CAD','view_id':'+Z','missed_marker_points':[
            {'source_point_names':['DATUMA_U06'],'view_id':'+Z','uv':[.2,.3]},
            {'source_point_names':['DATUMA_U16'],'view_id':'+Z','uv':[.7,.3]}]}])
    assert compact[0]['missed_marker_points'][0]['source_point_names']==['DATUMA_U06']
    assert compact[0]['missed_marker_points'][1]['source_point_names']==['DATUMA_U16']
    assert compact[0]['missed_marker_points'][0]['uv']==[.2,.3]


def test_positive_visual_aliases_are_canonicalized_only_for_complete_subset():
    parsed={'decision':'pass','markers':[
        {'source_point':'DATUMA_U01','cad_id':'C12','view_id':'+Z','uv':[.2,.3],
         'cad_xyz_mm':[1,2,3],'source_page':26,'role':'registration_point'},
        {'source_point':'DATUMA_U02','cad_id':'C14','view_id':'-X','uv':[.4,.5],
         'source_page':26}], 'confidence':'medium'}
    result=local_matching._normalize_positive_visual_chunk_decision(
        parsed,['DATUMA_U01','DATUMA_U02'])
    assert result['decision']=='AUTO_MARK'
    assert result['markers'][0]=={'source_point_names':['DATUMA_U01'],'candidate_id':'C12',
        'view_id':'+Z','uv':[.2,.3],'source_pages':[26]}
    assert result['markers'][1]['source_point_names']==['DATUMA_U02']
    assert 'cad_xyz_mm' not in result['markers'][0]
    assert result['markers'][0]['candidate_id']=='C12'


def test_compact_geometry_feedback_is_bounded_without_dropping_failure_code():
    details=[{'source_point_name':f'DATUMA_U{i:02d}','reason':'rigid_fit_residual_exceeded',
        'residual_mm':i/10,'max_residual_mm':.75,'message':'offset explanation '*10}
        for i in range(1,80)]
    compact=local_matching._compact_visual_failure_feedback([
        {'code':'registration_reference_audit_failed','details':details}],max_chars=1800)
    encoded=json.dumps(compact,ensure_ascii=False,separators=(',',':'))
    assert len(encoded)<=1800
    assert 'registration_reference_audit_failed' in encoded
    assert 'DATUMA_U01' in encoded and 'DATUMA_U79' not in encoded


def test_candidate_geometry_compaction_is_scoped_to_the_current_model_chunk():
    candidates=[
        {'id':1,'view_id':'+Z','uv':[.3,.4],'position':[1,2,3],
         'source_point_names':['P1','P2']},
        {'id':2,'view_id':'-X','uv':[.7,.8],'position':[4,5,6],
         'source_point_names':['P3']},
    ]
    evidence=local_matching._surface_candidate_geometry_evidence(candidates,['P1'])
    encoded=json.dumps(evidence)
    assert 'C1' in encoded and 'P1' in encoded
    assert 'C2' not in encoded and '[4.0, 5.0, 6.0]' not in encoded


def test_explicit_same_xy_z_distance_points_require_a_side_view_for_each_location():
    source=[{'name':'DATUMA_U09','xyz':[4,7,2], 'program':'DOWN','coordinate_system':'ALI_ABD'},
        {'name':'DATUMA_U27','xyz':[4,7,9], 'program':'DOWN','coordinate_system':'ALI_ABD'}]
    markers=[{'source_point_names':['DATUMA_U09'],'view_id':'+Z','uv':[.4,.5]},
        {'source_point_names':['DATUMA_U27'],'view_id':'+Z','uv':[.4,.5]}]
    pair=[{'source_point_names':['DATUMA_U09','DATUMA_U27'],'source_xy':[4,7],'axis':'Z'}]
    issues=local_matching._visual_marker_validation_issues(
        {'decision':'AUTO_MARK','fai':7,'markers':markers},fai=7,
        required_marker_count=2,known_source_points=source,rigid_group_possible=True,
        candidate_selection_valid=True,coordinate_choices_valid=True,truncated=False,
        coincident_xy_z_pairs=pair)
    collapsed=next(row for row in issues if row['code']=='coincident_xy_z_pair_view_collapsed')
    assert collapsed['details']==[{'source_point_names':['DATUMA_U09','DATUMA_U27'],
        'selected_views':['+Z','+Z'],'reason':'z_distance_pair_requires_side_view'}]
    markers[0].update(view_id='-X',uv=[.2,.5])
    markers[1].update(view_id='-X',uv=[.7,.5])
    assert not any(row['code']=='coincident_xy_z_pair_view_collapsed'
        for row in local_matching._visual_marker_validation_issues(
            {'decision':'AUTO_MARK','fai':7,'markers':markers},fai=7,
            required_marker_count=2,known_source_points=source,rigid_group_possible=True,
            candidate_selection_valid=True,coordinate_choices_valid=True,truncated=False,
            coincident_xy_z_pairs=pair))


def test_same_xy_z_distance_retry_feedback_keeps_pair_names_and_selected_views():
    compact=local_matching._compact_visual_failure_feedback([
        {'code':'coincident_xy_z_pair_view_collapsed','details':[
            {'source_point_names':['DATUMA_U09','DATUMA_U27'],
             'selected_views':['+Z','+Z'],'reason':'z_distance_pair_requires_side_view'}]}])
    assert compact[0]['details'][0]=={
        'source_point_names':['DATUMA_U09','DATUMA_U27'],
        'selected_views':['+Z','+Z'],'reason':'z_distance_pair_requires_side_view'}


def test_chunk_point_repair_replaces_only_failed_local_model_points_and_preserves_groups():
    previous={'decision':'AUTO_MARK','fai':7,'confidence':'medium','evidence':'first pass',
        'markers':[{'source_point_names':['P1'],'candidate_id':'C1'},
            {'source_point_names':['P2','P3'],'candidate_id':'C2'},
            {'source_point_names':['P4'],'candidate_id':'C4'}],
        'registration_points':[{'source_point_name':name,'uv':[.1,.2]} for name in ('P1','P2','P3','P4')],
        'coordinate_choices':[],'source_pages':[88]}
    repaired={'decision':'AUTO_MARK','fai':7,'confidence':'low','evidence':'repaired pair',
        'markers':[{'source_point_names':['P2'],'candidate_id':'C7'},
            {'source_point_names':['P3'],'candidate_id':'C8'}],
        'registration_points':[{'source_point_name':'P2','uv':[.2,.3]}],
        'coordinate_choices':[],'source_pages':[89]}
    merged,error=local_matching._merge_visual_chunk_point_repair(previous,repaired,{'P2'})
    assert error is None
    assert [row['candidate_id'] for row in merged['markers']]==['C1','C4','C7','C8']
    assert merged['registration_points']==[
        {'source_point_name':'P1','uv':[.1,.2]},
        {'source_point_name':'P4','uv':[.1,.2]},
        {'source_point_name':'P2','uv':[.2,.3]}]
    assert merged['source_pages']==[88,89]
    assert merged['confidence']=='low'


def test_chunk_point_repair_rejects_missing_or_extra_local_model_points():
    previous={'markers':[{'source_point_names':['P1'],'candidate_id':'C1'},
        {'source_point_names':['P2','P3'],'candidate_id':'C2'}]}
    merged,error=local_matching._merge_visual_chunk_point_repair(previous,
        {'markers':[{'source_point_names':['P2'],'candidate_id':'C2b'}]},{'P2'})
    assert merged is None and error['reason']=='repaired_chunk_source_coverage_incomplete'
    merged,error=local_matching._merge_visual_chunk_point_repair(previous,
        {'markers':[{'source_point_names':['P4'],'candidate_id':'C3'}]},{'P2'})
    assert merged is None and error['reason']=='repaired_chunk_contains_unrequested_source_name'


def test_visual_chunk_cache_keeps_exact_local_model_source_coverage_independent_of_candidate_ids():
    parsed={'decision':'AUTO_MARK','fai':'FAI 7','markers':[
        {'source_point_names':['P1'],'view_id':'+X','uv':[.2,.3]},
        {'source_point_names':['P2'],'view_id':'-Y','uv':[.7,.8]}]}
    assert local_matching._visual_chunk_source_coverage_valid(parsed,['P1','P2'],fai=7)
    parsed['markers'][1]['source_point_names']=['P1']
    assert not local_matching._visual_chunk_source_coverage_valid(parsed,['P1','P2'],fai=7)


def test_chunk_prompt_receives_prior_whole_group_geometry_failure():
    class Model:
        request=None
        def _generate_once(self,messages,**kwargs):
            self.request=messages
            return json.dumps({'decision':'AUTO_MARK','fai':7,'markers':[
                {'source_point_names':['DATUMA_U01'],'view_id':'-Z','uv':[.4,.5],
                 'feature':'DATUMA_U01','source_pages':[88]}],
                'registration_points':[],'confidence':'low','evidence':'MSOP p.88 point.'}),False,80
    model=Model()
    points=[{'name':'DATUMA_U01','xyz':[1,2,3],'program':'UP','coordinate_system':'ALI_ABC'}]
    feedback=[{'code':'registration_reference_audit_failed','details':[
        {'source_point_name':'DATUMA_U01','reason':'rigid_fit_residual_exceeded'}]}]
    parsed,*_=local_matching._request_visual_marker_chunk(model,
        {'role':'user','content':'Find the point from the source evidence.'},fai=7,
        expected_names=['DATUMA_U01'],source_points=points,global_source_points=points,
        allowed_source_pages={88},retry_feedback=feedback)
    assert parsed['decision']=='AUTO_MARK'
    assert 'registration_reference_audit_failed' in model.request[0]['content']
    assert 'DATUMA_U01' in model.request[0]['content']


def test_validation_diagnostic_name_extraction_is_exact_and_nested():
    names=local_matching._diagnostic_source_point_names([
        {'code':'candidate_id_selection_invalid','details':[
            {'source_point_name':'CWS_S01'},{'source_point_name':'CWS_NOT_A_POINT'}]},
        {'code':'registration_reference_audit_failed','details':{'point':'CWS_S02'}},
        {'code':'source_point_coverage_incomplete','missing_names':['CWS_N14']}],
        ['CWS_S01','CWS_S02','CWS_N14'])
    assert names=={'CWS_S01','CWS_S02','CWS_N14'}


def test_visual_chunk_compacts_retry_with_complete_source_rows_when_full_transcript_exceeds_budget():
    source='Complete original MSOP coordinate table. '+('PNT XYZ row preserved. '*2200)+'LAST_XYZ_ROW'
    markers=[{'source_point_names':[name],'view_id':'+Z','uv':uv,'feature':name,'source_pages':[1]}
        for name,uv in [('A_PNT1',[.3,.4]),('A_PNT2',[.7,.6])]]
    raw=json.dumps({'decision':'AUTO_MARK','fai':2,'markers':markers,
        'confidence':'low','evidence':'Source page defines both points.','source_pages':[1]},separators=(',',':'))
    class Model:
        calls=0
        request=None
        def _generate_once(self,messages,**kwargs):
            self.calls+=1;self.request=messages
            assert local_matching._complete_visual_context_budget(messages)<=kwargs['max_context_chars']
            return raw,False,100
    model=Model()
    message={'role':'user','content':'Verbose repeated visual task. '*1200,
        'attachments':[{'filename':'MSOP and drawing','parsed_text':source,'images':['source.png']}]}
    points=[{'name':'A_PNT1','pages':[1],'xyz':[1,2,3],'program':'UP','coordinate_system':'ALI_ABC'},
        {'name':'A_PNT2','pages':[1],'xyz':[4,5,6],'program':'UP','coordinate_system':'ALI_ABC'}]
    parsed,returned,truncated,tokens,_aliases,repairs=local_matching._request_visual_marker_chunk(
        model,message,fai=2,expected_names=['A_PNT1','A_PNT2'],source_points=points,
        global_source_points=points,allowed_source_pages={1})
    assert not truncated and parsed['decision']=='AUTO_MARK' and tokens==100
    assert model.calls==1 and repairs[-1]['contract_issues']==[]
    assert 'parsed_text' not in model.request[0]['attachments'][0]
    assert model.request[0]['attachments'][0]['images']==['source.png']
    assert 'A_PNT2' in model.request[0]['content'] and '[4,5,6]' in model.request[0]['content']


def test_drawing_count_retry_requires_distinct_explicit_candidate_ids():
    choices=[{'id':7},{'id':8}]
    assert _surface_candidate_selection_audit({'markers':[{'candidate_id':'C99'}]},choices)
    assert _surface_candidate_selection_audit({'markers':[{'uv':[.3,.6]}]},choices,
        require_marker_ids=True)
    issues=_surface_candidate_selection_audit({'markers':[
        {'candidate_id':'C7'},{'candidate_id':'C7'}]},choices,require_marker_ids=True)
    assert any(row.get('reason')=='candidate_id_reused_for_distinct_markers' for row in issues)
    assert not _surface_candidate_selection_audit({'markers':[
        {'candidate_id':'C7'},{'candidate_id':'C8'}]},choices,require_marker_ids=True)


def test_collapsed_xyz_blocks_only_the_ambiguous_view_pixel_and_keeps_other_uvs():
    choices=[
        {'id':1,'view_id':'+Z','uv':[.32,.96],
         'source_point_names':['N2_PNT2','N4_PNT2'],
         'blocked_uvs_by_source_name':{'N2_PNT2':[{'view_id':'+Z','uv':[.32,.96]}],
             'N4_PNT2':[{'view_id':'+Z','uv':[.32,.96]}]}},
        {'id':2,'view_id':'-X','uv':[.6,.4],
         'source_point_names':['N2_PNT2','N4_PNT2'],
         'blocked_uvs_by_source_name':{'N2_PNT2':[{'view_id':'+Z','uv':[.32,.96]}],
             'N4_PNT2':[{'view_id':'+Z','uv':[.32,.96]}]}},
    ]
    raw_uv=_surface_candidate_selection_audit({'markers':[
        {'source_point_names':['N2_PNT2'],'view_id':'+Z','uv':[.32,.96]}]},choices)
    assert any(row['reason']=='collapsed_source_point_ray_hit_forbidden' for row in raw_uv)
    candidate_id=_surface_candidate_selection_audit({'markers':[
        {'source_point_names':['N4_PNT2'],'candidate_id':'C1'}]},choices)
    assert any(row['reason']=='collapsed_source_point_ray_hit_forbidden' for row in candidate_id)
    alternate=_surface_candidate_selection_audit({'markers':[
        {'source_point_names':['N4_PNT2'],'candidate_id':'C2'}]},choices)
    assert not alternate
    same_view_other_uv=_surface_candidate_selection_audit({'markers':[
        {'source_point_names':['N2_PNT2'],'view_id':'+Z','uv':[.33,.96]}]},choices)
    assert not same_view_other_uv


def test_cloned_source_scoped_candidate_ids_cannot_hide_duplicate_physical_hits():
    choices=[
        {'id':1,'view_id':'+Z','uv':[.45,.45],'position':[1.0,2.0,3.0],
         'source_point_names':['J01']},
        {'id':2,'view_id':'+Z','uv':[.45,.45],'position':[1.0,2.0,3.0],
         'source_point_names':['A01']},
        {'id':3,'view_id':'+Z','uv':[.55,.45],'position':[2.0,2.0,3.0],
         'source_point_names':['J02']},
        {'id':4,'view_id':'-X','uv':[.55,.45],'position':[2.0,2.0,3.0],
         'source_point_names':['J01']},
        {'id':5,'view_id':'-Y','uv':[.55,.45],'position':[3.0,2.0,3.0],
         'source_point_names':['A01']},
        {'id':6,'view_id':'-X','uv':[.65,.45],'position':[2.5,2.0,3.0],
         'source_point_names':['J01']},
    ]
    parsed={'markers':[{'source_point_names':['J01'],'candidate_id':'C1'},
                       {'source_point_names':['A01'],'candidate_id':'C2'},
                       {'source_point_names':['J02'],'candidate_id':'C3'}]}
    issues=_surface_candidate_selection_audit(parsed,choices,require_marker_ids=True)
    duplicate=next(row for row in issues
        if row.get('reason')=='duplicate_physical_surface_for_distinct_markers')
    assert duplicate['source_point_names']==['A01','J01']
    members={row['source_point_name']:row for row in duplicate['members']}
    assert members['J01']['candidate_id']=='C1'
    assert {row['candidate_id'] for row in members['J01']['alternative_candidates']}=={'C4','C6'}
    assert all(row['cad_xyz_mm'] and row['view_id'] and row['uv']
        for row in members['J01']['alternative_candidates'])
    assert members['A01']['alternative_candidates'][0]['candidate_id']=='C5'
    assert len([row for row in issues
        if row.get('reason')=='duplicate_physical_surface_for_distinct_markers'])==1


def test_source_xy_layout_evidence_keeps_literal_coordinates_and_pair_distances():
    points=[
        {'name':'J01','source_coordinates_2d':{'axes':['X','Y'],'values':[-2.,3.],
            'page':14,'coordinate_system':'ALI_ABC'}},
        {'name':'A01','source_coordinates_2d':{'axes':['X','Y'],'values':[-2.,3.],
            'page':14,'coordinate_system':'ALI_ABC'}},
        {'name':'OTHER','source_coordinates_2d':{'axes':['X','Y'],'values':[900.,900.],
            'page':15,'coordinate_system':'OTHER'}},
        {'name':'NO_XY','source_coordinates_2d':{'axes':['X','Y'],'values':[1.,None],
            'page':14,'coordinate_system':'ALI_ABC'}},
    ]
    evidence=local_matching._source_xy_layout_evidence(points,['J01','A01','NO_XY'])
    assert evidence==[{'source_page':14,'coordinate_system':'ALI_ABC',
        'points':[{'name':'A01','xy_mm':[-2.,3.]},{'name':'J01','xy_mm':[-2.,3.]}],
        'pairwise_xy_distances':[{'points':['A01','J01'],'xy_distance_mm':0.0}]}]


def test_surface_candidate_geometry_groups_ray_hits_by_physical_hit_and_source_option_set():
    candidates=[
        {'id':1,'view_id':'+Z','uv':[.4,.5],'position':[1.,2.,3.],
         'source_point_names':['J01']},
        {'id':7,'view_id':'+Z','uv':[.4,.5],'position':[1.,2.,3.],
         'source_point_names':['A01']},
        {'id':2,'view_id':'-Z','uv':[.6,.5],'position':[1.,2.,3.],
         'source_point_names':['J01']},
        {'id':8,'view_id':'-Z','uv':[.6,.5],'position':[1.,2.,3.],
         'source_point_names':['A01']},
        {'id':3,'view_id':'+Z','uv':[.8,.2],'position':[4.,5.,6.],
         'source_point_names':['J01']},
    ]
    evidence=local_matching._surface_candidate_geometry_evidence(candidates,['J01','A01'])
    assert evidence==[
        {'cad_xyz_mm':[1.,2.,3.],'views':[
            {'view_id':'+Z','uv':[.4,.5],
             'candidate_options':[{'source_point_names':['J01'],'candidate_ids':['C1']},
                 {'source_point_names':['A01'],'candidate_ids':['C7']}]},
            {'view_id':'-Z','uv':[.6,.5],
             'candidate_options':[{'source_point_names':['J01'],'candidate_ids':['C2']},
                 {'source_point_names':['A01'],'candidate_ids':['C8']}]}]},
        {'cad_xyz_mm':[4.,5.,6.],'views':[
            {'view_id':'+Z','uv':[.8,.2],
             'candidate_options':[{'source_point_names':['J01'],'candidate_ids':['C3']}]}]}]


def test_later_visual_chunks_cannot_reuse_reserved_cad_hit_positions():
    candidates=[
        {'id':1,'view_id':'+Z','position':[1.,2.,3.],'source_point_names':['P1']},
        {'id':2,'view_id':'-Z','position':[1.,2.,3.000001],'source_point_names':['P2']},
        {'id':3,'view_id':'+Z','position':[4.,5.,6.],'source_point_names':['P2']},
    ]
    parsed={'markers':[{'source_point_names':['P1'],'candidate_id':'C1'}]}
    reserved=local_matching._selected_surface_candidate_positions(parsed,candidates)
    assert len(reserved)==1 and reserved[0].tolist()==[1.,2.,3.]
    remaining=local_matching._exclude_reserved_surface_positions(candidates,reserved)
    assert [row['id'] for row in remaining]==[3]


def test_duplicate_surface_retry_increases_temperature_and_uses_alternative_hits():
    candidates=[
        {'id':1,'view_id':'+Z','uv':[.45,.45],'position':[1.,2.,3.],'source_point_names':['J01']},
        {'id':2,'view_id':'+Z','uv':[.45,.45],'position':[1.,2.,3.],'source_point_names':['A01']},
        {'id':3,'view_id':'-X','uv':[.55,.45],'position':[2.,2.,3.],'source_point_names':['J01']},
        {'id':4,'view_id':'-Y','uv':[.55,.45],'position':[3.,2.,3.],'source_point_names':['A01']},
    ]
    outputs=[
        {'decision':'AUTO_MARK','fai':9,'confidence':'medium','evidence':'two points',
         'markers':[{'source_point_names':['J01'],'candidate_id':'C1','source_pages':[14]},
                    {'source_point_names':['A01'],'candidate_id':'C2','source_pages':[14]}]},
        {'decision':'AUTO_MARK','fai':9,'confidence':'medium','evidence':'two points',
         'markers':[{'source_point_names':['J01'],'candidate_id':'C3','source_pages':[14]},
                    {'source_point_names':['A01'],'candidate_id':'C4','source_pages':[14]}]},
    ]
    class Model:
        temperatures=[]
        messages=[]
        def _generate_once(self,_messages,**kwargs):
            self.messages.append(_messages[0]['content'])
            self.temperatures.append(kwargs['temperature'])
            return json.dumps(outputs[len(self.temperatures)-1]),False,120
    model=Model()
    message={'role':'user','content':'Choose the two measured positions.',
        'attachments':[{'filename':'MSOP and CAD','parsed_text':'MSOP p.14','images':[]}]}
    parsed,_,truncated,_,_,attempts=local_matching._request_visual_marker_chunk(
        model,message,fai=9,expected_names=['J01','A01'],
        source_points=[{'name':'J01','source_coordinates_2d':{'axes':['X','Y'],'values':[1.,2.],
                'page':14,'coordinate_system':'ALI_ABC'}},
            {'name':'A01','source_coordinates_2d':{'axes':['X','Y'],'values':[1.,2.],
                'page':14,'coordinate_system':'ALI_ABC'}}],allowed_source_pages={14},
        surface_candidates=candidates)
    assert not truncated
    assert [row['candidate_id'] for row in parsed['markers']]==['C3','C4']
    assert model.temperatures==[0.0,0.18]
    assert 'pairwise_xy_distances' in model.messages[0]
    assert 'candidate_options' in model.messages[0]
    assert 'equal XYZ rows across views are the same physical hit' in model.messages[0]
    assert 'Use the per-point alternative' in model.messages[1]
    assert 'common in-plane rigid transform' in model.messages[1]
    assert attempts[0]['contract_issues'][0]['members'][0]['alternative_candidates'][0]['candidate_id']=='C3'


def test_geometry_recovery_requires_unique_candidate_ids_only_for_points_with_candidates():
    choices=[
        {'id':1,'view_id':'-X','uv':[.31,.62],'source_point_names':['P1']},
        {'id':2,'view_id':'+Y','uv':[.72,.48],'source_point_names':['P1']},
    ]
    missing=_surface_candidate_selection_audit({'markers':[
        {'source_point_names':['P1'],'view_id':'+Z','uv':[.2,.2]},
        {'source_point_names':['P2'],'view_id':'+Z','uv':[.8,.8]}]},choices,
        require_marker_ids={'P1'})
    assert sum(row.get('reason')=='explicit_valid_candidate_id_required' for row in missing)==1
    selected=_surface_candidate_selection_audit({'markers':[
        {'source_point_names':['P1'],'candidate_id':'C1'},
        {'source_point_names':['P2'],'view_id':'+Z','uv':[.8,.8]}]},choices,
        require_marker_ids={'P1'})
    assert not selected
    reused=_surface_candidate_selection_audit({'markers':[
        {'source_point_names':['P1'],'candidate_id':'C1'},
        {'source_point_names':['P1'],'candidate_id':'C1'}]},choices,
        require_marker_ids={'P1'})
    assert any(row.get('reason')=='candidate_id_reused_for_distinct_markers' for row in reused)


def test_collapsed_group_diagnostic_blocks_only_exact_uv_for_all_overlapping_names():
    blocked=local_matching._blocked_uvs_from_geometry_diagnostic({
        'reason':'distinct_source_points_collapsed_to_same_view_uv','view_id':'+Z','uv':[.2,.7],
        'source_point_names':['IN_PNT_Q18','IN_PNT_Q28'],
        'missed_marker_points':[{'source_point_names':['IN_PNT_Q45'],'view_id':'+Z','uv':[.2,.7]}]})
    assert blocked=={name:[{'view_id':'+Z','uv':[.2,.7]}]
        for name in ('IN_PNT_Q18','IN_PNT_Q28','IN_PNT_Q45')}
    assert local_matching._blocked_uvs_from_geometry_diagnostic(
        {'reason':'marker_view_ray_missed_CAD','view_id':'+Z','source_point_names':['P1']})=={}
    assert local_matching._blocked_uvs_from_geometry_diagnostic(
        {'reason':'distinct_source_points_collapsed_to_same_view_uv','view_id':'+Z',
         'source_point_names':['P1']})=={}


def test_even_spacing_search_gate_rejects_clusters_without_assuming_a_straight_line():
    import numpy as np
    group={'group_count':1,'source_quotes':['5 WELD SPOTS EVENLY SPACED']}
    clustered=[{'position':[x,0,0]} for x in [0,1,2,2.1,20]]
    failure=local_matching._drawing_spacing_diagnostic(clustered,group)
    assert failure['reason']=='source_even_spacing_constraint_failed'
    assert not failure['is_source_tolerance']
    line=[{'position':[x,0,0]} for x in range(5)]
    assert local_matching._drawing_spacing_diagnostic(line,group) is None
    circle=[{'position':[np.cos(angle),np.sin(angle),0]} for angle in np.arange(5)*2*np.pi/5]
    assert local_matching._drawing_spacing_diagnostic(circle,group) is None
    assert local_matching._drawing_spacing_diagnostic(clustered,{**group,'group_count':2}) is None
    assert local_matching._drawing_spacing_diagnostic(clustered,{'group_count':1,'source_quotes':[]}) is None


@pytest.mark.parametrize('accept_pattern,accept_entity',[(False,False),(True,True),(True,False)])
def test_no_xyz_spacing_repair_requires_fresh_local_model_group_selection(tmp_path,monkeypatch,accept_pattern,accept_entity):
    from oracle_viewer.cad_view_atlas import project_point_to_view_uv
    mesh={'source_sha256':'spacing-model-test','faces':[{'id':'plane','surface':'plane',
        'positions':[0,0,0,30,0,0,30,30,0,0,30,0],'indices':[0,1,2,0,2,3]}]}
    source_positions=[[x,10,0] for x in [0,1,2,2.1,20]]
    group={'kind':'drawing_declared_locations','declared_count':5,'group_count':1,
        'locations_per_group':5,'source_page':2,'source_point_names':[],
        'source_quotes':['5 WELD SPOTS EVENLY SPACED'],'instruction':'Five weld spots'}
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-1.png'])
    monkeypatch.setattr(local_matching,'_resolve_drawing_location_manifest',
        lambda *args:(group,{'status':'confirmed','decision':{'group_count':1}}))
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args,**kwargs:store.annotations.append(args)
    class Model:
        def info(self):return {'image_inference':'strata_native_vision'}
        def _generate_once(self,messages,**kwargs):
            content=messages[0]['attachments'][0].get('parsed_text','')
            if '"source_manifest"' in content and '"markers"' in content:
                return json.dumps({'decision':'CONFIRM_ENTITY_GROUP' if accept_entity else 'REJECT','fai':1,
                    'reviewed_marker_ids':[1,2,3,4,5],'all_entities_match':accept_entity,
                    'all_instances_covered':True,'issues':[],
                    'feature_assignments':[{'marker_id':i,'source_names':[],'instance_path':[],
                        'evidence':'Original drawing rail weld location.'} for i in range(1,6)],
                    'evidence':'All five points belong to the rail shown in the drawing.'}),False,90
            if '"options"' in content:
                return json.dumps({'decision':'AUTO_MARK' if accept_pattern else 'ABSTAIN',
                    'fai':1,'candidate_id':'SP01','source_point_count':5,'confidence':'high',
                    'evidence':'Original drawing supports the complete straight row.'}),False,70
            return json.dumps({'decision':'AUTO_MARK','fai':1,'markers':[
                {'view_id':'+Z','uv':project_point_to_view_uv(mesh,'+Z',point)[0],
                 'feature':'Weld spot','source_pages':[1]} for point in source_positions],
                'confidence':'high','evidence':'Original drawing identifies this weld area.'}),False,100
    result=local_matching._mark_visual_fallback(store,{'id':'p','documents':[]},
        {'number':1,'document_id':'msop','pages':[1]},
        {'methods':[{'page':1,'text':'FAI 1 weld spot heights.'}],'dependencies':[]},Model(),mesh)
    if accept_pattern and accept_entity:
        assert result['result']=='accepted' and result['anchor_count']==5 and result['confidence']=='low'
        assert len(store.annotations)==1
        assert all(row['placement_method']=='local_model_selected_spacing_hypothesis' for row in result['locations'])
        assert [round(row['position'][0],6) for row in result['locations']]==[0,5,10,15,20]
    else:
        assert result['result']=='abstain_or_invalid' and not store.annotations


def test_coordinate_variant_cad_evidence_keeps_each_original_scope_and_all_options():
    mesh={'source_sha256':'variant-test','faces':[{'id':'plane',
        'positions':[0,0,0,10,0,0,0,10,0],'indices':[0,1,2]}]}
    point={'name':'TAB_P01','requires_variant_choice':True,'coordinate_variants':[
        {'page':2,'program':'UP','coordinate_system':'A','xyz':[2,2,0]},
        {'page':3,'program':'UP','coordinate_system':'B','xyz':[2,2,0]}]}
    identity=[[1,0,0],[0,1,0],[0,0,1]]
    frames=[{'program':'UP','coordinate_system':scope,'rotation':identity,'translation_mm':offset}
        for scope,offset in [('A',[0,0,0]),('B',[0,0,10])]]
    rows=local_matching._coordinate_variant_geometry_evidence(mesh,[point],
        {'selected_frame_transforms':frames})
    assert len(rows)==2
    assert rows[0]['distance_to_CAD_mm']==0 and rows[1]['distance_to_CAD_mm']==10
    assert rows[0]['source_xyz']==rows[1]['source_xyz']==[2,2,0]
    assert all(not row['source_to_cad_registered'] for row in rows)
    assert all('selected' not in row for row in rows)
    assert point['requires_variant_choice']  # diagnostics do not choose source rows


def test_direct_numeric_group_cannot_bypass_exact_source_manifest(tmp_path):
    detail={'methods':[{'page':8,'text':'Measure 2 vector points F1_PNT_Z1/Z2.'}]}
    source=[{'name':name,'xyz':[i,0,0]} for i,name in enumerate(['F1_PNT_L2','F1_PNT_L3'])]
    class NoModelCall:
        def _generate_once(self,*args,**kwargs):raise AssertionError('Wrong source list must not reach model')
    result=local_matching._mark_model_selected_xyz_group(
        SimpleNamespace(),{}, {'number':8},detail,NoModelCall(),{},
        {'known_source_points':source,'required_marker_count':2})
    assert result is None


def test_candidate_id_resolves_to_exact_verified_cad_ray_hit():
    parsed={'decision':'AUTO_MARK','markers':[
        {'candidate_id':'C7','feature':'N1','source_point_names':['N1_PNT1']},
        {'candidate':'8','feature':'N2','source_point_names':['N2_PNT1']}],
        'registration_points':[{'candidate_id':7,'source_point_name':'N1_PNT2'}]}
    candidates=[
        {'id':7,'view_id':'+X','uv':[.275,.625],'position':[1,2,3]},
        {'id':8,'view_id':'-Z','uv':[.825,.125],'position':[4,5,6]}]
    result=_resolve_surface_candidate_ids(parsed,candidates)
    assert result['markers'][0]['view_id']=='+X'
    assert result['markers'][0]['uv']==[.275,.625]
    assert result['markers'][1]['view_id']=='-Z'
    assert result['markers'][1]['uv']==[.825,.125]
    assert result['registration_points'][0]['uv']==[.275,.625]
    assert result['markers'][0]['source_point_names']==['N1_PNT1']


def test_candidate_list_requires_id_for_failed_source_points():
    candidates=[{'id':7,'source_point_names':['N1_PNT1','N1_PNT2'],
                 'view_id':'+Z','uv':[.25,.75]}]
    # An explicit model-selected view/UV remains eligible for exact ray-hit
    # and rigid-geometry validation; the atlas ID is an optional search aid.
    assert _surface_candidate_selection_valid({'markers':[
        {'source_point_names':['N1_PNT1'],'view_id':'+Z','uv':[.25,.75]}]},candidates)
    assert _surface_candidate_selection_valid({'markers':[
        {'source_point_names':['N1_PNT1'],'candidate_id':'C7'}],
        'registration_points':[{'source_point_name':'N1_PNT2','candidate_id':7}]},candidates)
    assert not _surface_candidate_selection_valid({'markers':[
        {'source_point_names':['N1_PNT1'],'candidate_id':'C8'}]},candidates)


def test_candidate_audit_returns_only_ids_matching_each_source_point():
    candidates=[{'id':7,'source_point_names':['N1_PNT1']},
                {'id':8,'source_point_names':['N2_PNT1','N2_PNT2']}]
    audit=_surface_candidate_selection_audit({'markers':[
        {'candidate_id':'C8','source_point_names':['N1_PNT1','N2_PNT1']}],
        'registration_points':[{'source_point_name':'N2_PNT2','candidate_id':'C8'}]},candidates)
    assert audit==[{'field':'markers','source_point_name':'N1_PNT1',
                    'selected_candidate_id':'C8','valid_candidate_ids':['C7']}]


def test_noncollinear_msop_xyz_scope_cannot_fall_back_without_rigid_fit():
    scope=('UP','ALI_ABC')
    points={str(i):{'program':scope[0],'coordinate_system':scope[1],
                    'xyz':xyz} for i,xyz in enumerate(([0,0,0],[1,0,0],[0,1,0],[0,0,1]))}
    required=_xyz_rigid_fit_required_scopes(points)
    assert scope in required
    assert _missing_required_xyz_fits({scope},required,{})==[scope]
    assert _missing_required_xyz_fits({scope},required,{scope:{'max_residual_mm':.2}})==[]
    collinear={str(i):{'program':scope[0],'coordinate_system':scope[1],
                       'xyz':[i,0,0]} for i in range(4)}
    assert scope not in _xyz_rigid_fit_required_scopes(collinear)


def test_unambiguous_point_name_and_numeric_page_aliases_normalize_exactly():
    parsed={'markers':[{'source_point_names':['DATUMC_P04'],
                        'source_pages':['61','999']}],
            'coordinate_choices':[{'name':'DATUMC_P04','page':'61'}]}
    normalized,aliases=_normalize_model_source_aliases(parsed,
        [{'name':'DATUMC_PNT_P04'}],{61})
    assert normalized['markers'][0]['source_point_names']==['DATUMC_PNT_P04']
    assert normalized['markers'][0]['source_pages']==[61,'999']
    assert normalized['coordinate_choices'][0]=={'name':'DATUMC_PNT_P04','page':61}
    assert aliases==[{'from':'DATUMC_P04','to':'DATUMC_PNT_P04'},
                     {'from':'DATUMC_P04','to':'DATUMC_PNT_P04'}]


def test_exact_feature_identifier_recovers_omitted_source_point_name_field():
    parsed={'markers':[{'feature':'R2_PNT_P2','view_id':'+Z','uv':[.5,.5]}]}
    normalized,aliases=_normalize_model_source_aliases(parsed,
        [{'name':'R2_PNT_P2'}],{28})
    assert normalized['markers'][0]['source_point_names']==['R2_PNT_P2']
    assert aliases==[{'from_feature':'R2_PNT_P2','to_source_point_name':'R2_PNT_P2'}]


def test_singular_source_point_and_placed_decision_normalize_without_guessing():
    parsed={'decision':'PLACED','markers':[{'source_point_name':'THICK_J01',
        'candidate_id':'C7','view_id':'+Z','uv':[.45,.55]}]}
    normalized,aliases=_normalize_model_source_aliases(parsed,[{'name':'THICK_J01'}],{14})
    assert normalized['decision']=='AUTO_MARK'
    assert normalized['markers'][0]['source_point_names']==['THICK_J01']
    assert aliases==[{'from_decision':'PLACED','to_decision':'AUTO_MARK'},
        {'from_source_point_name':'THICK_J01','to_source_point_name':'THICK_J01'}]


def test_placement_decision_alias_normalizes_but_still_requires_geometry_validation():
    parsed={'decision':'PLACEMENT','markers':[{'source_point_name':'THICK_J01',
        'candidate_id':'C7'}]}
    normalized,aliases=_normalize_model_source_aliases(parsed,[{'name':'THICK_J01'}],{14})
    assert normalized['decision']=='AUTO_MARK'
    assert normalized['markers'][0]['source_point_names']==['THICK_J01']
    assert aliases[0]=={'from_decision':'PLACEMENT','to_decision':'AUTO_MARK'}
    assert normalized['markers'][0]['candidate_id']=='C7'


def test_ambiguous_or_prose_feature_does_not_fill_source_point_names():
    parsed={'markers':[{'feature':'PNT_P2','view_id':'+Z','uv':[.5,.5]},
                       {'feature':'top left point','view_id':'+Z','uv':[.5,.6]}]}
    normalized,aliases=_normalize_model_source_aliases(parsed,
        [{'name':'R2_PNT_P2'},{'name':'S2_PNT_P2'}],{28})
    assert 'source_point_names' not in normalized['markers'][0]
    assert 'source_point_names' not in normalized['markers'][1]
    assert aliases==[]


def test_conflicting_xyz_variants_are_resolved_by_an_explicit_local_model_choice():
    point={'name':'P1','requires_variant_choice':True,'coordinate_variants':[
        {'page':29,'program':'UP','coordinate_system':'ALI_ABC','xyz':[1.0,2.0,3.0]},
        {'page':31,'program':'UP','coordinate_system':'ALI_ABC','xyz':[4.0,5.0,6.0]}]}
    class LocalModel:
        _lock=None
        def _generate_once(self,*args,**kwargs):
            return json.dumps({'decision':'SELECT_VARIANTS','fai':19,'confidence':'medium',
                'evidence':'MSOP method page 29 is the scoped measurement set.',
                'coordinate_choices':[{'name':'P1','page':29,'program':'UP',
                    'coordinate_system':'ALI_ABC','xyz':[1.0,2.0,3.0]}]}),False,64
    choices,audit=_select_coordinate_variants_with_local_model(
        LocalModel(),19,[point],'Measure P1 on page 29.')
    assert choices==[{'name':'P1','page':29,'program':'UP','coordinate_system':'ALI_ABC',
                      'xyz':[1.0,2.0,3.0]}]
    assert audit['status']=='selected'
    assert audit['attempts'][0]['valid'] is True


def test_local_coordinate_variant_selector_abstains_without_inventing_a_row():
    point={'name':'P1','requires_variant_choice':True,'coordinate_variants':[
        {'page':29,'program':'UP','coordinate_system':'ALI_ABC','xyz':[1.0,2.0,3.0]},
        {'page':31,'program':'UP','coordinate_system':'ALI_ABC','xyz':[4.0,5.0,6.0]}]}
    class LocalModel:
        _lock=None
        def _generate_once(self,*args,**kwargs):
            return json.dumps({'decision':'ABSTAIN','fai':19,'coordinate_choices':[]}),False,20
    choices,audit=_select_coordinate_variants_with_local_model(
        LocalModel(),19,[point],'No page resolves the conflict.')
    assert choices is None
    assert audit['status']=='abstained'
    assert len(audit['attempts'])==3


def test_ambiguous_point_name_alias_is_not_normalized():
    parsed={'markers':[{'source_point_names':['DATUMC_P04'],'source_pages':[61]}]}
    normalized,aliases=_normalize_model_source_aliases(parsed,
        [{'name':'DATUMC_PNT_P04'},{'name':'DATUMC_P04'}],{61})
    assert normalized['markers'][0]['source_point_names']==['DATUMC_P04']
    assert aliases==[]


def test_large_visual_point_sets_partition_without_changing_source_identity():
    points=[{'name':f'P{i}','xyz':[i,0,0]} for i in range(1,26)]
    chunks=_partition_visual_source_points(points)
    assert [len(chunk) for chunk in chunks]==[10,10,5]
    assert [point['name'] for chunk in chunks for point in chunk]==[point['name'] for point in points]
    assert _partition_visual_source_points(points[:14])==[]


def test_visual_point_partition_keeps_msop_measurement_groups_together_when_possible():
    points=[{'name':f'P{i}'} for i in range(1,19)]
    groups=[{'source_point_names':[f'P{i}' for i in range(1,5)]},
            {'source_point_names':[f'P{i}' for i in range(5,9)]},
            {'source_point_names':[f'P{i}' for i in range(9,13)]},
            {'source_point_names':[f'P{i}' for i in range(13,19)]}]
    chunks=_partition_visual_source_points(points,required_marker_groups=groups)
    assert [[p['name'] for p in chunk] for chunk in chunks]==[
        [f'P{i}' for i in range(1,9)],
        [f'P{i}' for i in range(9,19)]]


def test_small_complete_measurement_groups_can_enter_candidate_id_repair_path():
    points=[{'name':'R1_PNT_P2'},{'name':'S1_PNT_Z1'},{'name':'R1_PNT_Z3'}]
    groups=[{'source_point_names':['R1_PNT_P2']},
            {'source_point_names':['S1_PNT_Z1','R1_PNT_Z3']}]
    chunks=_partition_visual_source_points(points,required_marker_groups=groups,threshold=0)
    assert [[point['name'] for point in chunk] for chunk in chunks]==[
        ['R1_PNT_P2','S1_PNT_Z1','R1_PNT_Z3']]


def test_explicit_msop_point_pairs_stay_together_when_large_group_is_chunked():
    names=[*(f'THICK_J{i:02}' for i in range(1,7)),
           *(f'THICK_A{i:02}' for i in range(1,7))]
    methods=[{'text':'Output Z distance (6x) from point to point '
        '(J01-A01,J02-A02,J03-A03,J04-A04,J05-A05,J06-A06).'}]
    pairs=_explicit_source_point_relations(methods,names)
    assert pairs==[[f'THICK_J{i:02}',f'THICK_A{i:02}'] for i in range(1,7)]
    chunks=_partition_visual_source_points([{'name':name} for name in names],
        required_marker_groups=[{'source_point_names':names}],related_point_groups=pairs,
        chunk_size=10,threshold=0)
    assert [[point['name'] for point in chunk] for chunk in chunks]==[
        [name for i in range(1,6) for name in (f'THICK_J{i:02}',f'THICK_A{i:02}')],
        ['THICK_J06','THICK_A06']]


def test_explicit_xy_coincident_z_pairs_pack_three_plan_locations_per_decision():
    names=[*(f'THICK_J{i:02}' for i in range(1,7)),
           *(f'THICK_A{i:02}' for i in range(1,7))]
    methods=[{'text':'Output Z distance (6x) from point to point '
        '(J01-A01,J02-A02,J03-A03,J04-A04,J05-A05,J06-A06).'}]
    pairs=_explicit_source_point_relations(methods,names)
    chunks=_partition_visual_source_points([{'name':name} for name in names],
        required_marker_groups=[{'source_point_names':names}],related_point_groups=pairs,
        chunk_size=6,threshold=0)
    assert [[point['name'] for point in chunk] for chunk in chunks]==[
        [name for i in range(start,start+3) for name in (f'THICK_J{i:02}',f'THICK_A{i:02}')]
        for start in (1,4)]


def test_numbered_point_pairs_are_expanded_only_across_source_ellipsis():
    names=[*(f'EDGE_J{i:02}' for i in range(1,7)),
           *(f'EDGE_A{i:02}' for i in range(1,7))]
    methods=[{'text':'Output Z distance (6x) from point to point '
        '(J01-A01,J02-A02……J06-A06).'}]
    assert _explicit_source_point_relations(methods,names)==[
        [f'EDGE_J{i:02}',f'EDGE_A{i:02}'] for i in range(1,7)]


def test_numbered_point_pairs_without_ellipsis_do_not_invent_intermediate_relations():
    names=[*(f'EDGE_J{i:02}' for i in range(1,7)),
           *(f'EDGE_A{i:02}' for i in range(1,7))]
    methods=[{'text':'Output Z distance (2x) from point to point (J01-A01,J06-A06).'}]
    assert _explicit_source_point_relations(methods,names)==[
        ['EDGE_J01','EDGE_A01'],['EDGE_J06','EDGE_A06']]


def test_two_point_visual_chunk_uses_bounded_compact_generation():
    class LocalModel:
        max_new_tokens=None
        prompt=''
        def _generate_once(self,messages,**kwargs):
            self.max_new_tokens=kwargs['max_new_tokens']
            self.prompt=messages[0]['content']
            markers=[{'source_point_names':[name],'feature':name,'view_id':view,
                'uv':[.4,.6],'source_pages':[14],'candidate_id':f'C{index}'}
                for index,(name,view) in enumerate((('THICK_J01','+Z'),('THICK_A01','-Z')),1)]
            return json.dumps({'decision':'AUTO_MARK','fai':9,'markers':markers,
                'registration_points':[],'coordinate_choices':[],
                'confidence':'high','evidence':'MSOP p14'}),False,100
    model=LocalModel()
    candidates=[{'id':index,'view_id':view,'uv':[.4,.6],
        'source_point_names':[name],'position':[0.,0.,float(index)]}
        for index,(name,view) in enumerate((('THICK_J01','+Z'),('THICK_A01','-Z'),
                                            ('THICK_J06','+Z')),1)]
    global_points=[{'name':f'THICK_J{i:02}','source_coordinates_2d':{
        'axes':['X','Y'],'values':[float(i),float(i*i)],'page':14,'coordinate_system':'ALI_ABC'}}
        for i in range(1,7)]
    result,_,truncated,_,_,_=_request_visual_marker_chunk(model,
        {'role':'user','content':'Locate this MSOP pair.','attachments':[]},fai=9,
        expected_names=['THICK_J01','THICK_A01'],
        source_points=[{'name':'THICK_J01'},{'name':'THICK_A01'}],
        allowed_source_pages={14},surface_candidates=candidates,
        global_source_points=global_points)
    assert result and not truncated
    assert model.max_new_tokens==1024
    assert 'THICK_J06' in model.prompt
    assert 'Exact CAD candidate geometry' in model.prompt


def test_literal_msop_xy_rows_are_attached_without_fabricating_z_and_identify_z_pairs():
    method={'page':14,'text':'''Output system ALI_ABC
Features X Y
THICK_J01 12.5 34.0 THICK_A01 12.5 34.0
THICK_J02 18.0 34.0 THICK_A02 18.1 34.0
Output Z distance (1x) from point to point (J01-A01).'''}
    points=[{'name':name,'xyz':None} for name in
        ('THICK_J01','THICK_A01','THICK_J02','THICK_A02')]
    attached=_attach_explicit_source_xy(points,[method])
    assert attached[0]['source_coordinates_2d']=={
        'axes':['X','Y'],'values':[12.5,34.0],'coordinate_system':'ALI_ABC',
        'page':14,'coordinate_frame_source':'explicit_output_system_table',
        'original_row':'THICK_J01 12.5 34.0 THICK_A01 12.5 34.0'}
    assert all('z' not in row.get('source_coordinates_2d',{}) for row in attached)
    pairs=_explicit_coincident_xy_z_distance_pairs([method],attached)
    assert pairs==[{'source_point_names':['THICK_J01','THICK_A01'],
        'source_xy':[12.5,34.0],'coordinate_system':'ALI_ABC',
        'source_page':14,'axis':'Z'}]


def test_xy_pair_extraction_does_not_assume_z_relation_or_pair_noncoincident_points():
    method={'page':14,'text':'Output Y distance (1x) (J01-A01).'}
    points=[{'name':'THICK_J01','source_coordinates_2d':{'axes':['X','Y'],'values':[1,2]}},
            {'name':'THICK_A01','source_coordinates_2d':{'axes':['X','Y'],'values':[1,2.01]}}]
    assert _explicit_coincident_xy_z_distance_pairs([method],points)==[]


def test_small_visual_chunk_requires_candidate_id_when_an_exact_hit_option_exists():
    candidates=[{'id':7,'view_id':'+Z','uv':[.42,.5],
        'source_point_names':['CIR_N1_VISION'],'candidate_role':'global_surface_search'}]
    uv_only={'decision':'AUTO_MARK','fai':23,'markers':[{
        'source_point_names':['CIR_N1_VISION'],'view_id':'+Z','uv':[.42,.5]}]}
    rejected=local_matching._visual_chunk_contract_issues(uv_only,['CIR_N1_VISION'],
        fai=23,surface_candidates=candidates,require_candidate_ids={'CIR_N1_VISION'})
    assert any(issue.get('reason')=='explicit_valid_candidate_id_required' for issue in rejected)
    chosen={'decision':'AUTO_MARK','fai':23,'markers':[{
        'source_point_names':['CIR_N1_VISION'],'candidate_id':'C7'}]}
    assert local_matching._visual_chunk_contract_issues(chosen,['CIR_N1_VISION'],
        fai=23,surface_candidates=candidates,require_candidate_ids={'CIR_N1_VISION'})==[]


def test_invalid_candidate_ids_with_collapsed_uv_return_audit_instead_of_crashing():
    candidates=[
        {'id':1,'view_id':'+Z','uv':[.2,.3],'position':[0.,0.,0.],
         'source_point_names':['P1']},
        {'id':2,'view_id':'+Z','uv':[.2,.3],'position':[1.,0.,0.],
         'source_point_names':['P2']},
        {'id':3,'view_id':'-X','uv':[.7,.4],'position':[0.,1.,0.],
         'source_point_names':['P1']},
        {'id':4,'view_id':'-X','uv':[.8,.4],'position':[1.,1.,0.],
         'source_point_names':['P2']},
    ]
    parsed={'markers':[
        {'source_point_names':['P1'],'candidate_id':'C999','view_id':'+Z','uv':[.5,.5]},
        {'source_point_names':['P2'],'candidate_id':'C998','view_id':'+Z','uv':[.5,.5]},
    ]}
    issues=local_matching._surface_candidate_selection_audit(parsed,candidates)
    assert any(row.get('reason')=='explicit_valid_candidate_id_required' for row in issues)
    assert any(row.get('reason')=='duplicate_physical_surface_for_distinct_markers' for row in issues)


def test_chunked_visual_outputs_merge_only_complete_disjoint_model_decisions():
    chunks=[['P1','P2'],['P3']]
    results=[
        {'decision':'AUTO_MARK','fai':53,'confidence':'medium','evidence':'chunk one',
         'markers':[{'source_point_names':['P1'],'view_id':'+Z','uv':[.1,.1],
                     'feature':'N1','source_pages':[46]},
                    {'source_point_names':['P2'],'view_id':'+Z','uv':[.2,.2],
                     'feature':'N1','source_pages':[46]}],
         'coordinate_choices':[{'name':'P3','page':46,'program':'UP','coordinate_system':'ALI_ABC','xyz':[3,0,0]}],
         'registration_points':[{'source_point_name':'P3','view_id':'+Z','uv':[.3,.3]}]},
        {'decision':'AUTO_MARK','fai':53,'confidence':'high','evidence':'chunk two',
         'markers':[{'source_point_names':['P3'],'view_id':'+X','uv':[.3,.3],
                     'feature':'N1','source_pages':[46]}],
         'coordinate_choices':[{'name':'P3','page':46,'program':'UP','coordinate_system':'ALI_ABC','xyz':[3,0,0]}],
         'registration_points':[{'source_point_name':'P3','view_id':'+Z','uv':[.3,.3]}]},
    ]
    merged=_merge_visual_marker_chunks(results,chunks,fai=53)
    assert merged['decision']=='AUTO_MARK'
    assert {name for marker in merged['markers'] for name in marker['source_point_names']}=={'P1','P2','P3'}
    assert merged['confidence']=='medium'
    assert merged['_chunk_failures']==[]
    assert merged['coordinate_choices'][0]['name']=='P3'
    assert merged['registration_points'][0]['source_point_name']=='P3'


def test_complete_xyz_marker_groups_can_be_partitioned_into_short_model_responses():
    points=[{'name':f'P{index:02d}','xyz':[float(index),0.,0.]} for index in range(1,36)]
    chunks=local_matching._partition_visual_source_points(points,chunk_size=6,threshold=0)
    assert len(chunks)==6
    assert max(map(len,chunks))==6
    assert [point['name'] for chunk in chunks for point in chunk]==[
        point['name'] for point in points]


def test_chunked_visual_merge_abstains_if_any_point_is_repeated_or_missing():
    results=[{'decision':'AUTO_MARK','fai':53,'markers':[
        {'source_point_names':['P1'],'view_id':'+Z','uv':[.1,.1]},
        {'source_point_names':['P1'],'view_id':'+Z','uv':[.2,.2]}],
        'coordinate_choices':[],'registration_points':[],'confidence':'low'}]
    merged=_merge_visual_marker_chunks(results,[['P1','P2']],fai=53)
    assert merged['decision']=='ABSTAIN'
    assert any(row['reason']=='chunk_point_coverage_incomplete' for row in merged['_chunk_failures'])


def test_visual_validation_reports_required_count_and_missing_msop_points():
    parsed={'decision':'AUTO_MARK','fai':36,'markers':[
        {'source_point_names':['P1']},{'source_point_names':['P2']} ]}
    issues=_visual_marker_validation_issues(parsed,fai=36,required_marker_count=3,
        known_source_points=[{'name':'P1'},{'name':'P2'},{'name':'P3'}],
        rigid_group_possible=True,candidate_selection_valid=True,
        coordinate_choices_valid=True,truncated=False)
    assert issues==[
        {'code':'required_marker_count_mismatch','expected':3,'actual':2},
        {'code':'source_point_coverage_incomplete','missing_names':['P3'],
         'missing_count':1,'expected_count':3}]


def test_complete_model_selected_marker_group_can_be_reused_for_frame_fit():
    points=[{'name':'P1'},{'name':'P2'},{'name':'P3'}]
    markers=[{'source_point_names':['P1'],'view_id':'+Z','uv':[.2,.2]},
             {'source_point_names':['P2'],'view_id':'+Z','uv':[.5,.4]},
             {'source_point_names':['P3'],'view_id':'+Y','uv':[.8,.6]},
             {'source_point_names':[],'view_id':'-X','uv':[.4,.5]}]
    refs=_complete_marker_registration_references(points,markers)
    assert [row['source_point_name'] for row in refs]==['P1','P2','P3']
    assert _complete_marker_registration_references(points,markers[:2]) is None


def test_visual_context_compacts_repeated_registration_evidence_without_dropping_transform():
    source={'selected_frame_transforms':[dict(program='DOWN',coordinate_system='ALI_ABC',
        rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[1,2,3],
        hypothesis_id='H02',model_confidence='high',p95_residual_mm=.1,
        model_evidence=['x'*500 for _ in range(20)])],
        'no_geometric_anchor_authorized':True}
    compact=_compact_registration_diagnostic(source)
    assert compact['selected_frame_transforms'][0]['rotation']==source['selected_frame_transforms'][0]['rotation']
    assert compact['selected_frame_transforms'][0]['translation_mm']==[1,2,3]
    assert len(compact['selected_frame_transforms'][0]['model_evidence'])==3
    assert len(json.dumps(compact))<len(json.dumps(source))


class _Store:
    def __init__(self, root: Path, document: dict):
        self.root = root
        self.document = document

    def active_msop(self, _project):
        return self.document

    def _dir(self, project_id):
        return self.root / project_id


class _Model:
    def __init__(self, backend, capability):
        self.backend = backend
        self.capability = capability

    def info(self):
        return {'inference_backend': self.backend, 'image_inference': self.capability}


def test_large_drawing_exact_fai_detail_retains_vector_source_and_checks_hash(tmp_path):
    source=tmp_path/'large-drawing.pdf'
    pdf=pymupdf.open();page=pdf.new_page(width=2400,height=1600)
    page.insert_text((400,500),'FAI\n55',fontsize=12)
    page.insert_text((1800,1000),'FAI\n5',fontsize=12)
    page.draw_line((400,510),(650,640))
    pdf.save(source);pdf.close()
    document={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    details=local_matching._drawing_fai_detail_images(document,1,5,tmp_path)
    # FAI 55 is a prefix match in PDF search, but is not an FAI 5 callout.
    assert len(details)==1
    pix=pymupdf.Pixmap(details[0])
    assert 1000<pix.width<2400 and 1000<pix.height<1600
    with pymupdf.open(source) as pdf:
        page=pdf[0]
        # Actual crop is around the FAI 5 at the lower right, not FAI 55.
        hits=[rect for rect in page.search_for('FAI 5') if rect.x0>1500]
        rect=pymupdf.Rect(hits[0])
        for hit in hits[1:]:rect|=hit
        center=(rect.tl+rect.br)/2
        clip=pymupdf.Rect(center.x-432,center.y-352,center.x+432,center.y+352)&page.rect
        expected=page.get_pixmap(matrix=pymupdf.Matrix(2,2),clip=clip,alpha=False)
        # Preserve original pixels: no invented leaders or CAD markers.
        assert pix.samples==expected.samples
        assert 'drawing-detail-v1-' in Path(details[0]).name
    assert local_matching._drawing_fai_detail_images(document,1,6,tmp_path)==[]
    assert local_matching._drawing_fai_detail_images(
        {**document,'sha256':'0'*64},1,5,tmp_path)==[]


def test_method_pages_are_attached_only_for_configured_local_vision(tmp_path):
    source = tmp_path / 'msop.pdf'
    pdf = pymupdf.open()
    pdf.new_page(width=320, height=240).insert_text((30, 50), 'FAI 23 Plane View')
    pdf.new_page(width=320, height=240).insert_text((30, 50), 'FAI 24 Detail View')
    pdf.save(source)
    pdf.close()
    document = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    drawing_path = tmp_path / 'drawing.pdf'
    drawing_pdf = pymupdf.open()
    drawing_pdf.new_page(width=320, height=240).insert_text((30, 50), 'FAI 22')
    drawing_pdf.new_page(width=320, height=240).insert_text((30, 50), 'FAI 23 drawing callout')
    drawing_pdf.save(drawing_path)
    drawing_pdf.close()
    drawing = {'role': 'drawing', 'path': str(drawing_path),
               'sha256': hashlib.sha256(drawing_path.read_bytes()).hexdigest()}
    store = _Store(tmp_path / 'viewer', document)
    project = {'id': 'project123', 'documents': [document, drawing]}
    detail = {'methods': [{'page': 1}, {'page': 2}, {'page': 1}]}

    images = _method_page_images(
        store, project, detail, _Model('strata-native', 'local_auxiliary_vision'), 23)
    assert len(images) == 3
    assert all(Path(path).is_file() for path in images)
    assert _method_page_images(
        store, project, detail, _Model('strata-native', 'unsupported_by_native_text_model'), 23) == []


def test_coordinate_choice_images_include_every_competing_definition_page(tmp_path):
    source=tmp_path/'competing-pages.pdf';pdf=pymupdf.open()
    for index in range(4):pdf.new_page().insert_text((30,50),f'Source page {index+1}')
    pdf.save(source);pdf.close()
    document={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    store=_Store(tmp_path/'viewer',document)
    project={'id':'project123','documents':[document]}
    points=[{'name':'P01','requires_variant_choice':True,'coordinate_variants':[
        {'page':2,'program':'UP','coordinate_system':'A','xyz':[0,0,0]},
        {'page':4,'program':'UP','coordinate_system':'B','xyz':[0,0,1]}]}]
    images=local_matching._coordinate_variant_source_images(store,project,{'methods':[{'page':1}]},
        _Model('strata-native','strata_native_vision'),8,points)
    assert len(images)==3 and all(Path(path).is_file() for path in images)
    assert [int(Path(path).stem.rsplit('-',1)[1]) for path in images]==[1,2,4]
    class Model:
        def _generate_once(self,messages,**kwargs):
            assert messages[0]['attachments'][0]['images']==images
            assert kwargs['max_images']==3
            return json.dumps({'decision':'SELECT_VARIANTS','fai':8,'coordinate_choices':[
                {'name':'P01',**points[0]['coordinate_variants'][0]}]}),False,60
    choices,audit=_select_coordinate_variants_with_local_model(Model(),8,points,'Original methods.',
        source_images=[*images,str(tmp_path/'msop-decoy-page-99.png')])
    assert audit['status']=='selected' and choices[0]['page']==2


def test_drawing_location_manifest_requires_exact_source_quotes_and_local_count(tmp_path):
    detail=tmp_path/'drawing-detail-v1-source.png';detail.write_bytes(b'not-decoded-by-test-model')
    detail.with_suffix('.json').write_text(json.dumps({'fai':8,'page':2,
        'document_sha256':'source-hash','original_pdf_text':'4 SPOTS PER SIDE\nEVENLY SPACED\nFAI\n8'}))
    class Model:
        quote='4 SPOTS PER SIDE'
        count=8
        extra_quotes=[]
        def _generate_once(self,messages,**kwargs):
            assert kwargs['max_images']==1
            assert '原图确认' in messages[0]['content']
            return json.dumps({'decision':'CONFIRM_LOCATIONS','fai':8,'marker_count':self.count,
                'locations_per_group':4,'group_count':2,
                'feature':'Weld spots','source_quotes':[self.quote,*self.extra_quotes],
                'multiplicity_evidence':'Two sides, four original spots each','source_pages':[2]}),False,80
    model=Model()
    group,audit=local_matching._resolve_drawing_location_manifest(
        model,8,[str(detail)],{'sha256':'source-hash'})
    assert group['declared_count']==8 and audit['status']=='confirmed'
    # A correctly transcribed short balloon number is harmless. The full
    # quantity quote is independently mandatory for multi-location groups.
    model.extra_quotes=['FAI','8']
    group,audit=local_matching._resolve_drawing_location_manifest(
        model,8,[str(detail)],{'sha256':'source-hash'})
    assert group['declared_count']==8 and audit['source_quotes_valid']
    model.extra_quotes=[]
    # Invented or corrected text cannot justify a full-group count.
    model.quote='7 SPOTS PER CORNER'
    group,audit=local_matching._resolve_drawing_location_manifest(
        model,8,[str(detail)],{'sha256':'source-hash'})
    assert group is None and not audit['source_quotes_valid']
    model.quote='EVENLY SPACED'
    group,audit=local_matching._resolve_drawing_location_manifest(
        model,8,[str(detail)],{'sha256':'source-hash'})
    assert group is None and audit['source_quotes_valid'] and not audit['quantity_quote_valid']
    model.quote='4 SPOTS PER SIDE';model.count=1001
    group,audit=local_matching._resolve_drawing_location_manifest(
        model,8,[str(detail)],{'sha256':'source-hash'})
    assert group is None
    group,audit=local_matching._resolve_drawing_location_manifest(
        model,9,[str(detail)],{'sha256':'source-hash'})
    assert group is None and audit['status']=='not_available'


def test_exact_xyz_definition_dependency_page_is_attached_with_method_page(tmp_path):
    source = tmp_path / 'msop-with-coordinate-view.pdf'
    pdf = pymupdf.open()
    pdf.new_page(width=320, height=240).insert_text((30, 50), 'FAI 36 Measure 2 vector circles')
    pdf.new_page(width=320, height=240).insert_text((30, 50), 'DATUMB_PNT_P09 XYZ Plane View')
    pdf.new_page(width=320, height=240).insert_text((30, 50), 'Unrelated DATUMC_PNT_P01 XYZ')
    pdf.save(source)
    pdf.close()
    method_text = ('Measured System ALI_ABC Program UP Measure 1 vector points '
                   'DATUMB_PNT_P09.')
    document = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'method_pages': [
                    {'page': 1, 'text': method_text, 'program': 'UP', 'measured_system': 'ALI_ABC'},
                    {'page': 2, 'text': 'Location Definitions\nX, Y, Z (mm) 1.0 2.0 3.0 DATUMB_PNT_P09',
                     'program': 'UP', 'measured_system': 'ALI_ABC'},
                    {'page': 3, 'text': 'Location Definitions\nX, Y, Z (mm) 4.0 5.0 6.0 DATUMC_PNT_P01',
                     'program': 'UP', 'measured_system': 'ALI_ABC'},
                ]}
    store = _Store(tmp_path / 'viewer', document)
    project = {'id': 'project123', 'documents': [document]}
    detail = {
        'methods': [{'page': 1, 'text': method_text, 'program': 'UP', 'measured_system': 'ALI_ABC'}],
        'dependencies': [{'page': 2}, {'page': 3}],
    }

    images = _method_page_images(
        store, project, detail, _Model('strata-native', 'local_auxiliary_vision'), 36)

    assert len(images) == 2
    assert 'page-1.png' in images[0]
    assert 'page-2.png' in images[1]


def test_incomplete_explicit_point_group_still_attaches_known_xyz_source_page(tmp_path):
    source=tmp_path/'partial-point-group.pdf'
    pdf=pymupdf.open()
    pdf.new_page(width=320,height=240).insert_text((20,40),
        'FAI 48 Measure 2 surface points STRAP_PNT1, STANDOFF_U18.')
    pdf.new_page(width=320,height=240).insert_text((20,40),
        'Location Definitions X, Y, Z (mm) 0.0 0.821 0.97 STANDOFF_U18')
    pdf.save(source);pdf.close()
    document={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'method_pages':[
                  {'page':1,'text':'FAI 48 Measure 2 surface points STRAP_PNT1, STANDOFF_U18.',
                   'program':'DOWN','measured_system':'ALI_ABC'},
                  {'page':2,'text':'Location Definitions X, Y, Z (mm) 0.0 0.821 0.97 STANDOFF_U18',
                   'program':'DOWN','measured_system':'ALI_ABC'}]}
    store=_Store(tmp_path/'viewer',document)
    project={'id':'project123','documents':[document]}
    detail={'methods':[document['method_pages'][0]],'dependencies':[document['method_pages'][1]]}

    images=_method_page_images(
        store,project,detail,_Model('strata-native','local_auxiliary_vision'),48)

    assert len(images)==2
    assert 'page-1.png' in images[0]
    assert 'page-2.png' in images[1]


def test_model_selected_xyz_group_marks_all_points_with_compact_group_decision(tmp_path,monkeypatch):
    monkeypatch.setattr(local_matching,'_review_visual_entity_group',lambda *a,**k:{'status':'confirmed'})
    project={'id':'project123','drawing_no':'generic-frame','revision':'A'}
    item={'number':7,'document_id':'msop-doc','pages':[12],'headings':['FAI 7 datum plane']}
    points=[{'name':f'P{i+1}','xyz':xyz,'pages':[12],'program':'UP',
             'coordinate_system':'ALI_A','requires_variant_choice':False,'coordinate_variants':[]}
            for i,xyz in enumerate(([0,0,0],[4,0,0],[4,4,0],[0,4,0]))]
    candidate={'known_source_points':points,'required_point_count':4,
        'evidence':{'known_source_points':points,'registration_diagnostic':{
            'selected_frame_transforms':[{'program':'UP','coordinate_system':'ALI_A','hypothesis_id':'H00',
                'rotation':[[1,0,0],[0,1,0],[0,0,1]],'translation_mm':[0,0,2],
                'p95_residual_mm':0.02,'max_residual_mm':0.04,'model_confidence':'medium'}]}}}
    detail={'methods':[{'page':12,'text':'FAI 7 measures all four datum plane points.'}], 'dependencies':[]}
    class Store:
        annotations=[]
        def _dir(self,_project_id):
            (tmp_path/'automatic-learning').mkdir(exist_ok=True)
            return tmp_path
        def annotate(self,*args,**kwargs):self.annotations.append(args)
    class Model:
        def _generate_once(self,messages,**kwargs):
            assert 'Complete source XYZ rows' in messages[0]['attachments'][0]['parsed_text']
            return json.dumps({'decision':'AUTO_MARK','fai':7,'source_point_count':4,'candidate_id':'BASE',
                'evidence':'All four XYZ points align with the same CAD datum face.',
                'confidence':'low','source_pages':[12]}),False,54
    mesh={'source_sha256':'cad-hash','faces':[{'id':'datum-face','occt_face_valid':True,
        'positions':[-20,-20,2,20,-20,2,20,20,2,-20,20,2],
        'indices':[0,1,2,0,2,3]}]}
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-12.png'])
    result=local_matching._mark_model_selected_xyz_group(Store(),project,item,detail,Model(),mesh,candidate)

    assert result['result']=='accepted'
    assert result['xyz_group_review'] is True
    assert result['anchor_count']==4 and result['confidence']=='low'


def test_model_selected_xyz_group_allows_one_point_vision_fai(tmp_path,monkeypatch):
    monkeypatch.setattr(local_matching,'_review_visual_entity_group',lambda *a,**k:{'status':'confirmed'})
    project={'id':'project123','drawing_no':'generic-frame','revision':'A'}
    item={'number':23,'document_id':'msop-doc','pages':[33],'headings':['FAI 23 vision center']}
    point={'name':'CIR_N1_CENTER','xyz':[0,0,0],'pages':[33],'program':'UP',
           'coordinate_system':'ALI_A','requires_variant_choice':False,'coordinate_variants':[]}
    candidate={'known_source_points':[point],'required_point_count':1,
        'evidence':{'known_source_points':[point],'registration_diagnostic':{
            'selected_frame_transforms':[{'program':'UP','coordinate_system':'ALI_A','hypothesis_id':'H00',
                'rotation':[[1,0,0],[0,1,0],[0,0,1]],'translation_mm':[0,0,2],
                'p95_residual_mm':0.02,'max_residual_mm':0.04,'model_confidence':'medium'}]}}}
    detail={'methods':[{'page':33,'text':'FAI 23 vision circle center Y coordinate.'}], 'dependencies':[]}
    class Store:
        annotations=[]
        def _dir(self,_project_id):
            (tmp_path/'automatic-learning').mkdir(exist_ok=True)
            return tmp_path
        def annotate(self,*args,**kwargs):self.annotations.append(args)
    class Model:
        def _generate_once(self,messages,**kwargs):
            return json.dumps({'decision':'AUTO_MARK','fai':23,'source_point_count':1,'candidate_id':'BASE',
                'evidence':'The one source point is the MSOP vision-circle center on the CAD bore.',
                'confidence':'low','source_pages':[33]}),False,54
    mesh={'source_sha256':'cad-hash','faces':[{'id':'datum-face','occt_face_valid':True,
        'positions':[-20,-20,2,20,-20,2,20,20,2,-20,20,2],
        'indices':[0,1,2,0,2,3]}]}
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-33.png'])

    result=local_matching._mark_model_selected_xyz_group(
        Store(),project,item,detail,Model(),mesh,candidate)

    assert result['result']=='accepted'
    assert result['anchor_count']==1
    assert result['source_pages']==[33]


def test_model_selects_geometry_valid_fai_local_origin_candidate(tmp_path,monkeypatch):
    monkeypatch.setattr(local_matching,'_review_visual_entity_group',lambda *a,**k:{'status':'confirmed'})
    """Large FAI-local origin alternatives still require an explicit local-model choice."""
    from PIL import Image
    import oracle_viewer.cad_view_atlas as atlas
    project={'id':'project123','drawing_no':'generic-frame','revision':'A'}
    item={'number':38,'document_id':'msop-doc','pages':[52],'headings':['FAI 38']}
    points=[{'name':f'P{i}','xyz':[i,0,0],'pages':[52],'program':'UP',
             'coordinate_system':'ALI_A','requires_variant_choice':False,'coordinate_variants':[]}
            for i in range(3)]
    frame={'program':'UP','coordinate_system':'ALI_A','hypothesis_id':'H00',
           'rotation':[[1,0,0],[0,1,0],[0,0,1]],'translation_mm':[0,64.4,2],
           'p95_residual_mm':0.1,'max_residual_mm':0.2,'model_confidence':'high'}
    candidate={'known_source_points':points,'required_point_count':3,
        'evidence':{'known_source_points':points,'registration_diagnostic':
                    {'selected_frame_transforms':[frame]}}}
    detail={'methods':[{'page':52,'text':'Measure FAI 38 datum points.'}],'dependencies':[]}
    class Store:
        annotations=[]
        def _dir(self,_project_id):
            (tmp_path/'automatic-learning').mkdir(exist_ok=True)
            return tmp_path
        def annotate(self,*args,**kwargs):self.annotations.append(args)
    class Model:
        def _generate_once(self,messages,**kwargs):
            context=messages[0]['attachments'][0]['parsed_text']
            assert 'O01' in context
            return json.dumps({'decision':'AUTO_MARK','fai':38,'source_point_count':3,
                'candidate_id':'O01','evidence':'The O01 point group matches the MSOP feature.',
                'confidence':'medium','source_pages':[52]}),False,80
    def project_group(_mesh,_points,frames,_detail,_choices):
        if abs(frames[0]['translation_mm'][1])>1:
            return None,{'reason':'outside_gate'}
        return ([{'face_id':'face','position':[0,0,0]} for _ in points],[],[]),None
    def render(_mesh,_spec,path,**_kwargs):
        Image.new('RGB',(64,64),'white').save(path)
    monkeypatch.setattr(local_matching,'_project_model_selected_xyz_group',project_group)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-52.png'])
    monkeypatch.setattr(atlas,'render_registration_group_atlas',render)
    result=local_matching._mark_model_selected_xyz_group(
        Store(),project,item,detail,Model(),{'source_sha256':'cad-hash','faces':[]},candidate)
    assert result['result']=='accepted'
    assert result['selected_local_frame_candidate']=='O01'


@pytest.mark.parametrize('echo',['omitted','exact','different'])
def test_model_xyz_group_resolves_coordinate_variants_before_geometry(tmp_path,monkeypatch,echo):
    monkeypatch.setattr(local_matching,'_review_visual_entity_group',lambda *a,**k:{'status':'confirmed'})
    from PIL import Image
    import oracle_viewer.cad_view_atlas as atlas
    project={'id':'project123','drawing_no':'generic-frame','revision':'A'}
    item={'number':9,'document_id':'msop-doc','pages':[15],'headings':['FAI 9']}
    point={'name':'P1','xyz':[0,0,0],'pages':[15],'program':None,
        'coordinate_system':None,'requires_variant_choice':True,
        'coordinate_variants':[{'name':'P1','page':15,'program':'UP','coordinate_system':'ALI_A','xyz':[1,2,3]},
            {'name':'P1','page':15,'program':'UP','coordinate_system':'ALI_B','xyz':[4,5,6]}]}
    frame={'program':'UP','coordinate_system':'ALI_A','hypothesis_id':'H00',
        'rotation':[[1,0,0],[0,1,0],[0,0,1]],'translation_mm':[0,0,2],
        'p95_residual_mm':0.1,'max_residual_mm':0.2,'model_confidence':'high'}
    candidate={'known_source_points':[point],'required_point_count':1,
        'evidence':{'known_source_points':[point],'registration_diagnostic':
                    {'selected_frame_transforms':[frame]}}}
    detail={'methods':[{'page':15,'text':'Measure one point P1.'}],'dependencies':[]}
    class Store:
        annotations=[]
        def active_msop(self,_project):return {'method_pages':detail['methods']}
        def _dir(self,_project_id):
            (tmp_path/'automatic-learning').mkdir(exist_ok=True)
            return tmp_path
        def annotate(self,*args,**kwargs):self.annotations.append(args)
    class Model:
        def _generate_once(self,messages,**kwargs):
            context=messages[0]['attachments'][0]['parsed_text']
            assert 'Local-model coordinate variant resolution' in context
            answer={'decision':'AUTO_MARK','fai':9,'source_point_count':1,
                'candidate_id':'BASE','evidence':'The selected source row matches the CAD point.',
                'confidence':'medium','source_pages':[15]}
            if echo!='omitted':answer['coordinate_choices']=[point['coordinate_variants'][0 if echo=='exact' else 1]]
            return json.dumps(answer),False,50
    selected=[{'name':'P1','page':15,'program':'UP','coordinate_system':'ALI_A','xyz':[1,2,3]}]
    monkeypatch.setattr(local_matching,'_select_coordinate_variants_with_local_model',
        lambda *_args,**_kwargs:(selected,{'status':'selected','attempts':[],'selected_choices':selected}))
    monkeypatch.setattr(local_matching,'_refine_model_selected_xyz_frames',lambda _m,_p,frames:(frames,[]))
    def project_group(_mesh,points,frames,_detail,_choices):
        assert points[0]['xyz']==[1,2,3] and points[0]['coordinate_system']=='ALI_A'
        return ([{'face_id':'face','position':[0,0,2]}],[],[]),None
    monkeypatch.setattr(local_matching,'_project_model_selected_xyz_group',project_group)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-15.png'])
    monkeypatch.setattr(atlas,'render_registration_group_atlas',
        lambda _m,_s,path,**_k:Image.new('RGB',(64,64),'white').save(path))
    result=local_matching._mark_model_selected_xyz_group(
        Store(),project,item,detail,Model(),{'source_sha256':'cad-hash','faces':[]},candidate)
    if echo=='different':
        assert result['result']=='abstain_or_invalid' and not Store.annotations
        return
    assert result['result']=='accepted'
    assert result['coordinate_variant_audit']['selected_choices']==selected


def test_explicit_slash_abbreviated_msop_point_lists_resolve_only_named_group():
    z_names=['F1_PNT_Z1','F1_PNT_Z2','F2_PNT_Z1','F2_PNT_Z2']
    named={name:{} for name in z_names}
    methods=[{'page':48,'text':'Measure 4 vector points F1_PNT_Z1/Z2, F2_PNT_Z1/Z2.'}]
    assert _explicit_measure_point_names(methods,named)==set(z_names)

    datum_names=['DATUMC_P04','DATUMC_P03','DATUMC_P47','DATUMC_P46']
    named={name:{} for name in datum_names}
    methods=[{'page':97,'text':'Measure 4 vector points DATUMC_P04/03/47/46.'}]
    assert _explicit_measure_point_names(methods,named)==set(datum_names)


def test_scope_conflict_occurrences_become_exact_local_model_choices():
    point={'name':'F1_PNT_L2','coordinate_conflict':True,'scope_ambiguous':False,
        'occurrences':[{'page':29,'program':'UP','coordinate_system':None,
            'coordinate_system_candidates':['ALI_ABC','ALI_ABC_A'],'xyz':[-7.14,4.37,1.465]}]}
    choices=[{'name':'F1_PNT_L2','page':29,'program':'UP','coordinate_system':'ALI_ABC',
              'xyz':[-7.14,4.37,1.465]}]
    valid,normalized=local_matching._validate_coordinate_variant_choices([point],choices)
    assert valid and normalized==choices

    unresolved={'name':'P04','scope_ambiguous':True,'unresolved_scopes':[
        {'program':'UP','coordinate_system':'ALI_ABC_A','pages':[13],'xyz':[-18.57,129.22,-0.35]}]}
    valid,normalized=local_matching._validate_coordinate_variant_choices([unresolved],[
        {'name':'P04','page':13,'program':'UP','coordinate_system':'ALI_ABC_A',
         'xyz':[-18.57,129.22,-0.35]}])
    assert valid and len(normalized)==1


def test_xyz_projection_uses_real_cad_sections_without_relaxing_group_shape_gate():
    import numpy as np
    points=[{'name':'P_A','xyz':[0,0,0],'program':'UP','coordinate_system':'ALI_A','pages':[1]},
            {'name':'P_B','xyz':[0,10,0],'program':'UP','coordinate_system':'ALI_A','pages':[1]}]
    frames=[{'program':'UP','coordinate_system':'ALI_A',
             'rotation':np.eye(3).tolist(),'translation_mm':[0,0,0]}]
    for offset,expected in ((.2,None),(.4,'projected_group_shape_drift_exceeded_gate')):
        faces=[]
        for i,(y,dz) in enumerate(((0,offset),(10,-offset))):
            xyz=[[-1,y-1,-1+dz],[1,y-1,1+dz],
                 [1,y+1,1+dz],[-1,y+1,-1+dz]]
            faces.append({'id':f'face{i}','positions':np.asarray(xyz).ravel().tolist(),
                          'indices':[0,1,2,0,2,3]})
        mesh={'source_sha256':f'section-fixture-{offset}','faces':faces}
        result,error=local_matching._project_model_selected_xyz_group(mesh,points,frames,{},[])
        if expected:
            assert result is None and error['reason']==expected
        else:
            assert error is None
            anchors,rows,audit=result
            assert len(anchors)==2
            assert np.allclose([r['position'][2] for r in rows],[0,0],atol=1e-9)
            assert np.allclose([r['surface_residual_mm'] for r in rows],[.2,.2])
            assert audit[0]['source_sections'][0]['residual_spread_mm']<1e-9


def test_many_coordinate_conflicts_are_selected_locally_in_complete_image_grounded_batches():
    points=[{'name':f'P{i}','requires_variant_choice':True,'coordinate_variants':[
        {'page':2,'program':'UP','coordinate_system':'ALI_A','xyz':[i,0,0]},
        {'page':3,'program':'UP','coordinate_system':'ALI_B','xyz':[i,1,0]}]}
        for i in range(13)]
    class Model:
        calls=0
        def _generate_once(self,messages,**kwargs):
            self.calls+=1
            question=messages[0]['content']
            rows=json.JSONDecoder().raw_decode(question.split('Competing source rows: ',1)[1])[0]
            assert len(rows)<=6
            assert messages[0]['attachments'][0]['images']==['source-page.png']
            assert kwargs['max_images']==1
            choices=[{'name':row['name'],**row['variants'][0]} for row in rows]
            return json.dumps({'decision':'SELECT_VARIANTS','fai':7,
                'coordinate_choices':choices,'confidence':'medium',
                'evidence':'Original page coordinate blocks.'}),False,200
    model=Model()
    choices,audit=_select_coordinate_variants_with_local_model(
        model,7,points,'Measure the complete 13-point group.',source_images=['source-page.png'])
    assert model.calls==3
    assert audit['status']=='selected' and len(audit['chunks'])==3
    assert len(choices)==13 and {row['name'] for row in choices}=={p['name'] for p in points}


@pytest.mark.parametrize('base_passes_geometry',[False,True])
def test_fai_specific_alternatives_are_offered_even_when_base_passes_geometry(tmp_path,monkeypatch,
                                                                             base_passes_geometry):
    monkeypatch.setattr(local_matching,'_review_visual_entity_group',lambda *a,**k:{'status':'confirmed'})
    from PIL import Image
    import oracle_viewer.cad_view_atlas as atlas
    points=[{'name':f'P{i}','xyz':[i,0,0],'pages':[1],
             'program':'UP','coordinate_system':'ALI_A'} for i in range(3)]
    base={'program':'UP','coordinate_system':'ALI_A','translation_mm':[0,0,0],
          'rotation':[[1,0,0],[0,1,0],[0,0,1]]}
    alternative={**base,'hypothesis_id':'FAI_REVIEW_H01',
                 'rotation':[[0,-1,0],[1,0,0],[0,0,1]]}
    candidate={'known_source_points':points,'evidence':{'registration_diagnostic':{
        'selected_frame_transforms':[base],'frame_transform_hypotheses':[alternative]}}}
    class Store:
        def __init__(self):self.annotations=[]
        def _dir(self,_):
            (tmp_path/'automatic-learning').mkdir(exist_ok=True)
            return tmp_path
        def annotate(self,*args,**kwargs):self.annotations.append(args)
    class Model:
        def __init__(self,decision):self.decision=decision
        def _generate_once(self,messages,**kwargs):
            context=messages[0]['attachments'][0]['parsed_text']
            option_ids=__import__('re').findall(r'"candidate_id"\s*:\s*"(H\d+)"',context)
            chosen='H02' if base_passes_geometry else 'H01'
            assert chosen in option_ids
            if base_passes_geometry:assert 'BASE' in context
            return json.dumps({'decision':self.decision,'fai':9,'source_point_count':3,
                'candidate_id':chosen,'source_pages':[1],'confidence':'medium',
                'evidence':'The complete alternative aligns with the original method.'}),False,80
    def project(_m,_p,frames,_d,_c):
        if frames[0]['rotation']==base['rotation']:
            if not base_passes_geometry:return None,{'reason':'base_group_rejected'}
        return ([{'face_id':'face','position':[0,0,0]} for _ in points],[],[]),None
    monkeypatch.setattr(local_matching,'_project_model_selected_xyz_group',project)
    monkeypatch.setattr(local_matching,'_refine_model_selected_xyz_frames',lambda _m,_p,f:(f,[]))
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-1.png'])
    monkeypatch.setattr(atlas,'render_registration_group_atlas',
        lambda _m,_s,path,**_k:Image.new('RGB',(64,64),'white').save(path))
    for decision,expected in [('ABSTAIN','abstain_or_invalid'),('AUTO_MARK','accepted')]:
        store=Store()
        result=local_matching._mark_model_selected_xyz_group(store,
            {'id':'project123'}, {'number':9,'document_id':'msop-doc'},
            {'methods':[{'page':1,'text':'Measure P0 through P2.'}]},Model(decision),
            {'source_sha256':'cad-hash','faces':[]},candidate)
        assert result['result']==expected
        assert bool(store.annotations)==(decision=='AUTO_MARK')


def test_compact_xyz_group_cannot_mark_incomplete_or_extra_source_locations():
    for required,actual in ((4,3),(3,4)):
        candidate={'required_marker_count':required,'known_source_points':[
            {'name':f'P{i}','xyz':[i,0,0]} for i in range(actual)]}
        class Model:
            def _generate_once(self,*args,**kwargs):
                raise AssertionError('Incomplete source group must not be offered for acceptance')
        result=local_matching._mark_model_selected_xyz_group(
            None,{'id':'p'},{'number':1},{},Model(),{},candidate)
        assert result is None


def test_singleton_model_json_list_preserves_one_decision_without_selecting_among_decisions():
    decision={'decision':'AUTO_MARK','fai':7,'candidate_id':'BASE'}
    assert local_matching._object(json.dumps([decision]))==decision
    assert local_matching._object(json.dumps([decision,{'decision':'ABSTAIN','fai':7}])) is None


def test_explicit_manifest_keeps_printed_names_even_when_no_xyz_rows_exist():
    methods=[{'page':48,'text':'Measure 4 vector points F1_PNT_Z1/Z2, F2_PNT_Z1/Z2.'}]
    groups=local_matching._declared_marker_manifest(methods)
    assert groups[0]['source_point_names']==['F1_PNT_Z1','F1_PNT_Z2','F2_PNT_Z1','F2_PNT_Z2']
    assert local_matching._declared_marker_manifest([
        {'page':1,'text':'Measure 4 vector points as Plane View show.'}]) is None
    assert local_matching._declared_marker_manifest([
        {'page':1,'text':'Measure 1 vision circle CIR_N1 as Plane View show.'}]) is None
    groups=local_matching._declared_marker_manifest([
        {'page':2,'text':'Measure 3 vector points DATUMC_P04~06.'}])
    assert groups[0]['source_point_names']==['DATUMC_P04','DATUMC_P05','DATUMC_P06']
    groups=local_matching._declared_marker_manifest([{'page':48,'text':
        'Measure 2 circles FAI38_CIR_L& FAI38_CIR_R，Use top circle of Fang Hole. '
        'Measure 4 vector points F1_PNT_Z1/Z2, F2_PNT_Z1/Z2, then construct 2 lines F1_Z_LIN, F2_Z_LIN.'}])
    assert {name for group in groups for name in group['source_point_names']}=={
        'FAI38_CIR_L','FAI38_CIR_R','F1_PNT_Z1','F1_PNT_Z2','F2_PNT_Z1','F2_PNT_Z2'}


def test_local_manifest_confirmation_removes_unrelated_family_points_without_borrowing_xyz():
    methods=[{'page':48,'program':'UP','measured_system':'ALI_A',
              'text':'Measure 4 vector points F1_PNT_Z1/Z2, F2_PNT_Z1/Z2.'}]
    names=['F1_PNT_Z1','F1_PNT_Z2','F2_PNT_Z1','F2_PNT_Z2']
    unrelated=[{'name':name,'xyz':[1,2,3]} for name in
               ['F1_PNT_L2','F1_PNT_L3','F2_PNT_R2','F2_PNT_R3','F1_CIR_L','F1_CIR_R']]
    class Model:
        def _generate_once(self,messages,**kwargs):
            assert messages[0]['attachments'][0]['images']==['original-msop.png']
            return json.dumps({'decision':'CONFIRM_MANIFEST','fai':37,
                'source_point_names':names,'confidence':'high',
                'evidence':'The method explicitly lists the four Z points.'}),False,120
    points,audit=local_matching._resolve_declared_marker_manifest_with_local_model(
        Model(),37,methods,unrelated,4,['original-msop.png'])
    assert [point['name'] for point in points]==names
    assert all(point['xyz'] is None for point in points)
    assert all(point['explicit_method_manifest'] for point in points)
    assert audit['status']=='confirmed' and len(audit['excluded_retrieved_names'])==6
    issues=_visual_marker_validation_issues({'decision':'AUTO_MARK','fai':37,'markers':[
        {'source_point_names':['F1_PNT_Z1']},{'source_point_names':['F1_PNT_Z2']},
        {'source_point_names':['F2_PNT_Z1']},{'source_point_names':['F1_PNT_L2']}]},
        fai=37,required_marker_count=4,known_source_points=points,rigid_group_possible=False,
        candidate_selection_valid=True,coordinate_choices_valid=True,truncated=False)
    assert any(issue['code']=='source_point_coverage_incomplete'
               and issue['missing_names']==['F2_PNT_Z2'] for issue in issues)


def test_complete_confirmed_manifest_with_missing_xyz_can_enter_visual_recovery_without_aliasing():
    strategy={'next_route':'evidence_repair'}
    manifest={'status':'confirmed','coordinate_evidence_missing':['R1_PNT_Z3'],
        'groups':[{'declared_count':1,'source_point_names':['R1_PNT_P2']},
                  {'declared_count':2,'source_point_names':['S1_PNT_Z1','R1_PNT_Z3']}]}
    points=[{'name':'R1_PNT_P2','xyz':[-28.96,9.73,2.01],'explicit_method_manifest':True},
            {'name':'S1_PNT_Z1','xyz':[-30.56,4.43,0.0],'explicit_method_manifest':True},
            {'name':'R1_PNT_Z3','xyz':None,'explicit_method_manifest':True}]
    recovery=local_matching._source_relation_visual_recovery(strategy,manifest,points,3)
    assert recovery['missing_xyz_source_point_names']==['R1_PNT_Z3']
    assert recovery['source_xyz_substitution_allowed'] is False
    # Similar retrieved names remain excluded; a direct measured name cannot
    # inherit their XYZ through this route.
    assert points[2]['xyz'] is None


@pytest.mark.parametrize('points,count', [
    ([{'name':'R1_PNT_P2','xyz':None,'explicit_method_manifest':True}],1),
    ([{'name':'R1_PNT_P2','xyz':None,'explicit_method_manifest':True},
      {'name':'S1_PNT_Z1','xyz':None}],2),
])
def test_visual_recovery_requires_every_direct_name_to_be_confirmed(points,count):
    manifest={'status':'confirmed','coordinate_evidence_missing':[row['name'] for row in points],
        'groups':[{'declared_count':count,'source_point_names':['R1_PNT_P2','S1_PNT_Z1']}]}
    assert local_matching._source_relation_visual_recovery(
        {'next_route':'evidence_repair'},manifest,points,count) is None


def test_partial_xyz_evidence_repair_gets_optional_local_host_part_selection():
    assert local_matching._part_identity_route({'next_route':'evidence_repair',
        'evidence_class':'partial_xyz'})==(False,True)
    assert local_matching._part_identity_route({'next_route':'evidence_repair',
        'evidence_class':'named_features_no_xyz','failure_class':'source_manifest'})==(False,True)
    assert local_matching._part_identity_route({'next_route':'evidence_repair',
        'evidence_class':'complete_xyz_single_frame'})==(False,False)
    assert local_matching._part_identity_route({'next_route':'feature_topology_graph',
        'evidence_class':'unknown'})==(True,False)


def test_entity_review_coordinate_audit_maps_xyz_before_world_comparison():
    import numpy as np
    source=[{'name':'P1','xyz':[-28.96,9.73,2.01],
        'program':'UP','coordinate_system':'ALI_ABC'}]
    locations=[{'source_point_names':['P1']}]
    anchors=[{'position':[28.93,9.64,0.098]}]
    fits={('UP','ALI_ABC'):{'rotation':np.eye(3),'translation':np.array([57.89,-0.09,-1.912]),
        'max_residual_mm':0.01,'rms_residual_mm':0.005}}
    audit=local_matching._entity_coordinate_frame_audit(source_points=source,
        rigid_fits=fits,locations=locations,anchors=anchors)
    assert audit['status']=='registered_mapping_available'
    assert audit['raw_source_xyz_comparable_to_cad_world'] is False
    assert audit['source_xyz_to_cad_xyz'][0]['transformed_xyz']==pytest.approx([28.93,9.64,0.098])
    assert audit['source_xyz_to_cad_xyz'][0]['position']==anchors[0]['position']
    assert audit['numeric_comparison_allowed_only_after_transform'] is True


def test_entity_review_coordinate_audit_explicitly_marks_unregistered_xyz():
    audit=local_matching._entity_coordinate_frame_audit(source_points=[{'name':'P1','xyz':[1,2,3],
        'program':'UP','coordinate_system':'ALI_ABC'}],locations=[{'source_point_names':['P1']}],
        anchors=[{'position':[4,5,6]}])
    assert audit['status']=='partial_or_unregistered'
    assert audit['raw_source_xyz_comparable_to_cad_world'] is False
    assert audit['source_xyz_to_cad_xyz'][0]['transformed_xyz'] is None


def test_manifest_model_cannot_substitute_similar_names_or_authorize_a_partial_list():
    methods=[{'page':1,'text':'Measure 2 vector points F1_PNT_Z1/Z2.'}]
    class Model:
        def _generate_once(self,*args,**kwargs):
            return json.dumps({'decision':'CONFIRM_MANIFEST','fai':1,
                'source_point_names':['F1_PNT_L2','F1_PNT_L3'],
                'confidence':'low','evidence':'Similar feature family.'}),False,70
    points,audit=local_matching._resolve_declared_marker_manifest_with_local_model(
        Model(),1,methods,[{'name':'F1_CIR_L','xyz':[1,2,3]}],2,[])
    assert points is None and audit['status']=='abstained'


def test_confirmed_manifest_reuses_only_unique_exact_source_name_alias_coordinates():
    methods=[{'page':1,'text':'Measure 2 vector points DATUMB_P09, DATUMB_P10.'}]
    class Model:
        def _generate_once(self,*args,**kwargs):
            return json.dumps({'decision':'CONFIRM_MANIFEST','fai':1,
                'source_point_names':['DATUMB_P09','DATUMB_P10'],
                'confidence':'high','evidence':'The two printed direct measurement points.'}),False,70
    points,audit=local_matching._resolve_declared_marker_manifest_with_local_model(
        Model(),1,methods,[{'name':'DATUMB_PNT_P09','xyz':[1,2,3]},
                          {'name':'DATUMB_PNT_P11','xyz':[4,5,6]}],2,[])
    assert points[0]['name']=='DATUMB_PNT_P09' and points[0]['xyz']==[1,2,3]
    assert points[1]['name']=='DATUMB_P10' and points[1]['xyz'] is None


def test_duplicate_point_names_across_alignments_remain_model_choices_not_extra_markers():
    points=[{'name':'P1','xyz':[1,2,3],'pages':[1],'program':'UP','coordinate_system':'ALI_A'},
            {'name':'P2','xyz':[4,5,6],'pages':[1],'program':'UP','coordinate_system':'ALI_A'},
            {'name':'P1','xyz':[7,8,9],'pages':[2],'program':'DOWN','coordinate_system':'ALI_B'}]
    result=local_matching._coalesce_source_point_definitions(points)
    assert [point['name'] for point in result]==['P1','P2']
    assert result[0]['xyz'] is None and result[0]['requires_variant_choice'] is True
    assert result[0]['program'] is None and result[0]['coordinate_system'] is None
    assert {(row['program'],row['coordinate_system'],tuple(row['xyz']))
            for row in result[0]['coordinate_variants']}=={
        ('UP','ALI_A',(1,2,3)),('DOWN','ALI_B',(7,8,9))}
    assert result[1]==points[1]


def test_coherent_flat_face_points_are_not_moved_sideways_to_nominal_plane_intersection():
    import numpy as np
    points=[{'name':f'P_{i}','xyz':[x,y,0],'pages':[1],'program':'UP','coordinate_system':'ALI_A'}
            for i,(x,y) in enumerate(((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)))]
    mesh={'source_sha256':'flat-face-small-pose-error','faces':[{'id':'flat',
        'positions':[-2,-2,2,2,-2,2,2,2,2,-2,2,2],'indices':[0,1,2,0,2,3]}]}
    frame={'program':'UP','coordinate_system':'ALI_A','rotation':
        local_matching.Rotation.from_euler('y',.1,degrees=True).as_matrix().tolist(),
        'translation_mm':[0,0,2.00001]}
    result,error=local_matching._project_model_selected_xyz_group(mesh,points,[frame],{},[])
    assert error is None
    anchors,rows,audit=result
    assert len(anchors)==4
    assert np.allclose([row['position'][2] for row in rows],2)
    assert np.allclose(np.asarray([row['position'] for row in rows])[:,:2],
                       np.asarray([point['xyz'] for point in points])[:,:2],atol=1e-5)
    assert audit[0]['source_sections'][0]['residual_spread_mm']<.001


def test_plane_alignment_generates_only_bounded_complete_rigid_frame_hypotheses():
    import numpy as np
    points=[{'name':f'P_{i}','xyz':[x,y,0],'pages':[1],'program':'UP','coordinate_system':'ALI_A'}
            for i,(x,y) in enumerate(((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)))]
    mesh={'source_sha256':'planar-frame-proposal-test','faces':[{'id':'flat',
        'positions':[-2,-2,2,2,-2,2,2,2,2,-2,2,2],'indices':[0,1,2,0,2,3]}]}
    frame={'program':'UP','coordinate_system':'ALI_A','rotation':
        local_matching.Rotation.from_euler('x',.2,degrees=True).as_matrix().tolist(),
        'translation_mm':[0,0,2.0005]}
    options=local_matching._planar_frame_hypotheses(mesh,points,[frame])
    assert len(options)==1
    selected=options[0][0]
    assert np.allclose(selected['rotation'],np.eye(3),atol=1e-9)
    assert np.allclose(selected['translation_mm'],[0,0,2],atol=1e-9)
    assert set(selected['planar_alignment']['support_point_names'])=={point['name'] for point in points}
    assert selected['planar_alignment']['source_to_cad_registered'] is False
    out_of_bound={**frame,'rotation':local_matching.Rotation.from_euler('x',2,degrees=True).as_matrix().tolist()}
    assert local_matching._planar_frame_hypotheses(mesh,points,[out_of_bound])==[]


def test_coordinate_disambiguation_receives_individual_feature_and_all_variant_page_headers():
    class Store:
        def active_msop(self,_):return {'method_pages':[
            {'page':35,'heading':'FAI example: right rail','text':'Right rail coordinate definition.',
             'program':'UP','measured_system':'ALI_A'},
            {'page':52,'heading':'FAI example: left rail','text':'Left rail coordinate definition.',
             'program':'UP','measured_system':'ALI_A'},
            {'page':99,'heading':'Unrelated','text':'Unrelated coordinates.'}]}
    point={'name':'S1_PNT_X8','requires_variant_choice':True,'coordinate_variants':[
        {'page':35,'program':'UP','coordinate_system':'ALI_A','xyz':[1,2,3]},
        {'page':52,'program':'UP','coordinate_system':'ALI_A','xyz':[1,2.1,3]}]}
    context=local_matching._coordinate_disambiguation_context(Store(),{},
        {'number':65,'features':['S1 RIGHT RAIL']},[point],'A shared table contains many pull-test items.')
    assert 'S1 RIGHT RAIL' in context and 'Target FAI 65' in context
    assert 'Right rail coordinate definition.' in context and 'Left rail coordinate definition.' in context
    assert 'Unrelated coordinates.' not in context


def test_normal_origin_search_preserves_whole_group_and_cannot_fix_wrong_in_plane_location():
    import numpy as np
    points=[{'name':f'P_{i}','xyz':[x,y,0],'pages':[1],'program':'UP','coordinate_system':'ALI_A'}
            for i,(x,y) in enumerate(((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)))]
    mesh={'source_sha256':'normal-origin-synthetic-frame','faces':[{'id':'flat',
        'positions':[-2,-2,2,2,-2,2,2,2,2,-2,2,2],'indices':[0,1,2,0,2,3]}]}
    frame={'program':'UP','coordinate_system':'ALI_A',
           'rotation':np.eye(3).tolist(),'translation_mm':[0,0,0]}
    options=local_matching._normal_origin_frame_hypotheses(mesh,points,[frame])
    assert len(options)==1
    proposed=options[0][0]
    assert np.allclose(proposed['rotation'],np.eye(3))
    assert np.allclose(proposed['translation_mm'],[0,0,2])
    assert proposed['normal_origin_search']['source_coordinates_modified'] is False
    assert proposed['normal_origin_search']['metrology_accepted'] is False
    shifted=[{**point,'xyz':[point['xyz'][0]+10,point['xyz'][1],0]} for point in points]
    assert local_matching._normal_origin_frame_hypotheses(mesh,shifted,[frame])==[]


def test_manifest_retry_explains_numeric_confidence_protocol_error():
    methods=[{'page':1,'text':'Measure 2 vector points DATUMB_P09, DATUMB_P10.'}]
    class Model:
        calls=0
        def _generate_once(self,messages,**kwargs):
            self.calls+=1
            if self.calls==2:
                assert 'confidence_must_be_high_medium_or_low_string' in messages[0]['content']
            return json.dumps(dict(decision='CONFIRM_MANIFEST',fai=1,
                source_point_names=['DATUMB_P09','DATUMB_P10'],evidence='Printed list',
                confidence=1.0 if self.calls==1 else 'high')),False,80
    points,audit=local_matching._resolve_declared_marker_manifest_with_local_model(
        Model(),1,methods,[],2,[])
    assert points is not None and len(points)==2
    assert audit['attempts'][0]['validation_errors']==['confidence_must_be_high_medium_or_low_string']
    assert audit['attempts'][1]['valid']
