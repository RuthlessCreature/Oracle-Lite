import json
import copy
import threading
import pytest
from types import SimpleNamespace

import numpy as np
import oracle_viewer.automatic_candidates as automatic_candidates

from oracle_viewer.automatic_candidates import (build_candidates, _fit_bounded_tangent_shift,
    _fit_bounded_common_translation, _fit_bounded_minimax_translation, _required_point_count,
    _required_marker_count, _source_point_evidence, _same_source_stale_seed)
from oracle_viewer.local_matching import mark_geometry_candidates, _fai_number, _source_scope_conflicts, _prompt_data, _is_local_inference_model
from oracle_viewer.surface_index import SurfaceIndex
from oracle_viewer.auto_registration import registration_algorithm_sha256


class Store:
    def __init__(self, root):
        self.root=root
        self.project=dict(id='p',drawing_no='generic',revision='A')
        self.page=dict(page=1,program='UP',measured_system='ALI_A',text='''
            Measure 3 vector points PNT_A~PNT_C.
            X, Y, Z (mm) 1 1 0 PNT_A
            X, Y, Z (mm) 2 1 0 PNT_B
            X, Y, Z (mm) 1 2 0 PNT_C''')
        self.item=dict(number=1,document_id='msop',features=[],pages=[1],annotation=None,binding=None)
        self.annotations=[]
        self.removed=[]
        (root/'automatic-learning').mkdir()
        support=np.random.default_rng(7).normal(size=(10,3))
        receipt=dict(cad_sha256='cad',msop_sha256='msop',
                     algorithm_sha256=registration_algorithm_sha256(),frames=[
            dict(program='UP',coordinate_system='ALI_A',source_points=[dict(xyz=p.tolist()) for p in support],
                 triangle_refinement=dict(status='surface_residual_rejected',threshold_mm=.1,
                                          best=dict(rotation=np.eye(3).tolist(),translation_mm=[0,0,0],max_residual_mm=.9)))])
        receipt['local_model_registration']=dict(status='complete',frames=[dict(
            program='UP',coordinate_system='ALI_A',status='selected',hypothesis_id='H00',hypothesis_index=0)])
        (root/'automatic-learning'/'registration.json').write_text(json.dumps(receipt))
    def _dir(self,pid):return self.root
    def get(self,pid):return self.project
    def active_msop(self,p):return dict(sha256='msop',method_pages=[self.page])
    def mesh(self,pid):return dict(source_sha256='cad',faces=[dict(id='f',positions=[0,0,0,5,0,0,0,5,0],
                                                                    indices=[0,1,2],occt_face_valid=True)])
    def items(self,p):return [self.item]
    def detail(self,pid,n):return dict(methods=[self.page])
    def annotate(self,*args,**kwargs):self.annotations.append(args)
    def remove_annotation(self,pid,number):self.removed.append(number)
    def update_local_model_marking(self,*args,**kwargs):pass


def test_dry_run_preserves_entire_source_plane_without_persisting_geometry_guess(tmp_path):
    store=Store(tmp_path)
    report=build_candidates(store,'p')
    assert len(report['candidates'])==1 and not store.annotations
    candidate=report['candidates'][0]
    assert len(candidate['anchors'])==3
    assert all(a['position'][2]==0 for a in candidate['anchors'])
    assert not report['source_to_cad_registered'] and not report['metrology_accepted']
    assert candidate['evidence']['frame_hypotheses'][0]['global_status']=='surface_residual_rejected'


def test_candidate_builder_never_uses_algorithm_best_without_local_model_selection(tmp_path):
    store=Store(tmp_path)
    path=tmp_path/'automatic-learning'/'registration.json'
    receipt=json.loads(path.read_text());receipt.pop('local_model_registration')
    path.write_text(json.dumps(receipt))
    report=build_candidates(store,'p')
    assert len(report['candidates'])==1
    candidate=report['candidates'][0]
    assert candidate['candidate_kind']=='model_visual_localization'
    assert candidate['anchors']==[] and not store.annotations
    assert candidate['evidence']['registration_diagnostic']['no_geometric_anchor_authorized']
    assert report['items'][0]['result']=='local_visual_registration_fallback_ready'
    assert report['items'][0]['geometry_result']=='local_model_did_not_select_registration_hypothesis'


@pytest.mark.parametrize('receipt_state', ['missing', 'stale'])
def test_unregistered_or_stale_project_still_routes_fai_to_local_visual_model(tmp_path, receipt_state):
    store=Store(tmp_path)
    path=tmp_path/'automatic-learning'/'registration.json'
    if receipt_state=='missing':
        path.unlink()
    else:
        receipt=json.loads(path.read_text());receipt['algorithm_sha256']='stale'
        path.write_text(json.dumps(receipt))
    report=build_candidates(store,'p')
    assert report['registration_status']==('missing' if receipt_state=='missing' else 'stale_receipt')
    assert len(report['items'])==1 and len(report['candidates'])==1
    candidate=report['candidates'][0]
    assert candidate['candidate_kind']=='model_visual_localization'
    assert candidate['anchors']==[]
    assert candidate['evidence']['placement_decision']=='local_model_required'


def test_required_marker_count_reads_generic_x_location_callouts():
    methods=[dict(program=None,measured_system=None,text='6X WCA MEASUREMENT LOCATIONS')]
    assert _required_marker_count(methods)==6


def test_required_marker_count_sums_independent_points_and_vector_features():
    methods=[dict(program='UP',measured_system='ALI_ABC',text=(
        'Measure 2 vector circles F1_CIR_L& F1_CIR_R.\n'
        'Measure 9 vector points DATUMB_P09~P17 and Construct 1 line DATUMB_LEFT.\n'
        'Measure 9 vector points DATUMB_P33~P41 and Construct 1 line DATUMB_RIGHT.\n'
        'Construct 1 line DATUMB_LIN.'))]
    named={'F1_CIR_L':{},'F1_CIR_R':{},'DATUMB_P09':{},'DATUMB_P10':{},'DATUMB_P11':{},'DATUMB_P12':{},'DATUMB_P13':{},
           'DATUMB_P14':{},'DATUMB_P15':{},'DATUMB_P16':{},'DATUMB_P17':{},
           'DATUMB_P33':{},'DATUMB_P34':{},'DATUMB_P35':{},'DATUMB_P36':{},'DATUMB_P37':{},
           'DATUMB_P38':{},'DATUMB_P39':{},'DATUMB_P40':{},'DATUMB_P41':{}}
    assert _required_marker_count(methods,named)==20


def test_required_marker_count_includes_bare_measured_circles_and_points():
    methods=[dict(program='UP',measured_system='ALI_ABC',text=(
        'Measure 2 circles FAI38_CIR_L and FAI38_CIR_R.\n'
        'Measure 4 vector points F1_PNT_Z1/Z2, F2_PNT_Z1/Z2.'))]
    assert _required_marker_count(methods)==6


def test_required_marker_groups_keep_each_distinct_msop_instruction():
    from oracle_viewer.automatic_candidates import _required_marker_groups
    methods=[dict(page=33,program='UP',measured_system='ALI_ABC',text=(
        'Measure 2 vector circles F1_CIR_L, F1_CIR_R.\n'
        'Measure 2 vector points PNT_A,PNT_B.\n'
        'Measure 2 vector points PNT_C,PNT_D.\n'
        'Construct 1 line DATUMB_LIN.')),
        dict(page=34,program='UP',measured_system='ALI_ABC',text=(
        'Measure 2 vector circles F1_CIR_L, F1_CIR_R.'))]
    named={name:{} for name in ('F1_CIR_L','F1_CIR_R','PNT_A','PNT_B','PNT_C','PNT_D')}
    groups=_required_marker_groups(methods,named)
    assert [(group['kind'],group['declared_count']) for group in groups]==[
        ('measured_points',2),('measured_points',2),('measured_features',2)]
    assert groups[2]['source_point_names']==['F1_CIR_L','F1_CIR_R']
    assert 'DATUMB_LIN' not in ' '.join(group['instruction'] for group in groups)


def test_conflicting_xyz_rows_are_passed_to_model_as_alternatives_without_default():
    row=dict(name='DATUMC_PNT_P04',xyz=[-18.57,129.22,-.35],pages=[13],
        program=None,coordinate_system=None,coordinate_conflict=True,scope_ambiguous=True,
        occurrences=[
            dict(page=13,program='UP',coordinate_system='ALI_ABC_A',xyz=[-18.57,129.22,-.35]),
            dict(page=75,program='DOWN',coordinate_system='ALI_ABC_A',xyz=[-18.57,64.52,-.35]),
        ])
    point=_source_point_evidence([row])[0]
    assert point['xyz'] is None
    assert point['requires_variant_choice'] is True
    assert point['coordinate_variants']==[
        dict(page=13,program='UP',coordinate_system='ALI_ABC_A',xyz=[-18.57,129.22,-.35],
             frame_assignment_conflict=False),
        dict(page=75,program='DOWN',coordinate_system='ALI_ABC_A',xyz=[-18.57,64.52,-.35],
             frame_assignment_conflict=False),
    ]


def test_registration_scope_search_includes_exact_unresolved_source_variants():
    from oracle_viewer.automatic_candidates import _source_point_scopes
    points=[{'name':'P1','program':None,'coordinate_system':None,
             'coordinate_variants':[
                 {'program':'UP','coordinate_system':'ALI_A'},
                 {'program':'DOWN','coordinate_system':'ALI_B'}]},
            {'name':'P2','program':'UP','coordinate_system':'ALI_C'},
            {'name':'P3','program':'UP','coordinate_system':None}]
    assert _source_point_scopes(points)=={
        ('UP','ALI_A'),('DOWN','ALI_B'),('UP','ALI_C')}


def test_fai_frame_hypotheses_preserve_scope_and_reject_invalid_transforms():
    from oracle_viewer.automatic_candidates import _registration_frame_hypotheses
    proper={'rotation':np.eye(3).tolist(),'translation_mm':[0,0,2]}
    improper={'rotation':np.diag([-1,1,1]).tolist(),'translation_mm':[0,0,2]}
    receipt={'frames':[
        {'program':'UP','coordinate_system':'ALI_A',
         'triangle_refinement':{'best':proper,'alternatives':[proper,improper]}},
        {'program':'DOWN','coordinate_system':'ALI_A',
         'triangle_refinement':{'best':proper}}]}
    options=_registration_frame_hypotheses(receipt,{('UP','ALI_A')})
    assert len(options)==1
    assert options[0]['hypothesis_id']=='FAI_REVIEW_H00'
    assert options[0]['program']=='UP' and options[0]['coordinate_system']=='ALI_A'
    assert options[0]['candidate_source']=='geometric_hypothesis_pending_local_fai_model_review'


def test_dominant_page_resolution_keeps_alternative_rows_visible_to_local_model():
    row=dict(name='S1_PNT_X8',xyz=[-29.03,7.23,2.5],pages=[52],program='UP',
        coordinate_system='ALI_ABD',coordinate_conflict=False,scope_ambiguous=False,
        occurrences=[dict(page=52,program='UP',coordinate_system='ALI_ABD',xyz=[-29.03,7.23,2.5])],
        source_coordinate_variants=[
            dict(page=35,program='UP',coordinate_system='ALI_ABD',xyz=[-29.03,7.16,2.6]),
            dict(page=52,program='UP',coordinate_system='ALI_ABD',xyz=[-29.03,7.23,2.5]),
        ])
    point=_source_point_evidence([row])[0]
    assert point['xyz'] is None
    assert point['requires_variant_choice'] is True
    assert [v['page'] for v in point['coordinate_variants']]==[35,52]


def test_source_conflict_is_routed_to_local_model_with_all_xyz_variants(tmp_path):
    store=Store(tmp_path)
    method=dict(page=4,program='DOWN',measured_system='ALI_A',fai_ids=[1],
        heading='FAI 1',text='Measure 3 vector points PNT_A~PNT_C.')
    first=dict(page=1,program='DOWN',measured_system='ALI_A',text='\n'.join([
        'X, Y, Z (mm) 1 1 0 PNT_A','X, Y, Z (mm) 2 1 0 PNT_B','X, Y, Z (mm) 1 2 0 PNT_C']))
    alternate=dict(page=2,program='DOWN',measured_system='ALI_A',text='\n'.join([
        'X, Y, Z (mm) 1 1.2 0 PNT_A','X, Y, Z (mm) 2 1 0 PNT_B','X, Y, Z (mm) 1 2 0 PNT_C']))
    store.page=method
    store.item.update(pages=[4])
    store.active_msop=lambda _project:dict(sha256='msop',method_pages=[method,first,alternate])
    receipt_path=tmp_path/'automatic-learning'/'registration.json'
    receipt=json.loads(receipt_path.read_text())
    receipt['frames'][0]['program']='DOWN'
    receipt['local_model_registration']['frames'][0]['program']='DOWN'
    receipt_path.write_text(json.dumps(receipt))
    report=build_candidates(store,'p')
    candidate=next(row for row in report['candidates'] if row['fai']==1)
    points={row['name']:row for row in candidate['known_source_points']}
    assert report['items'][0]['result']=='local_visual_completion_candidate_ready'
    assert points['PNT_A']['xyz'] is None
    assert {row['page'] for row in points['PNT_A']['coordinate_variants']}=={1,2}
    diagnostic=candidate['evidence']['registration_diagnostic']
    assert diagnostic['selected_frame_transforms'][0]['coordinate_system']=='ALI_A'
    assert diagnostic['frame_transform_hypotheses'][0]['coordinate_system']=='ALI_A'


def test_candidate_anchors_follow_local_model_selected_nondefault_transform(tmp_path):
    store=Store(tmp_path)
    path=tmp_path/'automatic-learning'/'registration.json'
    receipt=json.loads(path.read_text())
    fit=receipt['frames'][0]['triangle_refinement']
    fit['best']['translation_mm']=[0,0,.08]
    fit['alternatives']=[dict(rotation=np.eye(3).tolist(),translation_mm=[0,0,0],max_residual_mm=.01)]
    receipt['local_model_registration']['frames'][0].update(
        hypothesis_id='H01',hypothesis_index=1,status='selected')
    path.write_text(json.dumps(receipt))

    report=build_candidates(store,'p')

    assert len(report['candidates'])==1
    selected=report['candidates'][0]
    assert all(abs(anchor['position'][2])<1e-8 for anchor in selected['anchors'])
    frame=selected['evidence']['frame_hypotheses'][0]
    assert frame['selected_by']=='local_model'
    assert frame['selected_hypothesis_id']=='H01'
    assert frame['translation_mm']==[0,0,0]


def test_candidate_generation_accepts_noncollinear_planar_frame_support(tmp_path):
    store=Store(tmp_path)
    path=tmp_path/'automatic-learning'/'registration.json'
    receipt=json.loads(path.read_text())
    receipt['frames'][0]['source_points']=[
        {'xyz':[float(x),float(y),0.0]}
        for x,y in ((0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2))]
    path.write_text(json.dumps(receipt))

    report=build_candidates(store,'p')

    assert report['candidates']
    assert len(report['candidates'][0]['anchors'])==3


def test_no_xyz_item_routes_to_local_vision_with_untrusted_same_source_seed_evidence(tmp_path):
    store=Store(tmp_path)
    store.page['text']='MSOP feature has no extracted XYZ row.'
    store.item['annotation']=dict(
        status='stale',origin='local_model_candidate',stale_reason='automatic_registration_changed',
        stale_previous_status='auto_marked',source_sha256='cad',document_id='msop',
        document_sha256='msop',confidence='low',
        anchors=[dict(face_id='f',position=[1,1,0])])

    report=build_candidates(store,'p')

    assert len(report['candidates'])==1
    candidate=report['candidates'][0]
    assert candidate['candidate_kind']=='model_visual_localization'
    assert report['items'][0]['result']=='local_visual_candidate_ready'
    assert candidate['anchors']==[]
    assert candidate['evidence']['prior_stale_seed']['source_hashes_verified']
    assert not report['persisted']


def test_stale_seed_is_rejected_when_source_msop_hash_changed(tmp_path):
    store=Store(tmp_path)
    store.page['text']='MSOP feature has no extracted XYZ row.'
    store.item['annotation']=dict(
        status='stale',origin='local_model_candidate',stale_reason='automatic_registration_changed',
        stale_previous_status='auto_marked',source_sha256='cad',document_id='msop',
        document_sha256='old-msop',anchors=[dict(face_id='f',position=[1,1,0])])

    report=build_candidates(store,'p')

    assert len(report['candidates'])==1
    assert report['candidates'][0]['candidate_kind']=='model_visual_localization'
    assert report['items'][0]['result']=='local_visual_candidate_ready'


def test_missing_xyz_for_explicit_multi_point_fai_does_not_become_single_visual_pin(tmp_path):
    store=Store(tmp_path)
    store.page['text']='Measure 3 vector points PNT_A~PNT_C.'
    report=build_candidates(store,'p')
    assert len(report['candidates'])==1
    assert report['items'][0]['result']=='local_visual_candidate_ready'
    assert report['candidates'][0]['anchors']==[]
    assert report['candidates'][0]['required_marker_count']==3


def test_feature_only_fai_enters_model_visual_localization_without_precomputed_anchors(tmp_path):
    store=Store(tmp_path)
    store.page['text']='FAI visual circle CIR_N1; no coordinate table on this method.'
    store.item['features']=['CIR_N1']
    report=build_candidates(store,'p')
    assert report['items'][0]['result']=='local_visual_candidate_ready'
    assert len(report['candidates'])==1
    candidate=report['candidates'][0]
    assert candidate['candidate_kind']=='model_visual_localization'
    assert candidate['origin']=='msop_visual_candidate'
    assert candidate['anchors']==[] and not store.annotations


def test_stale_local_model_mark_is_recomputed_from_current_registration(tmp_path):
    store=Store(tmp_path)
    store.item['annotation']=dict(origin='local_model_candidate',status='stale',anchors=[{}])
    report=build_candidates(store,'p')
    assert [candidate['fai'] for candidate in report['candidates']]==[1]


def test_one_bad_point_rejects_whole_item_instead_of_truncating_points(tmp_path):
    store=Store(tmp_path);store.page['text']=store.page['text'].replace('1 2 0 PNT_C','1 2 .2 PNT_C')
    report=build_candidates(store,'p')
    assert len(report['candidates'])==1 and not store.annotations
    assert report['candidates'][0]['candidate_kind']=='model_visual_localization'
    assert report['candidates'][0]['anchors']==[]
    assert report['candidates'][0]['evidence']['registration_diagnostic']['no_geometric_anchor_authorized']
    assert report['items'][0]['geometry_result'] in {'item_point_residual_rejected', 'source_section_has_no_nearby_CAD_intersection'}
    assert report['items'][0]['source_point_count'] == 3
    assert report['items'][0]['supported_point_count'] < 3
    assert report['items'][0]['failed_points']
    assert len(report['candidates'][0]['known_source_points'])==3
    failure=report['items'][0]['failed_points'][0]
    if failure['reason']=='source_section_has_no_nearby_CAD_intersection':
        assert failure['nearest_surface_residual_mm'] is not None
        assert failure['nearest_surface_plane_residual_mm'] is not None
        assert failure['section_search_triangle_count'] >= 0


def test_declared_measurement_count_blocks_candidate_when_one_xyz_is_missing(tmp_path):
    store=Store(tmp_path)
    store.page['text']=store.page['text'].replace('Measure 3 vector points','Measure 4 vector points')
    report=build_candidates(store,'p')
    assert len(report['candidates'])==1 and not store.annotations
    assert report['items'][0]['result']=='local_visual_completion_candidate_ready'
    assert report['items'][0]['coordinate_status']=='incomplete_measurement_point_coordinates'
    assert report['items'][0]['required_point_count']==4
    assert report['items'][0]['missing_measured_point_count']==1
    assert report['candidates'][0]['required_marker_count']==4
    assert len(report['candidates'][0]['known_source_points'])==3


def test_required_count_deduplicates_repeated_method_pages_but_keeps_distinct_groups():
    pages = [
        {'program': 'UP', 'measured_system': 'ALI_A',
         'text': 'Measure 3 vector points PNT_A~PNT_C.'},
        {'program': 'UP', 'measured_system': 'ALI_A',
         'text': 'Measure 3 vector points PNT_A~PNT_C.'},
    ]
    assert _required_point_count(pages, {}) == 3
    pages.append({'program': 'UP', 'measured_system': 'ALI_A',
                  'text': 'Measure 2 vector points PNT_D,PNT_E.'})
    assert _required_point_count(pages, {}) == 5


def test_required_visual_marker_count_handles_lines_and_deduplicates_repeated_method_pages():
    pages=[{'program':'UP','measured_system':'VISION','text':'Measure 2 vision lines RAIL_LIN and CHASSIS_LIN.'},
           {'program':'UP','measured_system':'VISION','text':'Measure 2 vision lines RAIL_LIN and CHASSIS_LIN.'},
           {'program':'DOWN','measured_system':'VISION','text':'Measure 1 vision circle N1.'}]
    assert _required_marker_count(pages)==3


def test_bounded_display_refinement_is_recomputed_for_each_fai(tmp_path, monkeypatch):
    store=Store(tmp_path)
    second=dict(store.item,number=2,key='msop:2')
    store.items=lambda project:[store.item,second]
    calls=[]
    original=automatic_candidates._fit_bounded_common_translation
    def observe(rows,*args,**kwargs):
        calls.append(tuple(row['name'] for row in rows))
        return original(rows,*args,**kwargs)
    monkeypatch.setattr(automatic_candidates,'_fit_bounded_common_translation',observe)
    report=build_candidates(store,'p')
    assert [c['fai'] for c in report['candidates']]==[1,2]
    assert len(calls)==2


def test_invalid_nearest_face_does_not_hide_valid_surface_in_same_bound(tmp_path):
    store=Store(tmp_path)
    original=store.mesh('p')
    bad=dict(original['faces'][0],id='bad',occt_face_valid=False)
    good=dict(original['faces'][0],id='good')
    store.mesh=lambda pid:dict(original,faces=[bad,good])
    report=build_candidates(store,'p')
    assert len(report['candidates'])==1
    assert {a['face_id'] for a in report['candidates'][0]['anchors']}=={'good'}


def test_alternate_hypothesis_preserves_entire_group_and_requires_model_review(tmp_path):
    store=Store(tmp_path)
    store.page['text']=store.page['text'].replace('1 1 0','1 1 .2').replace('2 1 0','2 1 .2').replace('1 2 0','1 2 .2')
    path=tmp_path/'automatic-learning/registration.json'
    receipt=json.loads(path.read_text())
    receipt['frames'][0]['triangle_refinement']['best']['translation_mm']=[0,0,-.2]
    directory=path.parent/'registration-alternatives';directory.mkdir()
    (directory/'alternative.json').write_text(json.dumps(receipt))
    report=build_candidates(store,'p')
    assert len(report['candidates'])==1 and not store.annotations
    assert len(report['candidates'][0]['anchors'])==3
    assert report['candidates'][0]['evidence']['registration_receipt_sha256']
    receipt['cad_sha256']='stale'
    (directory/'alternative.json').write_text(json.dumps(receipt))
    still_valid = build_candidates(store,'p')
    assert len(still_valid['candidates']) == 1
    assert still_valid['candidates'][0]['evidence']['registration_receipt_sha256']


def test_stale_registration_algorithm_alternative_is_ignored(tmp_path):
    store=Store(tmp_path)
    path=tmp_path/'automatic-learning/registration.json'
    primary=json.loads(path.read_text())
    primary['frames'][0]['triangle_refinement']['best']['translation_mm']=[0,0,.09]
    path.write_text(json.dumps(primary))
    alternative=copy.deepcopy(primary)
    alternative['frames'][0]['triangle_refinement']['best']['translation_mm']=[0,0,0]
    alternative['algorithm_sha256']='stale-registration-code'
    directory=path.parent/'registration-alternatives';directory.mkdir()
    alternate_path=directory/'stale.json'
    alternate_path.write_text(json.dumps(alternative))

    result=build_candidates(store,'p')
    primary_digest=__import__('hashlib').sha256(path.read_bytes()).hexdigest()
    assert len(result['candidates'])==1
    assert result['candidates'][0]['evidence']['registration_receipt_sha256']==primary_digest


def test_unmarked_item_is_written_only_after_local_model_accepts_full_group(tmp_path,monkeypatch):
    monkeypatch.setattr("oracle_viewer.local_matching._review_visual_entity_group",lambda *a,**k:{"status":"confirmed"})
    store=Store(tmp_path);report=build_candidates(store,'p')
    class Model:
        loaded=True;bundle=object();openai_base_url='http://127.0.0.1:8081';openai_model='local-strata'
        _lock=threading.Lock()
        def info(self):return dict(inference_backend='strata-native')
        def _generate_once(self,messages,**kwargs):
            assert 'source_sections_preserved' in messages[0]['attachments'][0]['parsed_text']
            return json.dumps(dict(decision='AUTO_MARK',fai='FAI 1',face_ids=['f'],confidence='high',
                                   automatic_mark=False,evidence='All source points retained')),False,50
    result=mark_geometry_candidates(store,Model(),None,'p',generated_candidates=report['candidates'])
    assert result['counts']['accepted']==1 and len(store.annotations)==1
    args=store.annotations[0]
    assert len(args[2])==3 and args[4]=='auto_marked' and args[7]=='local_model_candidate' and args[8]=='low'


def test_legacy_manual_binding_does_not_veto_local_model_marking(tmp_path,monkeypatch):
    monkeypatch.setattr("oracle_viewer.local_matching._review_visual_entity_group",lambda *a,**k:{"status":"confirmed"})
    store=Store(tmp_path)
    store.item['binding']=dict(face_ids=['f'])
    store.item['annotation']=dict(origin='manual',status='reviewed',anchors=[dict(face_id='f',position=[1,1,0])])
    report=build_candidates(store,'p')
    assert len(report['candidates'])==1
    class Model:
        loaded=True;bundle=object();openai_base_url='http://127.0.0.1:8081';openai_model='local-strata'
        _lock=threading.Lock()
        def info(self):return dict(inference_backend='strata-native')
        def _generate_once(self,messages,**kwargs):
            return json.dumps(dict(decision='AUTO_MARK',fai='FAI 1',face_ids=['f'],confidence='medium',evidence='complete MSOP point group')),False,40
    result=mark_geometry_candidates(store,Model(),None,'p',generated_candidates=report['candidates'])
    assert result['counts']['accepted']==1
    assert store.annotations[-1][4]=='auto_marked' and store.annotations[-1][7]=='local_model_candidate'


@pytest.mark.parametrize('endpoint,expected', [
    ('http://127.0.0.1:8081/v1', True),
    ('http://localhost:8081/v1', True),
    ('http://[::1]:8081/v1', True),
    ('https://inference.example/v1', False),
    ('https://192.168.1.20/v1', False),
])
def test_fai_inference_requires_a_loopback_local_model(endpoint, expected):
    model=SimpleNamespace(openai_base_url=endpoint,info=lambda:{'inference_backend':'strata-native'})
    assert _is_local_inference_model(model) is expected


def test_fai_transformers_model_is_local_without_http_endpoint():
    model=SimpleNamespace(openai_base_url='',info=lambda:{'inference_backend':'transformers-peft'})
    assert _is_local_inference_model(model)


def test_marker_writer_refuses_nonlocal_inference_before_loading_adapter(tmp_path):
    model=SimpleNamespace(loaded=True,bundle=object(),openai_base_url='https://inference.example/v1',
                          info=lambda:{'inference_backend':'hosted'})
    with pytest.raises(RuntimeError,match='只允许本机本地模型'):
        mark_geometry_candidates(None,model,tmp_path,'p',generated_candidates=[])


def test_model_abstention_removes_unaccepted_geometry_preview(tmp_path):
    store=Store(tmp_path);report=build_candidates(store,'p')
    class Model:
        loaded=True;bundle=object();openai_base_url='http://127.0.0.1:8081';openai_model='local-strata'
        _lock=threading.Lock()
        def info(self):return dict(inference_backend='strata-native')
        def _generate_once(self,messages,**kwargs):
            return json.dumps(dict(decision='ABSTAIN',fai=1,face_ids=[],confidence='low',evidence='feature mismatch')),False,30
    result=mark_geometry_candidates(store,Model(),None,'p',generated_candidates=report['candidates'])
    assert result['counts']['abstain']==1
    assert store.removed==[1] and not store.annotations


def test_exact_fai_label_normalization_cannot_accept_wrong_scope():
    assert _fai_number('FAI 0001')==1 and _fai_number(1)==1
    assert _fai_number('FAI 2')!=1
    assert _fai_number(True) is None and _fai_number('FAI 1 and FAI 2') is None


def test_bounded_tangent_refinement_rejects_without_section_improvement():
    rows = [dict(name=f'PNT_{i}', point_key=f'p{i}', xyz=p) for i, p in enumerate(
        ([1, 1, 0], [4, 1, 0], [1, 4, 0]), 1)]
    planes = {row['point_key']: dict(normal_source=[0, 0, 1], offset_source_mm=0)
              for row in rows}
    triangles = np.asarray([[[0, 0, 0], [10, 0, 0], [0, 10, 0]]], float)
    translation = np.asarray([.2, .1, 0])
    result, evidence = _fit_bounded_tangent_shift(
        rows, np.eye(3), translation, planes, SurfaceIndex(triangles), triangles)
    assert evidence is None
    assert np.allclose(result, translation)


def test_generated_candidate_scope_uses_resolved_source_page_evidence():
    old = dict(automatic_xyz_hypothesis=True, evidence={
        'all_source_points_retained': True, 'source_sections_preserved': True,
        'point_correspondences': [dict(point_key='["N3_PNT1","UP","ALI_ABD"]',
            source_pages=[73], program='UP', coordinate_system='ALI_ABD', source_xyz=[7.61, 64.81, .74])]
    })
    assert _source_scope_conflicts(old, {'methods': []}, [], {'features': []}) == []
    old['evidence']['point_correspondences'][0]['point_key'] = '["N3_PNT1",null,"unresolved"]'
    assert _source_scope_conflicts(old, {'methods': []}, [], {'features': []})


def test_local_model_prompt_separates_fai_residuals_from_frame_wide_residuals():
    old=dict(automatic_xyz_hypothesis=True,origin='geometry_candidate',anchors=[
        dict(face_id='f',position=[1,1,0])],note='frame-wide diagnostic',evidence={
        'all_source_points_retained':True,'source_sections_preserved':True,
        'max_surface_anchor_section_deviation_mm':.02,
        'coordinate_page_resolutions':[dict(selected_page=52,program='UP',coordinate_system='ALI_ABD',
            page_group_coverage=.8,resolved_point_names=['PNT_A'],source_variants_preserved=True)],
        'point_correspondences':[dict(point_key='a',source_pages=[7],program='UP',coordinate_system='A',
            source_xyz=[1,1,0],residual_mm=.0635,plane_residual_mm=.01)],
        'frame_hypotheses':[dict(program='UP',coordinate_system='A',global_status='surface_residual_rejected',
                                 global_max_residual_mm=2.22)]})
    question,context,_,_,_=_prompt_data(
        {'drawing_no':'example','revision':'A'},
        {'number':1,'features':[],'headings':[],'specifications':[],'warnings':[]},
        {'methods':[dict(page=7,heading='FAI 1',text='Measure point A')],
         'item':{'features':[]}},old,{'f':dict(surface_type='plane',center=[1,1,0])},[])
    proposal=json.loads(context.split('Geometry proposal and provenance: ',1)[1].split('\n\n[MSOP',1)[0])
    audit=proposal['fai_scoped_numeric_audit']
    assert audit['all_point_distances_within_gate'] is True
    assert audit['max_point_to_CAD_surface_distance_mm']==.0635
    assert proposal['frame_level_registration_diagnostic'][0]['frame_wide_max_residual_mm']==2.22
    assert proposal['coordinate_page_resolution_audit'][0]['selected_page']==52
    assert '不得将frame_wide_max_residual_mm套用为本FAI点残差' in question
    assert '所选页的点组确实与本FAI特征对应' in question


def test_stale_seed_prompt_requires_fresh_model_decision_and_caps_display_confidence():
    old=dict(candidate_kind='same_source_stale_seed',origin='same_source_stale_seed',
        anchors=[dict(face_id='f',position=[1,2,0])],note='same-source automatic seed',
        seed_evidence=dict(source_hashes_verified=True,previous_status='auto_marked',
                           metrology_validated=False))
    question,context,candidate_ids,_,_=_prompt_data(
        {'drawing_no':'example','revision':'A'},
        {'number':9,'features':[],'headings':[],'specifications':[],'warnings':[]},
        {'methods':[dict(page=12,heading='FAI 9',text='Measure feature F.')],
         'item':{'features':[]}},old,{'f':dict(surface_type='plane',center=[1,2,0])},[])
    proposal=json.loads(context.split('Geometry proposal and provenance: ',1)[1].split('\n\n[MSOP',1)[0])
    assert candidate_ids==['f']
    assert proposal['same_source_stale_seed_audit']['source_hashes_verified'] is True
    assert '不得继承旧决策' in question
    assert '显示置信度必须为low' in question
    assert 'fai_scoped_numeric_audit' not in proposal


def test_stale_xyz_seed_reenters_review_only_after_current_source_and_geometry_audit():
    mesh={'source_sha256':'cad','faces':[{'id':'f','positions':[
        -2,-2,0, 2,-2,0, 2,2,0, -2,2,0], 'indices':[0,1,2,0,2,3]}]}
    correspondences=[dict(name=name,program='UP',coordinate_system='ALI_A',
        source_xyz=xyz,position=position,residual_mm=.05,plane_residual_mm=.01)
        for name,xyz,position in (
            ('PNT_A',[0,0,0],[0,0,0]),('PNT_B',[1,0,0],[1,0,0]),
            ('PNT_C',[0,1,0],[0,1,0]))]
    evidence={'point_correspondences':correspondences,'all_source_points_retained':True,
        'source_sections_preserved':True,'max_surface_anchor_section_deviation_mm':.08}
    note='previous local-model mark\n'+json.dumps(evidence)
    item={'number':1,'document_id':'msop','annotation':dict(status='stale',origin='local_model_candidate',
        stale_reason='automatic_registration_changed',stale_previous_status='auto_marked',
        source_sha256='cad',document_id='msop',document_sha256='msop',anchors=[
            dict(face_id='f',position=row['position']) for row in correspondences],note=note)}
    rows=[dict(name=name,program='UP',coordinate_system='ALI_A',xyz=xyz)
          for name,xyz in (('PNT_A',[0,0,0]),('PNT_B',[1,0,0]),('PNT_C',[0,1,0]))]
    seed=_same_source_stale_seed(item,mesh,{'sha256':'msop'}, {'f':mesh['faces'][0]},set(),
        source_rows=rows,required_point_count=3)
    assert seed['automatic_xyz_hypothesis'] is True
    assert seed['seed_evidence']['freshly_validated'] is True
    assert seed['evidence']['point_correspondences']==correspondences

    rows[0]['xyz']=[.2,0,0]
    mismatched=_same_source_stale_seed(item,mesh,{'sha256':'msop'},
        {'f':mesh['faces'][0]},set(),source_rows=rows,required_point_count=3)
    assert mismatched is not None
    assert mismatched.get('automatic_xyz_hypothesis') is not True


def test_local_model_prompt_identifies_explicit_measure_points_as_authoritative():
    page=dict(page=7,program='DOWN',measured_system='FRAME',
        text='Measure 2 vector points DATUM_E01,E02 and construct a line DATUM_TOP.\n'
             'X, Y, Z (mm) 1 2 3 DATUM_E01\n'
             'X, Y, Z (mm) 4 5 6 DATUM_E02\n'
             'X, Y, Z (mm) 8 9 10 DATUM_E99')
    old=dict(automatic_xyz_hypothesis=True,origin='geometry_candidate',anchors=[
        dict(face_id='f',position=[1,2,3])],evidence={'point_correspondences':[],
        'all_source_points_retained':True,'source_sections_preserved':True})
    _,context,_,_,_=_prompt_data(
        {'drawing_no':'example','revision':'A'},
        {'number':1,'features':['DATUM_E99'],'headings':[],'specifications':[],'warnings':[]},
        {'methods':[page],'item':{'features':['DATUM_E99']}},old,
        {'f':dict(surface_type='plane',center=[1,2,3])},[page])
    proposal=json.loads(context.split('Geometry proposal and provenance: ',1)[1].split('\n\n[MSOP',1)[0])
    assert proposal['explicit_measured_point_names']==['DATUM_E01','DATUM_E02']
    assert 'DATUM_E99' not in proposal['explicit_measured_point_names']
    assert 'not measured by this FAI' in context


def test_local_model_prompt_exposes_specific_cross_page_feature_support_for_constructed_circle():
    from oracle_viewer.candidate_mining import extract_named_points
    source=dict(page=65,program='UP',measured_system='ALI_ABD',text='\n'.join(
        f'X, Y, Z (mm) {x} {y} 0.99 N8_PNT{i}'
        for i,(x,y) in enumerate(((-3.61,63.1),(-4.21,62.5),(-3.61,61.9),(-3.01,62.5)),1)))
    method=dict(page=33,program='UP',measured_system='ALI_ABD',
        text='Measure 1 vision circle CIR_N8_VISION.\nDimension Y location 62.58 ± 0.12')
    pages=[source,method]
    points=extract_named_points(pages)
    old=dict(automatic_xyz_hypothesis=True,origin='geometry_candidate',anchors=[
        dict(face_id='f',position=[-3.61,62.5,1.0])],evidence={
        'all_source_points_retained':True,'source_sections_preserved':True,
        'max_surface_anchor_section_deviation_mm':.02,
        'point_correspondences':[dict(name=name,point_key=name,source_xyz=row['xyz'],source_pages=[65],
            program='UP',coordinate_system='ALI_ABD',residual_mm=.01,plane_residual_mm=0.0)
            for name,row in points.items()]})
    question,context,_,_,_=_prompt_data(
        {'drawing_no':'example','revision':'A'},
        {'number':23,'features':['CIR_N8_VISION'],'headings':[],'specifications':[],'warnings':[]},
        {'methods':[method],'item':{'features':['CIR_N8_VISION']}},old,
        {'f':dict(surface_type='plane',center=[-3.61,62.5,1.0])},pages)
    proposal=json.loads(context.split('Geometry proposal and provenance: ',1)[1].split('\n\n[MSOP',1)[0])
    links={row['name']:row['semantic_feature_links'] for row in proposal['cross_page_point_references']}
    assert all(links[name]==['CIR_N8_VISION'] for name in points)
    assert 'vision circle' in context
    assert '不能仅因方法页未逐一列出支撑点就自动弃答' in question


def test_xyz_prompt_compacts_duplicate_provenance_without_dropping_the_full_point_group():
    correspondences=[]
    for i in range(33):
        xyz=[float(i),float(i%4),0.0]
        correspondences.append(dict(name=f'DATUMA_PNT_{i:02d}',point_key=f'p{i}',program='UP',
            coordinate_system='ALI_ABD',source_pages=[80,81],source_xyz=xyz,
            transformed_xyz=[xyz[0],xyz[1],2.0],position=[xyz[0],xyz[1],2.01],
            face_id='face-a',residual_mm=.01,plane_residual_mm=.01,
            coordinate_variant_span_mm=(.01 if i==1 else 0.0),
            source_coordinate_variants=[dict(page=page,xyz=([xyz[0]+(.01 if i==1 and page==83 else 0),xyz[1],xyz[2]]),
                program='UP',coordinate_system='ALI_ABD') for page in range(80,84)]))
    old=dict(automatic_xyz_hypothesis=True,origin='geometry_candidate',
        anchors=[dict(face_id='face-a',position=[float(i),float(i%4),2.01]) for i in range(33)],
        note='duplicate serialized evidence ' * 1000,evidence={
            'all_source_points_retained':True,'source_sections_preserved':True,
            'max_surface_anchor_section_deviation_mm':.02,'point_correspondences':correspondences,
            'frame_hypotheses':[],'coordinate_page_resolutions':[]})
    question,context,_,_,_=_prompt_data(
        {'drawing_no':'generic','revision':'A'},
        {'number':7,'features':[],'headings':[],'specifications':[],'warnings':[]},
        {'methods':[],'item':{'features':[]}},old,
        {'face-a':dict(surface_type='plane',center=[16,1,2])},[])
    assert len(question)+len(context)<48000
    proposal=json.loads(context.split('Geometry proposal and provenance: ',1)[1].split('\n\n[MSOP',1)[0])
    rows=proposal['cross_page_point_references']
    assert len(rows)==33
    assert {row['name'] for row in rows}=={row['name'] for row in correspondences}
    assert 'source_coordinate_variants' not in rows[0]
    assert len(rows[1]['source_coordinate_variants'])==4
    assert len(proposal['note'])<200


def test_bounded_common_translation_preserves_all_source_point_vectors():
    source = np.asarray([[1, 1, 0], [4, 1, 0], [1, 4, 0]], float)
    rows = [dict(name=f'PNT_{i}', xyz=p.tolist()) for i, p in enumerate(source, 1)]
    triangles = np.asarray([[[0, 0, 0], [10, 0, 0], [0, 10, 0]]], float)
    index = SurfaceIndex(triangles)
    translation, evidence = _fit_bounded_common_translation(
        rows, np.eye(3), [0, 0, .08], index, max_residual_mm=.1)
    assert evidence and evidence['max_residual_after_mm'] < 1e-8
    assert evidence['preserves_all_inter_point_vectors'] is True
    assert np.allclose(translation, [0, 0, 0], atol=1e-8)
    too_far, rejected = _fit_bounded_common_translation(
        rows, np.eye(3), [0, 0, .5], index, max_residual_mm=.1)
    assert rejected is None and np.allclose(too_far, [0, 0, .5])


def test_bounded_minimax_translation_recovers_full_group_after_median_fit_stalls():
    source=np.asarray([[1,1,0],[4,1,0],[1,4,0]],float)
    rows=[dict(name=f'PNT_{i}',xyz=p.tolist()) for i,p in enumerate(source,1)]
    triangles=np.asarray([[[0,0,0],[10,0,0],[0,10,0]]],float)
    index=SurfaceIndex(triangles)
    moved,evidence=_fit_bounded_minimax_translation(rows,np.eye(3),[0,0,.245],index,
                                                     max_residual_mm=.1,max_shift_mm=.3)
    assert evidence and evidence['max_residual_after_mm']<1e-3
    assert evidence['preserves_all_inter_point_vectors'] is True
    assert evidence['optimizer']=='approximate_surface_search_then_exact_all_point_validation'
    assert abs(moved[2])<1e-3
    too_far,rejected=_fit_bounded_minimax_translation(rows,np.eye(3),[0,0,.5],index,
                                                       max_residual_mm=.1,max_shift_mm=.3)
    assert rejected is None and np.allclose(too_far,[0,0,.5])
