from pathlib import Path
import json
from copy import deepcopy
import numpy as np
import pytest
from oracle_viewer.registration import fit_rigid,frame_proposals
from oracle_viewer.engineering import coverage_status,audit_task,audit_library
from oracle_viewer.adaptation import export_adaptation,training_preflight
from oracle_viewer.physical import triangles_hit_cylinder
from oracle_viewer.msop_coordinates import _coordinate_rows


def test_rigid_fit_with_independent_nonplanar_support_and_outlier():
    a=np.array([[0,0,0],[3,0,0],[0,4,0],[0,0,5],[2,2,3],[4,1,2]],float)
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    b=a@rotation.T+[3,5,7]
    fit=fit_rigid(a,b,max_residual_mm=.01)
    assert fit['status']=='geometrically_supported_proposal'
    assert np.allclose(fit['rotation'],rotation)
    assert fit['held_out_max_mm']<1e-10 and not fit['source_to_cad_registered']
    b[-1,2]+=2
    assert fit_rigid(a,b,max_residual_mm=.01)['status']=='residual_rejected'


def test_planar_fit_cannot_register_layer_or_handedness():
    p=[[0,0,0],[1,0,0],[0,1,0],[1,1,0]]
    assert fit_rigid(p,p)['status']=='underdetermined'
    with pytest.raises(ValueError):fit_rigid(p,[[0,0,0]])


def test_mirrored_frame_not_accepted_as_physical_rotation():
    a=np.array([[0,0,0],[3,0,0],[0,4,0],[0,0,5],[2,2,3],[4,1,2]],float)
    b=a*[-1,1,1]
    assert fit_rigid(a,b)['status']=='residual_rejected'


def test_duplicate_names_are_not_independent_registration_evidence():
    point=dict(name='P1',source_xyz=[0,0,0],cad_position_mm=[0,0,0],face_id='f',coordinate_system='ALI_X')
    tasks=[dict(number=i,msop_coordinate_map=dict(mapped_points=[point])) for i in range(5)]
    fit=frame_proposals(tasks)[0]
    assert len(fit['correspondences'])==1 and fit['status']=='insufficient_correspondences'
    tasks[0]['msop_coordinate_map']['mapped_points']=[dict(point,cad_position_mm=[0,0,1])]
    assert frame_proposals(tasks)[0]['conflicts']


def test_triangle_crosses_ring_wall_without_any_vertex_inside():
    ring=dict(origin=np.zeros(3),basis=np.eye(3),height=2,outer=2,inner=1)
    assert triangles_hit_cylinder(np.array([[-4,-4,1],[4,-4,1],[0,4,1]]),[0,1,2],ring)
    assert not triangles_hit_cylinder(np.array([[0,0,1],[.2,0,1],[0,.2,1]]),[0,1,2],ring)
    assert not triangles_hit_cylinder(np.array([[-4,-4,3],[4,-4,3],[0,4,3]]),[0,1,2],ring)
    # Vertical degenerate XY projection through a wall still intersects.
    assert triangles_hit_cylinder(np.array([[1.5,0,-1],[1.5,0,3],[1.5,.1,1]]),[0,1,2],ring)


def test_coordinate_evidence_survives_without_calibration(tmp_path):
    from oracle_viewer.store import ViewerStore
    from oracle_viewer.workflow import coordinate_tasks
    store=ViewerStore(tmp_path);p=store.create('Other','other','C')
    p['documents']=[dict(id='m',role='msop',method_pages=[dict(page=1,text='X, Y, Z (mm) 1 2 3 PNT_A',measured_system='ALI_A')])]
    tasks=[dict(number=1,anchors=[],methods=p['documents'][0]['method_pages'],dependencies=[],features=[])]
    result=coordinate_tasks(store,p,tasks)[0]['msop_coordinate_map']
    assert result['source_points'][0]['xyz']==[1,2,3]
    assert result['status']=='coordinates_preserved_unmapped'


def test_local_table_does_not_drop_cross_page_references():
    pages=[dict(page=1,text='X, Y, Z (mm) 1 2 3 PNT_B',measured_system='ALI_A'),
           dict(page=2,text='X, Y, Z (mm) 4 5 6 PNT_A\nMeasure points PNT_B PNT_A',measured_system='ALI_A')]
    rows=_coordinate_rows([pages[1]],[],['PNT_B','PNT_A'],pages)
    assert {p['name'] for p in rows}=={'PNT_A','PNT_B'}


def test_coverage_never_converts_proposals_into_accepted_measurements():
    tasks=[dict(number=1,marked=True,location_status='pending_review'),dict(number=2,marked=False)]
    run=dict(status='complete',stale=False,result=dict(selected_fai=[1],method_routes=[dict(number=1)],heads=[{}],uncovered_fai=[]))
    c=coverage_status(tasks,run)
    assert c['method_proposed_fai']==1 and c['optical_acquisition_fai']==1 and c['metrology_accepted_fai']==0
    assert c['unmarked_numbers']==[2]


def test_marker_off_face_and_source_conflict_require_relocation():
    faces={'f':dict(positions=[0,0,0,2,0,0,0,2,0],indices=[0,1,2])}
    t=dict(number=1,anchors=[dict(face_id='f',position=[.5,.5,1])],methods=[],dependencies=[])
    a=audit_task(t,faces,['source conflict'])
    assert a['evidence_grade']=='conflicting' and a['anchors'][0]['status']=='needs_relocation'
    assert not a['confidence_is_calibrated']


def test_training_preflight_blocks_incompatible_inference_only_model():
    p=training_preflight(dict(model_alias='native-quantized',inference_backend='strata-native'))
    assert p['status']=='blocked_prerequisites' and len(p['reasons'])==2
    assert not p['weights_changed'] and not p['fallback_to_other_base']


def test_training_quarantines_unreviewed_and_splits_by_project(tmp_path):
    class Store:
        def get(self,pid):return dict(id=pid,drawing_no=pid)
        def items(self,p):return [dict(number=1,annotation=dict(status='reviewed' if p['id']!='C' else 'pending_review',anchors=[dict(face_id='f')]))]
    reports=[dict(project_id=pid,source_hashes={},fai=[dict(fai=1,pages=[1],contract={},anchors=[dict(status='on_display_surface')],model_confidence='low')]) for pid in ['A','B','C']]
    manifest=export_adaptation(Store(),reports,tmp_path,dict(inference_backend='strata-native'))
    assert manifest['train_groups']==['A'] and manifest['test_groups']==['B'] and manifest['quarantine_count']==1
    assert set(manifest['train_groups']).isdisjoint(manifest['test_groups'])
    readiness={r['drawing_no']:r for r in manifest['project_readiness']}
    assert readiness['C']['quarantine_reasons']=={'annotation_not_reviewed':1}
    assert readiness['A']['eligible_labels']==1


def test_reviewed_conflicting_source_is_not_training_ground_truth(tmp_path):
    class Store:
        def get(self,pid):return dict(id=pid,drawing_no=pid)
        def items(self,p):return [dict(number=1,annotation=dict(status='reviewed',anchors=[dict(face_id='f')]))]
    reports=[dict(project_id='A',source_hashes={},fai=[dict(fai=1,pages=[1],contract={},
        anchors=[dict(status='on_display_surface')],model_confidence='high',evidence_grade='conflicting')])]
    manifest=export_adaptation(Store(),reports,tmp_path,{})
    assert manifest['train_count']==0 and manifest['quarantine_count']==1
    row=json.loads((tmp_path/'quarantine.jsonl').read_text())
    assert row['quarantine_reasons']==['source_evidence_conflicting']


def test_library_hash_mismatch_is_not_accepted(tmp_path):
    (tmp_path/'a.pdf').write_bytes(b'%PDF bad')
    cat=dict(items=[dict(id='C',kind='camera',spec={},assets=[dict(file='a.pdf',kind='pdf',sha256='bad')])])
    audit=audit_library(cat,tmp_path)
    assert audit['invalid_assets']==['a.pdf'] and audit['rows'][0]['missing_layout_fields']


def test_cancelled_workflow_cannot_be_overwritten_by_worker(tmp_path):
    from oracle_viewer.workflow import WorkflowService,WorkflowCancelled
    from oracle_viewer.store import ViewerStore
    store=ViewerStore(tmp_path);p=store.create('test');service=WorkflowService(store,None,None)
    r=dict(id='r',status='running',progress=dict(message='running'))
    service.save(p['id'],r);service.cancel(p['id'])
    with pytest.raises(WorkflowCancelled):service.save(p['id'],deepcopy(r))
    r['status']='cancelled';service.save(p['id'],r)
    assert json.loads(service.path(p['id']).read_text())['status']=='cancelled'


def test_controller_ranges_and_missing_specs():
    from oracle_viewer.hardware_checks import controller_check
    light=dict(id='L',spec=dict(voltage_v=36,current_a=2,power_w=72))
    controller=dict(id='C',spec=dict(output_voltage_v=[24,48],output_current_a=4,output_power_w=100))
    assert controller_check(light,controller)['status']=='numeric_ranges_supported'
    light['spec']['current_a']=5
    assert 'channel_current_exceeded' in controller_check(light,controller)['violations']
    assert controller_check(dict(id='L',spec={}),controller)['status']=='incomplete_specs'


def test_ring_layout_fields_not_box_fields():
    item=dict(id='R',kind='light',spec=dict(type='ring',body_outer_mm=80,body_inner_mm=40,body_height_mm=20,light_distance_mm=[20,80]))
    assert audit_library(dict(items=[item]))['rows'][0]['missing_layout_fields']==[]


def test_archive_restore_is_a_checked_fork(tmp_path):
    from oracle_viewer.store import ViewerStore
    from oracle_viewer.backup import archive_project,restore_project
    store=ViewerStore(tmp_path/'store');p=store.create('Example','1','A')
    asset=store._dir(p['id'])/'assets';asset.mkdir();(asset/'x.pdf').write_bytes(b'original')
    p['documents']=[dict(path=str(asset/'x.pdf'),role='drawing')];store._write(p)
    target=archive_project(store,p['id'],tmp_path/'x.zip');new=restore_project(store,target)
    assert new['id']!=p['id']
    restored=store.get(new['id']);assert Path(restored['documents'][0]['path']).read_bytes()==b'original'
    assert store.get(p['id'])['id']==p['id']


def test_restarting_does_not_leave_workflow_running(tmp_path):
    from oracle_viewer.store import ViewerStore
    store=ViewerStore(tmp_path);p=store.create('Example','1','A')
    path=store._dir(p['id'])/'aoi-workflow.json'
    path.write_text(json.dumps(dict(status='running',result=dict(heads=[1]),progress={})))
    store.recover_tasks();r=json.loads(path.read_text())
    assert r['status']=='failed' and r['result']['heads']==[1]


def test_motion_screens_between_clear_endpoints():
    from oracle_viewer.motion import screen_motion
    def head(x):
        return dict(scene=dict(normal=[0,0,1],basis_x=[1,0,0],basis_y=[0,1,0],lens_front_mm=[x,0,10],camera_flange_mm=[x,0,20],lens_length_mm=10,lens_diameter_mm=2,camera_body_mm=[2,2,4],light_direction=[0,0,1],light_face_mm=[x,0,-2],light_shape='panel',light_body_mm=[2,2,1]))
    heads={'A':head(-10),'B':head(10)};c=[dict(id='C',head_ids=['A','B'],home_head_id='A',motion_mode='move_camera')]
    mesh=dict(faces=[dict(id='wall',positions=[0,-20,5,0,20,5,0,0,30],indices=[0,1,2])])
    r=screen_motion(c,heads,mesh,dict(minimum_clearance_mm=0))
    assert r['potential_interference_count']==1 and not r['path_validated']
    c[0]['motion_mode']='move_part';r=screen_motion(c,heads,mesh,{})
    assert r['unresolved_unit_ids']==['C'] and r['cameras'][0]['legs']==[]


def test_native_images_use_explicit_local_visual_model(tmp_path,monkeypatch):
    import urllib.request
    from talker.model import TalkerModel
    model=object.__new__(TalkerModel);model.openai_model='native-text';model.openai_base_url='http://127.0.0.1:8081';model.openai_backend_label='strata-native'
    monkeypatch.setattr('talker.model._strata_native_vision_ready',lambda *_:False)
    monkeypatch.setattr('talker.model._local_vision_model_ready',lambda base,name:bool(base and name))
    image=tmp_path/'page.png';image.write_bytes(b'image')
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(dict(choices=[dict(message=dict(content='seen'),finish_reason='stop')],usage=dict(completion_tokens=2))).encode()
    captured=[]
    monkeypatch.setattr(urllib.request,'urlopen',lambda request,**kwargs:captured.append(request) or Response())
    monkeypatch.setenv('ORACLE_LOCAL_VISION_BASE_URL','http://127.0.0.1:7871');monkeypatch.setenv('ORACLE_LOCAL_VISION_MODEL','local-vision')
    assert model._generate_openai('read',[str(image)],20,'',temperature=.2)==('seen',False,2)
    assert captured[0].full_url=='http://127.0.0.1:7871/v1/chat/completions'
    assert json.loads(captured[0].data)['model']=='local-vision'
    assert json.loads(captured[0].data)['temperature']==.2
    monkeypatch.delenv('ORACLE_LOCAL_VISION_BASE_URL')
    with pytest.raises(RuntimeError,match='未加载视觉编码器'):model._generate_openai('read',[str(image)],20,'')


def test_source_loop_order_preserves_outer_and_hole():
    from oracle_viewer.parasolid_worker import ordered_uv_rings
    hole=[[1,1],[1,2],[2,2],[2,1]];outer=[[0,0],[4,0],[4,4],[0,4]]
    uv,ends=ordered_uv_rings(hole+outer,[4,8])
    assert uv==outer+hole and ends==[4,8]


def test_restarting_recovers_marking_and_presentation_jobs(tmp_path):
    from oracle_viewer.store import ViewerStore
    store=ViewerStore(tmp_path);p=store.create('Example','1','A')
    store.update_local_model_marking(p['id'],status='running',items=[dict(fai=1,result='accepted')])
    path=store._dir(p['id'])/'assembly-presentations'/'job'/'job.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(status='running',model_attempt=dict(text='retained'))))
    store.recover_tasks();assert store.get(p['id'])['local_model_marking']['status']=='failed'
    assert store.get(p['id'])['local_model_marking']['items']==[dict(fai=1,result='accepted')]
    job=json.loads(path.read_text());assert job['status']=='failed' and job['model_attempt']['text']=='retained'


def test_model_evidence_column_encoding_is_lossless():
    from oracle_viewer.integrated_schemes import table_evidence,expand_evidence
    source=dict(all_fai=[dict(number=i,points=[dict(x=1.23456789+i,page=j,normal=None,label='原文点位') for j in range(5)],requirements=['keep every source']) for i in range(80)])
    packed=table_evidence(source)
    assert expand_evidence(packed)==source
    assert len(json.dumps(packed))<len(json.dumps(source))


def test_layout_repairs_overlapping_ids_without_accepting_first_output():
    import threading
    from oracle_viewer.workflow import review_layout
    class Model:
        _lock=threading.RLock()
        prompts=[]
        def _generate_once(self,messages,**kwargs):
            self.prompts.append(messages[0]['content'])
            return json.dumps(dict(selected_head_ids=['H1'],alternate_head_ids=['H1'] if len(self.prompts)==1 else ['H2'],review='draft')),False,20
    model=Model();attempts=[]
    decision,heads,final=review_layout(model,'Select layouts','complete evidence',[dict(id='H1'),dict(id='H2')],lambda a:attempts.append(list(a)))
    assert decision['alternate_head_ids']==['H2'] and final['retry']==1
    assert 'disjoint' in model.prompts[1] and attempts[0][0]['validation_error']


def test_light_motion_detects_wall_missed_by_camera_sweep():
    from oracle_viewer.motion import screen_motion
    def head(x):
        return dict(scene=dict(normal=[0,0,1],basis_x=[1,0,0],basis_y=[0,1,0],lens_front_mm=[x,0,10],camera_flange_mm=[x,0,20],lens_length_mm=10,lens_diameter_mm=2,camera_body_mm=[2,2,4],light_direction=[0,0,1],light_face_mm=[x,0,-2],light_shape='panel',light_body_mm=[2,2,1]))
    heads={'A':head(-10),'B':head(10)}
    cameras=[dict(id='C',head_ids=['A','B'],home_head_id='A',motion_mode='move_optics')]
    lights=[dict(id='L',mode='existing_moving_light',head_ids=['A','B'],home_head_id='A',head=heads['A'])]
    mesh=dict(faces=[dict(id='low_wall',positions=[0,-20,-5,0,20,-5,0,0,0],indices=[0,1,2])])
    r=screen_motion(cameras,heads,mesh,dict(minimum_clearance_mm=0),lights)
    assert r['cameras'][0]['legs'][0]['conflicts']==[]
    assert r['lights'][0]['legs'][0]['conflicts']==[dict(object='workpiece',face_id='low_wall')]
    assert r['potential_interference_count']==1
    assert not r['simultaneous_motion_validated'] and not r['path_validated']


def test_camera_sweep_checks_independently_placed_static_light():
    from oracle_viewer.motion import screen_motion
    def head(x,z=-2):
        return dict(scene=dict(normal=[0,0,1],basis_x=[1,0,0],basis_y=[0,1,0],lens_front_mm=[x,0,10],camera_flange_mm=[x,0,20],lens_length_mm=10,lens_diameter_mm=2,camera_body_mm=[2,2,4],light_direction=[0,0,1],light_face_mm=[x,0,z],light_shape='panel',light_body_mm=[2,2,1]))
    heads={'A':head(-10),'B':head(10)}
    r=screen_motion([dict(id='C',head_ids=['A','B'],home_head_id='A',motion_mode='move_optics')],heads,{},dict(minimum_clearance_mm=0),[dict(id='PANEL',mode='merged_panel_requirement',head=head(0,15))])
    assert r['cameras'][0]['legs'][0]['conflicts']==[dict(object='PANEL')]
    assert r['static_light_ids']==['PANEL'] and r['lights']==[]
