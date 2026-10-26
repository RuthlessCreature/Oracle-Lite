import json
from pathlib import Path
from copy import deepcopy
import pytest
from oracle_viewer.workflow import library,optical_pair,solve_cover,plan,compile_tasks,validate_model_selection,fingerprint
from oracle_viewer.store import ViewerStore


def design(**kw):
 return dict(fai_numbers=[7,8],roi_px=[2400,2000],feature_radius_mm=3,max_sampling_um_px=10,normal_axis='+Z',backlight_gap_mm=20,
             _pair_ids=['acA2440-35um|TC23024'],_light_ids=['LT2BC048036-R-CO','LTRN024RD'])|kw

def task(number,position,**kw):
 return dict(number=number,strategy='2d_visual_circle',anchors=[{'position':position}],gaps=[])|kw

def test_default_sensor_is_not_covered_but_explicit_roi_is():
 cat=library();cam=cat['items'][0];lens=next(i for i in cat['items'] if i['id']=='TC23024')
 full=optical_pair(cam,lens,[2448,2048],10)
 assert not full['eligible'] and 'image circle' in full['reasons'][0]
 roi=optical_pair(cam,lens,[2400,2000],10)
 assert roi['eligible'] and roi['fov_mm']==pytest.approx([23.657142857,19.714285714])
 assert roi['sampling_um_px']==pytest.approx(9.857142857)
 broken=deepcopy(cam);broken['spec']['mount']='F'
 assert not optical_pair(broken,lens,[2400,2000],10)['eligible']


def test_cover_finds_global_small_universe_optimum_not_greedy():
 c=lambda id,nums:dict(id=id,fai_numbers=nums,optics={'sampling_um_px':1})
 candidates=[c('a',[1,2,3,4]),c('b',[1,2,5]),c('c',[3,4,6]),c('d',[5]),c('e',[6])]
 assert {c['id'] for c in solve_cover([1,2,3,4,5,6],candidates)}=={'b','c'}


def test_cover_prefers_larger_single_view_when_head_count_and_precision_are_valid():
 small=dict(id='small',fai_numbers=[1,2],optics={'sampling_um_px':2,'fov_mm':[20,15]})
 large=dict(id='large',fai_numbers=[1,2],optics={'sampling_um_px':9,'fov_mm':[60,45]})
 assert [c['id'] for c in solve_cover([1,2],[small,large])]==['large']


def test_geometry_spacing_depth_unknown_method_and_uncovered():
 tasks=[task(7,[0,0,0]),task(8,[0,9,0])]
 merged=plan(tasks,library(),design());assert len(merged['heads'])==1
 moved=plan([tasks[0],task(8,[0,40,0])],library(),design());assert len(moved['heads'])==2
 depth=plan([tasks[0],task(8,[0,9,5])],library(),design());assert len(depth['heads'])==2
 unsupported=plan([tasks[0],task(8,[0,9,0],strategy='needs_method_adapter')],library(),design())
 assert unsupported['uncovered_fai']==[] and not unsupported['ready_for_procurement']
 other=plan(tasks,library(),design(normal_axis='other'));assert not other['heads']
 with pytest.raises(ValueError):plan(tasks,library(),design(fai_numbers=[999]))


def test_model_ids_guard_no_rules_fallback():
 candidates=[{'id':'H3','fai_numbers':[7,8]}]
 d,h=validate_model_selection('{"selected_head_ids":["H3"],"review":"assumptions"}',candidates);assert h==candidates
 for raw in ['{"selected_head_ids":["unknown"],"review":"x"}','{"selected_head_ids":["H3","H3"],"review":"x"}','not JSON']:
  with pytest.raises((ValueError,json.JSONDecodeError)):validate_model_selection(raw,candidates)
 d,h=validate_model_selection('{"selected_head_ids":[],"review":"cannot confirm"}',candidates);assert h==[]


def test_general_project_parser_retains_method_chain_and_rejects_old_geometry(tmp_path):
 store=ViewerStore(tmp_path);data=store.create('Unseen project','different-part','B')
 method=dict(page=5,heading='FAI 7',fai_ids=[7],note_fai_ids=[],features=['CIR_CUSTOM_VISION'],probe='VISION',program='DOWN',measured_system='ALI_CUSTOM',dimension_system='ALI_CUSTOM',text='Measure 1 vision circle CIR_CUSTOM_VISION. Settings: Density: NORMAL Filter: Outlier: 2. Dimension Y Value',specification={'nominal':'13','upper':'+0.1','lower':'-0.1','unit':'mm'},settings=['Outlier: 2'],warnings=[])
 data['documents']=[dict(id='doc',role='msop',filename='custom.pdf',sha256='docsha',method_pages=[method],page_count=5)];data['model']={'source_sha256':'cadsha','status':'preview'}
 data['annotations']={'doc:7':dict(source_sha256='oldcad',document_sha256='docsha',status='reviewed',anchors=[{'position':[0,0,0]}])}
 tasks=compile_tasks(store,data);assert len(tasks)==1
 assert tasks[0]['methods'][0]['program']=='DOWN' and not tasks[0]['anchors']
 assert tasks[0]['status']=='blocked'
 f=fingerprint(data,library());data['documents'][0]['method_pages'][0]['text']+=' changed'
 assert fingerprint(data,library())!=f


def test_automatic_fai_location_does_not_require_a_human_review_step(tmp_path):
 store=ViewerStore(tmp_path);data=store.create('Unseen project','different-part','B')
 method=dict(page=5,heading='FAI 7',fai_ids=[7],note_fai_ids=[],features=['CIR_CUSTOM_VISION'],probe='VISION',program='DOWN',measured_system='ALI_CUSTOM',dimension_system='ALI_CUSTOM',text='Measure 1 vision circle CIR_CUSTOM_VISION. Dimension Y Value',specification={'nominal':'13','upper':'+0.1','lower':'-0.1','unit':'mm'},settings=[],warnings=[])
 data['documents']=[dict(id='doc',role='msop',filename='custom.pdf',sha256='docsha',method_pages=[method],page_count=5)]
 data['model']={'source_sha256':'cadsha','status':'preview','metrology_ready':False}
 data['annotations']={'doc:7':dict(source_sha256='cadsha',document_sha256='docsha',status='auto_marked',origin='local_model_candidate',confidence='low',anchors=[dict(face_id='f',position=[0,0,0])])}
 tasks=compile_tasks(store,data)
 assert tasks[0]['location_status']=='auto_marked'
 assert '标记位置未完成自动定位' not in tasks[0]['gaps']
 assert '尚未完成方法等效、基准配准与计量验证' in tasks[0]['gaps']


def test_library_downloads_have_verified_files_and_hashes():
 from oracle_viewer.workflow import CATALOG
 import hashlib
 cat=library();assert len(cat['items'])>=7
 for i in cat['items']:
  for a in i['assets']:
   blob=(CATALOG/'assets'/a['file']).read_bytes()
   assert hashlib.sha256(blob).hexdigest()==a['sha256']
   if a['kind']=='pdf':assert blob.startswith(b'%PDF')


def test_service_never_uses_solver_result_when_local_model_rejects(tmp_path):
 from oracle_viewer.workflow import WorkflowService
 import threading
 class Model:
  loaded=True;_lock=threading.Lock()
  def info(self):return dict(model_alias='local-test',inference_backend='strata-native')
  def _generate_once(self,*args,**kwargs):return '{"selected_head_ids":["invented"],"review":"bad"}',False,10
 class Pool:
  def submit(self,*args):pass
 store=ViewerStore(tmp_path);project=store.create('new','new','01')
 service=WorkflowService(store,Model(),Pool())
 # Persist a source-free abstention run to exercise the actual worker failure boundary.
 record=dict(id='run',project_id=project['id'],status='queued',task_snapshot=[],design=design(),result=dict(candidates=[],tool_recommendation_head_ids=[],blockers=[],heads=[],bom=[],selected_fai=[7,8]),input_fingerprint=fingerprint(project,library()))
 service.save(project['id'],record);service.review(project['id'],record)
 saved=service.get(project['id'])['run'];assert saved['status']=='failed'
 assert saved['result']['heads']==[]
 assert not any(b['kind']=='camera' for b in saved['result']['bom'])


def test_single_fai_design_also_requires_local_model(monkeypatch):
 from types import SimpleNamespace
 from oracle_viewer import assistant
 monkeypatch.setattr(assistant,'source_checked_circle_plan',lambda *a:'deterministic tool suggestion')
 monkeypatch.setattr(assistant,'is_design_request',lambda q:True)
 with pytest.raises(RuntimeError):assistant.answer_project(SimpleNamespace(loaded=False),'生成方案','source')


def test_lighting_candidates_are_not_forced_to_transmission():
 result=plan([task(7,[0,0,0]),task(8,[0,9,0])],library(),design())
 merged=[c for c in result['candidates']+result['rejected_candidates'] if c['fai_numbers']==[7,8]]
 assert {c['lighting_strategy'] for c in merged}=={'transmitted_backlight','oblique_panel_reflection'}
 for c in merged:
  assert c['lighting_effect_status']=='unverified_trial'
  if c['lighting_strategy']=='transmitted_backlight':assert c['scene']['light_face_mm'][2]<0
  else:
   assert c['scene']['light_face_mm'][2]>0 and c['scene']['light_shape']=='panel'
   delta=[c['scene']['target_mm'][k]-c['scene']['light_face_mm'][k] for k in range(3)]
   length=sum(v*v for v in delta)**.5
   assert c['scene']['light_direction']==pytest.approx([v/length for v in delta])
 assert result['optimization']['physical_station_count'] is None


def test_alternate_lighting_is_also_model_selected_and_validated():
 candidates=[{'id':'A'},{'id':'B'}]
 d,h=validate_model_selection('{"selected_head_ids":["A"],"alternate_head_ids":["B"],"review":"trial"}',candidates)
 assert d['alternate_head_ids']==['B'] and h==[candidates[0]]
 for ids in [['A'],['B','B'],['invented']]:
  with pytest.raises(ValueError):validate_model_selection(json.dumps(dict(selected_head_ids=['A'],alternate_head_ids=ids,review='bad')),candidates)


def test_automatic_covers_entire_scope_without_silently_dropping_contact_tasks():
 tasks=[task(7,[0,0,0]),task(8,[0,9,0],strategy='needs_method_adapter')]+[task(n,[0,0,0],anchors=[],strategy='needs_method_adapter') for n in range(9,40)]
 r=plan(tasks,library(),design(mode='automatic',fai_numbers=[]))
 assert r['selected_fai']==[7,8]
 assert len(r['task_coverage'])==2
 assert 39 in r['not_marked_fai'] and not r['uncovered_fai']


def test_installation_screen_rejects_mount_clearance_and_forbidden_volume():
 from oracle_viewer.physical import screen
 r=plan([task(7,[0,0,0])],library(),design(fai_numbers=[7]))
 head=next(h for h in r['candidates'] if h['lighting_strategy']=='transmitted_backlight')
 ring=deepcopy(head);ring['lighting_strategy']='front_ring'
 ring['scene'].update(light_shape='ring',light_face_mm=[0,0,100],light_direction=[0,0,-1],light_outer_mm=120,light_inner_mm=44,light_height_mm=39)
 assert any(c['type']=='insufficient_ring_mount_clearance' for c in screen(ring,{})['collisions'])
 check=screen(head,dict(forbidden_zones=[dict(name='fixture',min_mm=[-100,-100,65],max_mm=[100,100,300])]))
 assert check['status']=='rejected' and not check['ready_for_installation']
 assert any(c['type']=='forbidden_zone_intersection' for c in check['collisions'])
 mesh=dict(faces=[dict(id='blocking_face',positions=[0,0,80,1,1,80,2,2,80])])
 assert any(c.get('face_id')=='blocking_face' for c in screen(head,{},mesh)['collisions'])


def test_large_project_solver_is_bounded_and_does_not_claim_exact_minimum():
 r=plan([task(i,[i*30,0,0]) for i in range(1,17)],library(),design(fai_numbers=list(range(1,17))))
 assert len(r['heads'])==16
 assert r['optimization']['algorithm']=='bounded_spatial_candidates_greedy_set_cover'


def test_box_triangle_crossing_is_detected_even_with_vertices_outside():
 from oracle_viewer.physical import solids,triangles_hit_box
 r=plan([task(7,[0,0,0])],library(),design(fai_numbers=[7]))
 head=next(h for h in r['candidates'] if h['lighting_strategy']=='transmitted_backlight')
 box=next(x for x in solids(head['scene']) if x['id']=='camera')
 import numpy as np
 z=head['scene']['camera_flange_mm'][2]+10
 assert triangles_hit_box(np.array([[-100,-100,z],[100,-100,z],[0,100,z]]),[0,1,2],box)
 assert not triangles_hit_box(np.array([[-100,-100,z+500],[100,-100,z+500],[0,100,z+500]]),[0,1,2],box)


def test_all_marked_multipt_tasks_split_into_multiple_frames_without_losing_points():
 tasks=[task(7,[0,0,0],marked=True,strategy='needs_method_adapter',anchors=[{'position':[0,0,0]},{'position':[0,60,0]},{'position':[0,60,4]}]),task(8,[0,5,0],marked=True),task(9,[0,0,0],marked=False,anchors=[])]
 r=plan(tasks,library(),design(mode='automatic',fai_numbers=[]))
 assert r['selected_fai']==[7,8] and r['not_marked_fai']==[9]
 assert {a['id'] for a in r['required_atoms']}=={'7:0','7:1','7:2','8:0'}
 covered={a for h in r['heads'] for a in h['coverage_atoms']}
 assert not r['uncovered_fai'] and not r['uncovered_atoms']
 assert len(r['heads'])>=3 and covered=={'7:0','7:1','7:2','8:0'}
 assert all(h['purpose']=='acquisition_layout_only_method_route_separate' for h in r['heads'])
 assert not r['metrology_accepted']


def test_method_routes_require_local_model_and_exact_batch_coverage():
 from oracle_viewer.channel_routes import propose
 import threading
 class Model:
  _lock=threading.Lock()
  def _generate_once(self,*args,**kw):return self.text,False,20
 model=Model();t=task(7,[0,0,0],features=[],probes=['CONTACT'],symbols=['⏥'],specifications=[],methods=[{'text':'Measure vector points and plane flatness'}],dependencies=[])
 progress=[];model.text='{"routes":[{"number":7,"channel":"3d_surface","proposal":"Sample 3D points and fit the source plane","requirements":["Select sensor and validate equivalence"]}]}'
 routes,attempts=propose(model,[t],lambda *args:progress.append(args))
 assert routes[0]['channel']=='3d_surface' and not routes[0]['metrology_accepted'] and not routes[0]['hardware_selected']
 assert progress[0][0]==0 and progress[-1][:2]==(1,1) and attempts[0]['input_sha256']
 model.text='{"routes":[]}'
 with pytest.raises(ValueError):propose(model,[t],lambda *args:None)


def test_complete_candidate_universe_is_archived_without_losing_cover_or_lighting_types(tmp_path):
 import gzip,hashlib
 from oracle_viewer.workflow import archive_layout_candidates
 heads=[dict(id='P',coverage_atoms=['1:0'],scene=dict(normal=[0,0,1]),lighting_strategy='oblique_panel_reflection'),
        dict(id='B',coverage_atoms=['1:0'],scene=dict(normal=[0,0,1]),lighting_strategy='oblique_bar_reflection'),
        dict(id='P2',coverage_atoms=['1:0'],scene=dict(normal=[0,0,1]),lighting_strategy='oblique_panel_reflection')]
 result=dict(candidates=heads,rejected_candidates=[{'id':str(n)} for n in range(250)],tool_recommendation_head_ids=['P'])
 archive_layout_candidates(result,tmp_path,'run','project')
 assert {c['id'] for c in result['candidates']}=={'P','B'}
 assert len(result['rejected_candidates'])==200 and result['rejected_candidate_count']==250
 audit=result['candidate_audit'];path=tmp_path/audit['file']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['sha256']
 with gzip.open(path,'rt') as f:original=json.load(f)
 assert len(original['candidates'])==3 and len(original['rejected_candidates'])==250


def test_layout_group_references_are_explicit_local_decisions_and_expand_checked_ids():
 candidates=[dict(id='H1'),dict(id='H2')];groups={'TOOL_COVER':['H1','H2']}
 decision,heads=validate_model_selection(json.dumps(dict(selected_head_group_ids=['TOOL_COVER'],alternate_head_ids=[],review='Shared complete cover')),candidates,groups)
 assert decision['selected_head_ids']==['H1','H2'] and heads==candidates
 for raw in [dict(selected_head_group_ids=['UNKNOWN']),dict(selected_head_group_ids=['TOOL_COVER','TOOL_COVER']),dict(selected_head_group_ids=['TOOL_COVER'],selected_head_ids=['H1'])]:
  with pytest.raises(ValueError):validate_model_selection(json.dumps(raw|dict(review='bad')),candidates,groups)


def test_geometry_checkpoint_changes_with_source_lens_or_design_evidence():
 from oracle_viewer.geometry_cache import geometry_key
 data=dict(model=dict(source_sha256='cad'),documents=[]);tasks=[dict(number=1,anchors=[dict(position=[1,2,3])])]
 cat=dict(items=[dict(id='L',spec=dict(magnification=.35))]);settings=dict(roi_mode='maximum');choice=dict(pair_ids=['C|L'],light_ids=['B'])
 original=geometry_key(data,tasks,cat,settings,choice)
 assert original!=geometry_key(data,[dict(number=1,anchors=[dict(position=[1,2,4])])],cat,settings,choice)
 assert original!=geometry_key(data,tasks,dict(items=[dict(id='L',spec=dict(magnification=.7))]),settings,choice)
 assert original!=geometry_key(data,tasks,cat,dict(roi_mode='fixed'),choice)
 assert original!=geometry_key(data|dict(model=dict(source_sha256='new-cad')),tasks,cat,settings,choice)


def test_whole_fai_cover_keeps_single_view_even_when_split_covers_more_atoms():
 from oracle_viewer.workflow import recommend_fai_cover,fai_capture_summary
 atoms=[dict(id=f'{n}:{i}',number=n) for n,count in [(1,2),(2,2)] for i in range(count)]
 def c(id,ids):return dict(id=id,coverage_atoms=ids,optics={'fov_mm':[20,20],'sampling_um_px':5})
 candidates=[c('split-a',['1:0','2:0']),c('split-b',['1:1','2:1']),c('whole-1',['1:0','1:1']),c('whole-2',['2:0','2:1'])]
 chosen=recommend_fai_cover(atoms,candidates)
 assert {h['id'] for h in chosen}=={'whole-1','whole-2'}
 assert all(r['status']=='single_view' for r in fai_capture_summary(atoms,candidates,chosen))
 split=fai_capture_summary(atoms,candidates,candidates[:2])
 assert all(r['status']=='split_despite_single_view_available' for r in split)


def test_single_view_requires_all_points_and_preserves_blocked_atoms():
 from oracle_viewer.workflow import fai_capture_summary
 atoms=[dict(id=f'1:{i}',number=1) for i in range(3)]
 candidates=[dict(id='a',coverage_atoms=['1:0','1:1'])]
 row=fai_capture_summary(atoms,candidates,candidates)[0]
 assert row['status']=='uncovered' and row['single_view_candidate_count']==0
 assert row['uncovered_atoms']==['1:2']


def test_candidate_generation_does_not_let_neighbors_split_a_complete_fai():
 tasks=[task(1,[-8,0,0],anchors=[{'position':[-8,0,0]},{'position':[8,0,0]}]),task(2,[-11,0,0]),task(3,[11,0,0])]
 result=plan(tasks,library(),design(fai_numbers=[1,2,3]))
 assert any({'1:0','1:1'}<=set(c['coverage_atoms']) for c in result['candidates'])
 assert next(row for row in result['fai_capture_summary'] if row['number']==1)['status']=='single_view'
