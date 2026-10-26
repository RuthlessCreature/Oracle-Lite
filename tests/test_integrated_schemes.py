import copy
import pytest
from oracle_viewer.integrated_schemes import validate_scheme
from oracle_viewer.workflow import library,optical_pair

def sample():
 cat=library();heads=[dict(id='H'+str(n),camera='acA2440-35um',lens='TC23024',light='LTRN024RD',coverage_atoms=['7:'+str(n)],fai_numbers=[7],lighting_strategy='oblique_ring') for n in range(3)]
 result=dict(heads=heads,method_routes=[dict(number=7,channel='2d_vision')],selected_fai=[7],uncovered_fai=[],blockers=[],validation=[])
 raw=dict(title='Shared station',review='One camera traverses three views',gaps=[],stations=[dict(id='S1',name='Optical',head_ids=['H0','H1','H2'],method_fai_numbers=[7],sensor_ids=[],trial_light_ids=[],controller_id='LTIC1CH-D1-4',sharing='sequential',motion_mode='move_optics',motion='XYZ trajectory and fixture clearance pending',datum='Actual part datum registered each cycle',sequence='Move, settle, trigger ring and camera, evaluate original method')])
 return cat,result,raw

def test_pose_count_does_not_multiply_camera_or_light_bom():
 cat,result,raw=sample();s=validate_scheme(raw,result,cat,'shared')
 assert s['exposure_count']==3 and s['station_count']==1
 assert next(b['quantity'] for b in s['bom'] if b['id']=='acA2440-35um')==1
 assert next(b['quantity'] for b in s['bom'] if b['id']=='LTRN024RD')==1
 assert len(next(r['exposure_ids'] for r in s['resources'] if r['item_id']=='acA2440-35um'))==3
 assert s['coverage']['assigned']==1 and s['coverage']['metrology_accepted']==0
 assert not s['ready_for_procurement']

@pytest.mark.parametrize('change',['missing_fai','duplicate_head','unknown_device','simultaneous_sharing','missing_datum'])
def test_whole_scheme_rejects_invented_resources_and_incomplete_assignments(change):
 cat,result,raw=sample();st=raw['stations'][0]
 if change=='missing_fai':st['method_fai_numbers']=[]
 if change=='duplicate_head':st['head_ids'].append('H0')
 if change=='unknown_device':st['sensor_ids']=['invented']
 if change=='simultaneous_sharing':st['sharing']='parallel'
 if change=='missing_datum':st['datum']=''
 with pytest.raises(ValueError):validate_scheme(raw,result,cat,'bad')

def test_partial_catalog_does_not_enter_optical_geometry_silently():
 cat=library();c=next(i for i in cat['items'] if i['id']=='MV-CS050-10GM');lens=next(i for i in cat['items'] if i['id']=='OPT-HP03-110-F8')
 check=optical_pair(c,lens,[2400,2000],10)
 assert not check['eligible'] and 'Missing verified layout specifications' in check['reasons'][0]
 assert optical_pair(c,next(i for i in cat['items'] if i['id']=='TC23024'),[2400,2000],10)['eligible']

def test_optical_shortlist_prioritizes_largest_feasible_single_view(monkeypatch):
 import oracle_viewer.workflow as workflow
 import oracle_viewer.integrated_schemes as schemes
 class Model: pass
 cat={'items':[
  {'id':'C1','kind':'camera','vendor':'V1','spec':{}},
  {'id':'C2','kind':'camera','vendor':'V2','spec':{}},
  {'id':'L1','kind':'lens','vendor':'V1','spec':{}},
  {'id':'L2','kind':'lens','vendor':'V2','spec':{}},
  {'id':'B1','kind':'light','vendor':'V','spec':{'type':'backlight','active_mm':[100,100],'body_mm':[100,100,5],'voltage_v':24,'current_a':1}},
 ]}
 def optics(camera,lens,roi,sampling):
  area={('C1','L1'):100,('C1','L2'):180,('C2','L1'):150,('C2','L2'):120}[(camera['id'],lens['id'])]
  return {'eligible':True,'reasons':[],'fov_mm':[area**.5,area**.5],'sampling_um_px':5,'working_distance_mm':100,'dof_mm':5}
 monkeypatch.setattr(workflow,'optical_pair',optics)
 def decision(_model,_prompt,evidence,_limit):
  pairs=evidence['checked_pairs'];assert pairs and all(p['id']=='C1|L2' for p in pairs)
  return {'pair_ids':[pairs[0]['id']],'light_ids':['B1'],'review':'选择最大可用单次视野'},{}
 monkeypatch.setattr(schemes,'generate_json',decision)
 selected,_=schemes.shortlist(Model(),cat,{'roi_px':[100,100],'max_sampling_um_px':10})
 assert selected['pair_ids']==['C1|L2']

def test_separate_stations_need_separate_physical_instances_and_channel_gaps():
 cat,result,raw=sample();result['selected_fai'].append(8);result['method_routes'].append(dict(number=8,channel='confocal_height'))
 st=copy.deepcopy(raw['stations'][0]);st.update(id='S2',head_ids=['H2'],method_fai_numbers=[8]);raw['stations'][0]['head_ids']=['H0','H1'];raw['stations'].append(st)
 s=validate_scheme(raw,result,cat,'parallel')
 assert next(b['quantity'] for b in s['bom'] if b['id']=='acA2440-35um')==2
 assert any('FAI 8' in g and '共焦' in g for g in s['gaps'])
 assert s['coverage']['metrology_accepted']==0

def test_truncated_local_decision_is_retained_and_retried_without_acceptance():
 import json,threading
 from oracle_viewer.integrated_schemes import propose_schemes
 cat,result,raw=sample()
 class Model:
  _lock=threading.Lock();calls=0;design_calls=0
  def _generate_once(self,*args,**kwargs):
   self.calls+=1
   if args[0][0]['content'].startswith('审校'):return json.dumps(dict(review='Separate camera and depth sensors, motion and metrology pending',warnings=[])),False,20
   self.design_calls+=1
   proposal=copy.deepcopy(raw)
   if self.design_calls==3:proposal['stations'][0]['controller_id']='OPT-DPA2024EC-4'
   if self.design_calls==4:proposal['stations'][0]['trial_light_ids']=[next(i['id'] for i in cat['items'] if i['kind']=='light' and i['vendor']=='OPT')]
   return ('{"stations":[' if self.design_calls==1 else json.dumps(proposal)),self.design_calls==1,20
 history=[];model=Model();schemes,attempts=propose_schemes(model,result,cat,lambda *args:history.append(args))
 assert len(schemes)==3 and model.calls==7
 assert attempts[0]['truncated'] is True and attempts[0]['text']=='{"stations":['
 assert 'validation_error' in attempts[0]
 assert all(s['coverage']['assigned']==1 for s in schemes)
 assert any(args[-1] and args[-1][0]['truncated'] for args in history)

def test_model_source_group_refs_expand_full_scope_without_manual_assignments():
 cat,result,raw=sample();s=raw['stations'][0];s.pop('head_ids');s.pop('method_fai_numbers');s.update(head_group_ids=['ALL'],method_group_ids=['2d_vision'],motion_mode='move_part')
 whole=validate_scheme(raw,result,cat,'groups')
 assert whole['stations'][0]['head_ids']==['H0','H1','H2']
 assert whole['stations'][0]['method_fai_numbers']==[7]
 assert whole['coverage']['assigned']==1


def test_fixed_camera_and_fixed_part_cannot_share_multiple_pose_locations():
 cat,result,raw=sample();raw['stations'][0]['motion_mode']='fixed'
 with pytest.raises(ValueError,match='requires explicit motion_mode'):validate_scheme(raw,result,cat,'fixed')


def test_overlapping_group_refs_never_silently_deduplicate_model_assignment():
 cat,result,raw=sample();s=raw['stations'][0];s.pop('head_ids');s.update(head_group_ids=['ALL','POSE_GROUP_1'])
 with pytest.raises(ValueError,match='duplicate exposure'):validate_scheme(raw,result,cat,'overlap')

def test_collaborating_stations_require_model_selected_unique_result_owner():
 cat,result,raw=sample();st=copy.deepcopy(raw['stations'][0]);st.update(id='S2',head_ids=['H2']);raw['stations'][0]['head_ids']=['H0','H1'];raw['stations'].append(st)
 with pytest.raises(ValueError,match='unique result owner'):validate_scheme(copy.deepcopy(raw),result,cat,'collaborative')
 raw['method_group_owners']={'2d_vision':'S2'}
 whole=validate_scheme(raw,result,cat,'collaborative')
 assert whole['coverage']['assigned']==1
 assert whole['fai_result_owners']=={'7':'S2'}
 assert whole['fai_station_memberships']=={'7':['S1','S2']}
 assert whole['stations'][0]['final_measurement_fai']==[] and whole['stations'][1]['final_measurement_fai']==[7]
 assert any('唯一汇总拟合' in g for g in whole['gaps'])

def test_depth_search_regions_do_not_duplicate_camera_hardware():
 cat,result,raw=sample();raw['stations'].append(dict(id='S2',name='Depth alignment',head_ids=[],scan_region_group_ids=['ALL'],method_fai_numbers=[],sensor_ids=['MV-DP3020-01H'],trial_light_ids=[],controller_id=None,sharing='sequential',motion_mode='move_optics',motion='Move depth sensor, verify calibration and clearance',datum='Register actual part datum',sequence='Acquire source marker search regions'))
 whole=validate_scheme(raw,result,cat,'depth-regions')
 assert len(whole['stations'][1]['scan_regions'])==3
 assert next(b['quantity'] for b in whole['bom'] if b['id']=='acA2440-35um')==1
 assert not any('acA2440-35um' in id for id in whole['stations'][1]['resources'])
 assert all(p['status']=='marker_search_region_only_not_validated_depth_sensor_pose' for p in whole['stations'][1]['scan_regions'])

def test_shared_catalog_templates_preserve_every_model_specification():
 from oracle_viewer.integrated_schemes import compact_catalog_specs
 from oracle_viewer.workflow import library
 rows=[dict(id=i['id'],kind=i['kind'],vendor=i['vendor'],spec=i['spec']) for i in library()['items']]
 compact,templates=compact_catalog_specs(rows)
 assert templates and len(compact)==len(rows)
 for original,row in zip(rows,compact):
  expanded=templates.get(row.get('spec_template'),{})|row['spec']
  assert expanded==original['spec'] and row['id']==original['id']

@pytest.mark.parametrize('bad_kind',['split_ids','unknown_light','truncated'])
def test_optical_shortlist_repairs_model_output_and_preserves_failed_attempt(monkeypatch,bad_kind):
 import oracle_viewer.integrated_schemes as schemes
 import oracle_viewer.workflow as workflow
 cat={'items':[
  dict(id='C',kind='camera',vendor='V',spec={}),
  dict(id='L',kind='lens',vendor='V',spec={}),
  dict(id='B',kind='light',vendor='V',spec=dict(type='backlight',active_mm=[100,100],body_mm=[100,100,5],voltage_v=24,current_a=1))]}
 monkeypatch.setattr(workflow,'optical_pair',lambda *a:dict(eligible=True,fov_mm=[20,10],sampling_um_px=5))
 valid=dict(pair_ids=['C|L'],light_ids=['B'],review='最大合格视野，安装待验证')
 calls=[];updates=[]
 def generate(model,prompt,evidence,budget):
  calls.append((prompt,evidence,budget))
  if len(calls)==1:
   if bad_kind=='truncated':raise schemes.DecisionError('truncated',dict(text='{"pair_ids":["C|L"]',truncated=True,tokens=budget))
   raw=copy.deepcopy(valid)
   raw['pair_ids' if bad_kind=='split_ids' else 'light_ids']=['C','L'] if bad_kind=='split_ids' else ['invented']
   return raw,dict(text=str(raw),truncated=False)
  assert '修正上次错误' in prompt
  assert evidence['allowed_pair_ids']==['C|L']
  return valid,dict(text=str(valid),truncated=False)
 monkeypatch.setattr(schemes,'generate_json',generate)
 selected,receipt=schemes.shortlist(object(),cat,dict(roi_px=[100,100],max_sampling_um_px=10),updates.append)
 assert selected==valid and len(calls)==2
 assert len(receipt['attempts'])==2 and receipt['attempts'][0]['validation_error']
 assert updates[0][0]['text'] and len(updates[-1])==2
 assert calls[1][2]>calls[0][2]


def test_optical_shortlist_exhaustion_never_substitutes_a_tool_decision(monkeypatch):
 import oracle_viewer.integrated_schemes as schemes
 import oracle_viewer.workflow as workflow
 cat={'items':[
  dict(id='C',kind='camera',vendor='V',spec={}),dict(id='L',kind='lens',vendor='V',spec={}),
  dict(id='B',kind='light',vendor='V',spec=dict(type='backlight',active_mm=[10,10],body_mm=[10,10,5],voltage_v=24,current_a=1))]}
 monkeypatch.setattr(workflow,'optical_pair',lambda *a:dict(eligible=True,fov_mm=[20,10],sampling_um_px=5))
 monkeypatch.setattr(schemes,'generate_json',lambda *a: (dict(pair_ids=['invented'],light_ids=['B']),dict(text='invalid',truncated=False)))
 updates=[]
 with pytest.raises(schemes.DecisionError) as exc:
  schemes.shortlist(object(),cat,dict(roi_px=[100,100],max_sampling_um_px=10),updates.append)
 assert len(exc.value.attempt['attempts'])==3 and len(updates)==3
 assert all('validation_error' in a for a in exc.value.attempt['attempts'])


def test_maximum_view_uses_real_sensor_and_image_circle_instead_of_requested_oversize_roi():
 from oracle_viewer.workflow import optical_pair_for_design
 cat=library();cam=next(i for i in cat['items'] if i['id']=='acA2440-35um');lens=next(i for i in cat['items'] if i['id']=='TC23024')
 fixed=optical_pair_for_design(cam,lens,dict(roi_px=[5000,5000],roi_mode='fixed',max_sampling_um_px=10))
 assert not fixed['eligible']
 maximum=optical_pair_for_design(cam,lens,dict(roi_px=[5000,5000],roi_mode='maximum',max_sampling_um_px=10))
 assert maximum['eligible'] and maximum['roi_px']==[2443,2048]
 assert maximum['sensor_diagonal_mm']<=lens['spec']['image_circle_mm']
 assert maximum['fov_mm'][0]*maximum['fov_mm'][1]>23.66*19.72
 # A larger width at this height leaves the verified lens image circle.
 invalid=optical_pair(cam,lens,[2444,2048],10)
 assert not invalid['eligible'] and 'sensor ROI diagonal exceeds lens image circle' in invalid['reasons']


def test_maximum_roi_has_globally_largest_integer_area_for_asymmetric_sensors():
 from oracle_viewer.workflow import maximum_sensor_roi
 for width,height,diagonal in [(17,9,15),(5,20,16),(21,22,19),(10,10,15)]:
  w,h=maximum_sensor_roi(width,height,diagonal)
  feasible=[x*y for x in range(1,width+1) for y in range(1,height+1) if x*x+y*y<=diagonal*diagonal]
  assert w*h==max(feasible)


def test_lens_magnification_controls_fov_sampling_and_maximum_sensor_area():
 from oracle_viewer.workflow import optical_pair_for_design
 cat=library();camera=next(i for i in cat['items'] if i['id']=='acA2440-35um');lens=next(i for i in cat['items'] if i['id']=='TC23024')
 design=dict(roi_mode='maximum',roi_px=[5000,5000],max_sampling_um_px=10)
 a=optical_pair_for_design(camera,lens,design)
 higher=copy.deepcopy(lens);higher['spec']['magnification']*=2
 b=optical_pair_for_design(camera,higher,design)
 assert a['eligible'] and b['eligible'] and a['roi_px']==b['roi_px']
 assert b['fov_mm']==pytest.approx([v/2 for v in a['fov_mm']])
 assert b['sampling_um_px']==pytest.approx(a['sampling_um_px']/2)
 smaller=copy.deepcopy(lens);smaller['spec']['image_circle_mm']=8
 cropped=optical_pair_for_design(camera,smaller,design)
 assert cropped['eligible'] and cropped['sensor_diagonal_mm']<=8
 assert cropped['fov_mm'][0]*cropped['fov_mm'][1]<a['fov_mm'][0]*a['fov_mm'][1]


@pytest.mark.parametrize('changed',['none','model','task','engine','document'])
def test_station_method_reuse_requires_identical_local_model_and_source_evidence(changed):
 import hashlib
 from pathlib import Path
 from oracle_viewer.workflow import station_method_cache
 import oracle_viewer.channel_routes as route_engine
 tasks=[dict(number=1,methods=[dict(text='Original XYZ and datum')],anchors=[dict(position=[1,2,3])])]
 identity=dict(model_alias='local-model',inference_backend='strata-native',base_model_path=None,adapter_dir=None)
 docs=[dict(role='msop',filename='source.pdf',sha256='original')]
 routes=[dict(number=1,channel='3d_surface',status='proposed_unvalidated',metrology_accepted=False,proposal='Original local proposal',requirements=['Validate registration'])]
 station=dict(id='station',status='complete',model_identity=copy.deepcopy(identity),source_summary=dict(documents=copy.deepcopy(docs)),task_snapshot=copy.deepcopy(tasks),routes=routes,attempts=[dict(numbers=[1],input_sha256='model-input',text='local output')],route_engine_sha256=hashlib.sha256(Path(route_engine.__file__).read_bytes()).hexdigest())
 # The layout adds derived coordinates independently; they do not replace source evidence.
 tasks[0]['msop_coordinate_map']=dict(status='unregistered',source_frame_registered_to_cad=False)
 if changed=='model':station['model_identity']['model_alias']='other-model'
 if changed=='task':station['task_snapshot'][0]['methods'][0]['text']='Changed nominal'
 if changed=='engine':station['route_engine_sha256']='old-engine'
 if changed=='document':station['source_summary']['documents'][0]['sha256']='other-document'
 cached=station_method_cache(station,tasks,identity,docs)
 assert bool(cached)==(changed=='none')
 if cached:assert cached[0]==routes


def test_roi_inside_large_sensor_requires_a_complete_camera_lens_pair():
 from oracle_viewer import integrated_schemes as schemes
 from oracle_viewer.workflow import library
 with pytest.raises(ValueError,match='20000 × 20000.*像素范围内.*完整相机—镜头组合'):
  schemes.shortlist(object(),library(),dict(roi_mode='fixed',roi_px=[20000,20000],max_sampling_um_px=100))
