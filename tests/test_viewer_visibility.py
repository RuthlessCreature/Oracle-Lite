import numpy as np
from oracle_viewer.visibility import ViewGeometry,frame
from oracle_viewer.workflow import plan,library

def face(id,points):
 return dict(id=id,positions=np.array(points).reshape(-1).tolist(),indices=[0,1,2,0,2,3])
def slab():
 return dict(faces=[face('top',[[-5,-5,1],[5,-5,1],[5,5,1],[-5,5,1]]),face('bottom',[[-5,-5,0],[-5,5,0],[5,5,0],[5,-5,0]]),face('side',[[-5,5,0],[-5,5,1],[5,5,1],[5,5,0]])])
def task(n,p,f):return dict(number=n,anchors=[dict(position=p,face_id=f)],gaps=[],marked=True)
def design():
 cat=library();cam=cat['items'][0]['id'];return dict(mode='automatic',fai_numbers=[],normal_axis='auto',roi_px=[2400,2000],feature_radius_mm=.1,max_sampling_um_px=10,backlight_gap_mm=20,_pair_ids=[cam+'|TC23024'])
def test_backside_markers_cannot_be_covered_from_front_and_opaque_slab_has_no_transmission():
 r=plan([task(1,[0,0,1],'top'),task(2,[0,0,0],'bottom')],library(),design(),slab())
 assert not r['uncovered_fai']
 assert r['candidates']
 for h in r['candidates']:
  assert h['fai_numbers']!=[1,2]
  assert h['lighting_strategy']!='transmitted_backlight'
  assert h['visibility']['status']=='marker_rays_clear'
  assert h['scene']['normal'][2]>0 if h['fai_numbers']==[1] else h['scene']['normal'][2]<0

def test_side_face_has_actual_y_view_and_right_handed_camera_frame():
 r=plan([task(3,[0,5,.5],'side')],library(),design(),slab())
 assert r['heads'] and not r['uncovered_fai']
 for h in r['heads']:
  assert abs(h['scene']['normal'][1])>.99
  f=np.array([h['scene']['basis_x'],h['scene']['basis_y'],h['scene']['normal']])
  assert np.allclose(f@f.T,np.eye(3)) and np.linalg.det(f)>0

def test_missing_or_offset_face_never_gets_default_z_direction():
 r=plan([task(1,[0,0,10],'top'),task(2,[0,0,0],'not-a-face')],library(),design(),slab())
 assert r['heads']==[] and r['uncovered_fai']==[1,2] and len(r['visibility_issues'])==2
 r=plan([task(1,[0,0,0],'top')],library(),design())
 assert not r['heads'] and r['uncovered_fai']==[1]

def test_workpiece_and_fixture_block_view_rays_including_small_thickness():
 g=ViewGeometry(slab(),[dict(min_mm=[-1,-1,5],max_mm=[1,1,6])])
 assert g.blocked([0,0,1],[0,0,1],100)=='fixture_keepout'
 assert ViewGeometry(slab()).blocked([0,0,0],[0,0,1],100)=='workpiece_occlusion'
 assert g.blocked([0,0,1],[0,0,-1],100)=='workpiece_occlusion'
 assert g.blocked([0,0,0],[0,0,-1],100) is None
 for n in ([0,1,0],[0,-1,0],[1,1,1]):assert np.allclose(frame(n)@frame(n).T,np.eye(3))

def test_old_axis_input_is_ignored_when_real_geometry_exists():
 d=design()|{'normal_axis':'+Z'}
 r=plan([task(2,[0,0,0],'bottom')],library(),d,slab())
 assert r['view_direction_mode']=='cad_geometry_auto'
 assert r['heads'] and all(h['scene']['normal'][2]<0 for h in r['heads'])

def test_model_candidate_scope_preserves_all_available_atoms_and_keeps_view_sides_separate():
 from oracle_viewer.workflow import bounded_model_candidates,solve_cover
 r=plan([task(1,[0,0,1],'top'),task(2,[0,0,0],'bottom')],library(),design(),slab())
 offered=bounded_model_candidates(r['candidates'],[h['id'] for h in r['heads']])
 assert {a for h in offered for a in h['coverage_atoms']}=={a for h in r['candidates'] for a in h['coverage_atoms']}
 assert len(offered)<=len(r['candidates'])
 for h in offered:assert h['fai_numbers']!=[1,2]

def test_production_submission_never_uses_user_axis(tmp_path,monkeypatch):
 from oracle_viewer import workflow
 from oracle_viewer.store import ViewerStore
 class Pool:
  def submit(self,*args):pass
 store=ViewerStore(tmp_path);project=store.create('Unseen part','different','B')
 t=task(1,[0,0,1],'top')
 monkeypatch.setattr(workflow,'compile_tasks',lambda *args:[t])
 result=workflow.WorkflowService(store,None,Pool()).submit(project['id'],design()|{'normal_axis':'+Z'})
 assert result['design']['normal_axis']=='auto'

def test_assembly_requires_rotation_for_opposite_views():
 import pytest
 from oracle_viewer.assembly import compile_assembly
 r=plan([task(1,[0,0,1],'top'),task(2,[0,0,0],'bottom')],library(),design(),slab())
 heads=r['heads'];groups={'G1':[heads[0]['id']],'G2':[heads[1]['id']]};hids=[h['id'] for h in heads]
 scheme=dict(id='shared',stations=[dict(id='S1',head_ids=hids,sensor_ids=[],scan_regions=[])],bom=r['bom'])
 raw=dict(review='Multi-side acquisition',camera_units=[dict(id='CAM1',station_id='S1',group_ids=list(groups),home_head_id=hids[0],motion_mode='move_optics',motion='Translate and rotate; validate fixture clearance')],light_units=[dict(id='L1',station_id='S1',group_ids=list(groups),mode='existing_moving_light',home_head_id=hids[0])])
 run=dict(result=r,design=design())
 with pytest.raises(ValueError,match='rotary_mount'):compile_assembly(raw,run,scheme,groups,slab())
 raw['camera_units'][0]['rotation_mode']='rotary_mount'
 a=compile_assembly(raw,run,scheme,groups,slab())
 assert a['camera_units'][0]['rotation_required'] and len(a['camera_units'][0]['poses'])==2
 assert not a['ready_for_installation']


def test_identical_local_poses_in_different_stations_do_not_collide():
 import copy
 from oracle_viewer.assembly import compile_assembly
 r=plan([task(1,[0,0,1],'top')],library(),design(),slab())
 first=r['heads'][0];second=copy.deepcopy(first);second['id']='OTHER';r['heads']=[first,second]
 groups={'G1':[first['id']],'G2':['OTHER']}
 scheme=dict(id='shared',stations=[dict(id='S1',head_ids=[first['id']],sensor_ids=[],scan_regions=[]),
                                 dict(id='S2',head_ids=['OTHER'],sensor_ids=[],scan_regions=[])],bom=r['bom'])
 raw=dict(review='Two independent stations',camera_units=[],light_units=[])
 for index,h in enumerate((first,second),1):
  raw['camera_units'].append(dict(id='CAM'+str(index),station_id='S'+str(index),group_ids=['G'+str(index)],home_head_id=h['id'],motion_mode='move_optics',motion='Local station motion'))
  raw['light_units'].append(dict(id='L'+str(index),station_id='S'+str(index),group_ids=['G'+str(index)],mode='existing_moving_light',home_head_id=h['id']))
 run=dict(result=r,design=design())
 separate=compile_assembly(copy.deepcopy(raw),run,scheme,groups,slab())
 assert not any(c['type'].startswith('inter_unit') for c in separate['collisions'])
 assert len(separate['motion_screen']['stations'])==2
 scheme['stations'][0]['head_ids'].append('OTHER');scheme['stations'].pop()
 for unit in raw['camera_units']+raw['light_units']:unit['station_id']='S1'
 same=compile_assembly(raw,run,scheme,groups,slab())
 assert any(c['type'].startswith('inter_unit') for c in same['collisions'])

def test_visual_circle_uses_cylinder_axis_instead_of_radial_wall_normal():
 points=[];indices=[]
 for i in range(16):
  a=i*2*np.pi/16;b=(i+1)*2*np.pi/16;k=len(points)
  points.extend([[np.cos(a),np.sin(a),0],[np.cos(b),np.sin(b),0],[np.cos(b),np.sin(b),1],[np.cos(a),np.sin(a),1]])
  indices.extend([k,k+1,k+2,k,k+2,k+3])
 mesh=dict(faces=[dict(id='hole',surface='cylinder',positions=np.array(points).reshape(-1).tolist(),indices=indices)])
 t=task(1,[1,0,1],'hole')|{'strategy':'2d_visual_circle'}
 r=plan([t],library(),design(),mesh)
 assert r['heads']
 assert r['required_atoms'][0]['surface_evidence']['direction_basis']=='cylinder_axis_for_vision_circle'
 assert all(abs(h['scene']['normal'][2])>.99 for h in r['heads'])

def test_method_format_retry_retains_evidence_and_resumes_validated_prefix():
 import json,threading
 from oracle_viewer.channel_routes import propose
 class Model:
  _lock=threading.Lock();calls=0
  def _generate_once(self,*args,**kwargs):
   self.calls+=1
   if self.calls==1:return '{"routes":[',False,10
   return json.dumps({'routes':[{'number':2,'channel':'2d_vision','proposal':'Measure the specified circle in the source datum','requirements':['Validate edge and datum']}]}),False,30
 prior=[dict(number=1,channel='2d_vision',proposal='Previously validated proposal',requirements=['Validate datum'],status='proposed_unvalidated',metrology_accepted=False,hardware_selected=False)]
 history=[];model=Model();routes,attempts=propose(model,[task(1,[0,0,1],'top')|{'dependencies':[]},task(2,[0,0,0],'bottom')|{'dependencies':[]}],lambda *args:None,prior_routes=prior,attempt_progress=lambda attempts:history.append(attempts))
 assert [r['number'] for r in routes]==[1,2] and model.calls==2
 assert attempts[0]['validation_error'] and not attempts[1].get('validation_error') and history
 assert not routes[1]['metrology_accepted']

def test_label_drag_does_not_invalidate_measurement_evidence_but_moving_point_does():
 from copy import deepcopy
 from oracle_viewer.workflow import fingerprint,library,planning_evidence
 data=dict(documents=[],model={},annotations={'fai':{'anchors':[dict(face_id='face',position=[1,2,3],label_offset=[32,-26])]}},bindings={})
 initial=fingerprint(data,library());moved=deepcopy(data);moved['annotations']['fai']['anchors'][0]['label_offset']=[-50,200]
 assert fingerprint(moved,library())==initial and planning_evidence(moved)==planning_evidence(data)
 moved['annotations']['fai']['anchors'][0]['position'][2]=4
 assert fingerprint(moved,library())!=initial

def test_assembly_context_templates_preserve_all_pose_numbers_and_directions():
 from oracle_viewer.assembly import assembly_context,groups_for
 r=plan([task(1,[0,0,1],'top'),task(2,[0,0,0],'bottom')],library(),design(),slab())
 hids=[h['id'] for h in r['heads']];run=dict(result=r,design=design());scheme=dict(stations=[dict(id='S1',head_ids=hids,motion='Rotate and move')]);ctx=assembly_context(run,scheme,groups_for(r['heads']))
 original={h['id']:h for h in r['heads']}
 for h in ctx['heads']:assert ctx['scene_templates'][h['scene_template']]|h['scene']==original[h['id']]['scene']

def test_assembly_repair_identifies_missing_groups_and_duplicate_poses():
 from oracle_viewer.assembly import ownership_error
 error=ownership_error(['H1','H1'],{'H1':{},'H2':{}},{'G1':['H1'],'G2':['H2']},'physical camera')
 assert 'missing_groups=["G2"]' in error and 'duplicate_heads=["H1"]' in error
