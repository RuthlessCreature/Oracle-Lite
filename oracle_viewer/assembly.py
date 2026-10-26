"""Native-model assembly layouts. Exposure groups are distinct from installed units.
Merged emitters without vendor dimensions are requirement surfaces, never fake hardware.
"""
import copy, hashlib, json, math, uuid
from pathlib import Path
import numpy as np
from .integrated_schemes import generate_json, DecisionError
from .physical import solids, samples, inside, screen
from .workflow import library, now

VERSION='assembly-v3-motion-screen'

def evidence_key(run, scheme_id):
 return hashlib.sha256(json.dumps(dict(version=VERSION,engine=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('motion.py').read_bytes()+Path(__file__).with_name('hardware_checks.py').read_bytes()).hexdigest(),run=run['id'],fingerprint=run['input_fingerprint'],scheme=scheme_id),sort_keys=True).encode()).hexdigest()

def groups_for(heads):
 # Spatial group proposals only; the local model chooses hardware ownership and home poses.
 buckets={}
 for h in heads:
  # Exposure groups represent reusable equipment families, not viewing poses.
  # Keep distinct normals/basis vectors in each group's source heads so the
  # model and compiler can require rotation where needed. Splitting by exact
  # normal created one group per pose (31 groups in the pilot), overwhelming
  # the assembly decision without adding any physical constraint.
  key=(h['camera'],h['lens'],h['lighting_strategy'])
  buckets.setdefault(key,[]).append(h)
 groups={}
 for bucket in buckets.values():
  axis=max(range(3),key=lambda k:max(h['scene']['target_mm'][k] for h in bucket)-min(h['scene']['target_mm'][k] for h in bucket))
  ordered=sorted(bucket,key=lambda h:h['scene']['target_mm'][axis])
  count=min(4,max(1,math.ceil(len(ordered)/8)))
  for chunk in np.array_split(np.arange(len(ordered)),count):
   groups['G'+str(len(groups)+1)]=[ordered[int(i)]['id'] for i in chunk]
 return groups

def ownership_error(assigned,heads,groups,owner):
 missing=set(heads)-set(assigned)
 duplicate={h for h in assigned if assigned.count(h)>1}
 return (f'Every exposure needs exactly one {owner}; '
         f'missing_groups={json.dumps([g for g,hs in groups.items() if any(h in missing for h in hs)])}; '
         f'duplicate_heads={json.dumps(sorted(duplicate))}. Preserve all other allocations when repairing the layout.')

def choose_camera_homes(cameras,heads,groups,scheme,design,mesh,clearance):
 """Choose collision-screened home poses from model-assigned exposure members.

 The model still owns camera/group allocation. This bounded geometric solver
 only selects a home pose already present in each camera's assigned source
 exposures, preferring the model's choice and minimizing envelope conflicts.
 It never claims that the motion path between poses is collision-free.
 """
 station_heads={s['id']:set(s['head_ids']) for s in scheme['stations']}
 options=[]
 solid_cache={};base_cache={}
 for c in cameras:
  refs=c.get('group_ids',[])
  if not isinstance(refs,list) or any(g not in groups for g in refs):return
  hids=[h for g in refs for h in groups[g]]
  available=[h for h in hids if h in station_heads.get(c.get('station_id'),set())]
  eligible=[]
  for hid in available:
   h=heads[hid]
   if (h['camera'],h['lens']) != (heads.get(c.get('home_head_id'),h)['camera'],heads.get(c.get('home_head_id'),h)['lens']):
    continue
   if hid not in solid_cache:solid_cache[hid]=solids(h['scene'])[:2]
   if hid not in base_cache:
    base_cache[hid]=[x for x in screen(h,design,mesh)['collisions'] if 'light' not in x['objects']]
   if not base_cache[hid]:eligible.append(hid)
  if not eligible:eligible=available
  if c.get('home_head_id') in eligible:eligible.remove(c['home_head_id']);eligible.insert(0,c['home_head_id'])
  options.append(eligible)
 if len(options)<2 or any(not o for o in options):return
 pair_cache={}
 def pair_cost(i,a,j,b):
  if cameras[i].get('station_id')!=cameras[j].get('station_id'):return 0
  key=(i,a,j,b)
  if key in pair_cache:return pair_cache[key]
  score=0
  for sa in solid_cache[a]:
   for sb in solid_cache[b]:
    if np.any(inside(samples(sa),sb,.001)) or np.any(inside(samples(sb),sa,.001)):
     score+=100
    elif np.any(inside(samples(sa),sb,-clearance)) or np.any(inside(samples(sb),sa,-clearance)):
     score+=10
  pair_cache[key]=score
  return score
 # Beam search keeps this tractable for projects with many exposure heads.
 beam=[(0.0,[])]
 for i,candidates in enumerate(options):
  expanded=[]
  preferred=cameras[i].get('home_head_id')
  for score,chosen in beam:
   for hid in candidates:
    if hid in chosen:continue
    added=sum(pair_cost(i,hid,j,other) for j,other in enumerate(chosen))
    expanded.append((score+added+(0 if hid==preferred else .001),chosen+[hid]))
  if not expanded:return
  beam=sorted(expanded,key=lambda x:x[0])[:256]
 best_score,best=beam[0]
 if best_score < sum(pair_cost(i,c['home_head_id'],j,d['home_head_id']) for i,c in enumerate(cameras) for j,d in enumerate(cameras[:i]) if c.get('home_head_id') in solid_cache and d.get('home_head_id') in solid_cache):
  for c,hid in zip(cameras,best):c['home_head_id']=hid

def compile_assembly(raw,run,scheme,groups,mesh):
 heads={h['id']:h for h in run['result']['heads']};items={i['id']:i for i in library()['items']}
 cameras=raw.get('camera_units');lamps=raw.get('light_units')
 if not isinstance(cameras,list) or not 1<=len(cameras)<=8 or not isinstance(lamps,list) or not 1<=len(lamps)<=8:raise ValueError('Need 1..8 camera units and 1..8 light units')
 if not isinstance(raw.get('review'),str):raise ValueError('Missing model review')
 if len({c.get('id') for c in cameras})!=len(cameras) or len({l.get('id') for l in lamps})!=len(lamps):raise ValueError('Duplicate unit IDs')
 station_ids={s['id'] for s in scheme['stations'] if s['head_ids']}
 choose_camera_homes(cameras,heads,groups,scheme,run['design'],mesh,run['design'].get('minimum_clearance_mm',2))
 rotation_errors=[]
 for c in cameras:
  refs=c.get('group_ids',[])
  if not isinstance(refs,list) or any(g not in groups for g in refs):continue
  hs=[heads[h] for g in refs for h in groups[g]]
  if len({(tuple(h['scene']['normal']),tuple(h['scene'].get('basis_x',[]))) for h in hs})>1:
   expected='rotary_mount' if c.get('motion_mode')=='move_optics' else 'rotary_fixture'
   if c.get('rotation_mode')!=expected:rotation_errors.append(dict(id=c.get('id'),motion_mode=c.get('motion_mode'),required_rotation_mode=expected,received=c.get('rotation_mode')))
 if rotation_errors:raise ValueError('Different viewing sides require explicit rotation for ALL listed cameras: '+json.dumps(rotation_errors)+'. Translation cannot make backside points visible from the front.')
 assigned=[];units=[]
 for c in cameras:
  refs=c.get('group_ids',[])
  if not isinstance(refs,list) or not refs or len(refs)!=len(set(refs)) or any(g not in groups for g in refs):raise ValueError('Unknown camera exposure groups')
  hids=[h for g in refs for h in groups[g]];assigned+=hids
  if c.get('station_id') not in station_ids or c.get('home_head_id') not in hids:raise ValueError('Camera station/home not in source exposures')
  if c.get('motion_mode') not in ('move_optics','move_part'):raise ValueError('Multi-pose unit requires relative motion')
  if not isinstance(c.get('id'),str) or not c['id'] or not isinstance(c.get('motion'),str):raise ValueError('Invalid camera identity/motion')
  hs=[heads[h] for h in hids];home=heads[c['home_head_id']]
  if len({(h['camera'],h['lens']) for h in hs})!=1:raise ValueError('One physical camera cannot mix camera/lens models')
  orientation_change=len({(tuple(h['scene']['normal']),tuple(h['scene'].get('basis_x',[]))) for h in hs})>1
  expected='rotary_mount' if c['motion_mode']=='move_optics' else 'rotary_fixture'
  if orientation_change and c.get('rotation_mode')!=expected:raise ValueError('Different viewing sides require explicit '+expected+'; translation cannot make backside points visible from the front')
  c['rotation_required']=orientation_change;c['rotation_validation_status']='pending' if orientation_change else 'not_required'

  if any(h not in next(s for s in scheme['stations'] if s['id']==c['station_id'])['head_ids'] for h in hids):raise ValueError('Exposure assigned outside owning station')
  units.append(dict(**c,head_ids=hids,head=copy.deepcopy(home),fai_numbers=sorted({n for h in hs for n in h['fai_numbers']}),targets_mm=[h['scene']['target_mm'] for h in hs],camera=home['camera'],lens=home['lens'],poses=[dict(id=h['id'],scene=h['scene'],coverage_atoms=h['coverage_atoms']) for h in hs]))
 if len(assigned)!=len(set(assigned)) or set(assigned)!=set(heads):raise ValueError(ownership_error(assigned,heads,groups,'physical camera'))
 assigned_lights=[];lights=[]
 for l in lamps:
  refs=l.get('group_ids',[])
  if not isinstance(refs,list) or not refs or len(refs)!=len(set(refs)) or any(g not in groups for g in refs):raise ValueError('Invalid light exposure groups')
  hids=[h for g in refs for h in groups[g]];assigned_lights+=hids;hs=[heads[h] for h in hids]
  if not isinstance(l.get('id'),str) or not l['id']:raise ValueError('Invalid light identity')
  if l.get('station_id') not in station_ids or any(h not in next(s for s in scheme['stations'] if s['id']==l['station_id'])['head_ids'] for h in hids):raise ValueError('Light groups outside owning station')
  if l.get('mode') in ('merged_panel_requirement','catalog_panel'):
   if any(h['lighting_strategy']!='transmitted_backlight' or h['scene']['light_shape']!='panel' for h in hs):raise ValueError('Ring, oblique, reflective lighting cannot become a transmission panel')
   normals={tuple(h['scene']['normal']) for h in hs};directions={tuple(h['scene']['light_direction']) for h in hs}
   from .visibility import frame
   basis=frame(hs[0]['scene']['normal'])
   camera_frames=[np.array([h['scene'].get('basis_x',basis[0]),h['scene'].get('basis_y',basis[1]),h['scene']['normal']]) for h in hs]
   if len(normals)!=1 or len(directions)!=1:raise ValueError('Merged panel requires the same viewing side and lighting direction; opposite sides need separate lights')
   margin=run['design'].get('minimum_clearance_mm',2);corners=[]
   for h,camera_frame in zip(hs,camera_frames):
    center=np.array(h['scene']['target_mm']);w,height=h['scene']['fov_xy_mm']
    corners.extend((center+x*camera_frame[0]+y*camera_frame[1])@basis.T for x in [-w/2,w/2] for y in [-height/2,height/2])
   points=np.array(corners);lo=points.min(axis=0);hi=points.max(axis=0);lo[:2]-=margin;hi[:2]+=margin
   center=(lo+hi)/2;center[2]=lo[2]-run['design']['backlight_gap_mm'];face=center@basis
   lights.append(dict(id=l['id'],mode=l['mode'],head_ids=hids,station_id=l.get('station_id'),active_requirement_mm=(hi-lo)[:2].tolist(),face_mm=face.tolist(),direction=list(next(iter(directions))),basis_x=basis[0].tolist(),basis_y=basis[1].tolist(),status='required_emitting_area_only_no_verified_supplier_body',gaps=['发光面积按同向拍摄视场合并，正反面光源不能共用为一块背光','实际透射光路、均匀性和被测边缘需实拍验证','厚度、安装孔、功率与控制器待选型'],fai_numbers=sorted({n for h in hs for n in h['fai_numbers']})))
   from .hardware_checks import panel_candidates
   lights[-1]['hardware_candidates']=panel_candidates(library(),lights[-1]['active_requirement_mm'])
   if l.get('mode')=='catalog_panel':
    selected=next((p for p in lights[-1]['hardware_candidates'] if p['item_id']==l.get('item_id') and 0 in p['rotations_deg']),None)
    if not selected:raise ValueError('Selected shared backlight lacks verified active/body dimensions or does not cover the required area in the declared orientation')
    h=copy.deepcopy(hs[0]);h['scene'].update(target_mm=(((lo+hi)/2)@basis).tolist(),light_face_mm=face.tolist(),light_body_mm=selected['body_mm'],light_direction=list(next(iter(directions))),normal=hs[0]['scene']['normal'],basis_x=basis[0].tolist(),basis_y=basis[1].tolist())
    h['light']=selected['item_id'];lights[-1].update(item_id=selected['item_id'],head=h,status='catalog_dimension_envelope_installation_unverified',source_url=selected['source_url'])
  elif l.get('mode')=='existing_moving_light':
   home=l.get('home_head_id')
   if home not in hids or len({h['light'] for h in hs})!=1:raise ValueError('Existing light needs one known model and member pose')
   lights.append(dict(id=l['id'],mode=l['mode'],station_id=l['station_id'],head_ids=hids,home_head_id=home,head=copy.deepcopy(heads[home]),item_id=heads[home]['light'],status='moving_light_envelope_installation_unverified',fai_numbers=sorted({n for h in hs for n in h['fai_numbers']})))
  else:raise ValueError('Unknown light mode')
 if len(assigned_lights)!=len(set(assigned_lights)) or set(assigned_lights)!=set(heads):raise ValueError(ownership_error(assigned_lights,heads,groups,'light owner'))
 # Screen camera/lens home placements pairwise; never mark dynamic travel as safe.
 collisions=[];bodies=[]
 for c in units:
  for solid in solids(c['head']['scene'])[:2]:bodies.append((c['id']+':'+solid['id'],solid))
  p=screen(c['head'],run['design'],mesh)
  for hit in p['collisions']:
   # Original small light is replaced in merged-panel mode, so do not report its collision.
   if 'light' not in hit['objects']:collisions.append(dict(hit,unit=c['id']))
 owners={c['id']:c['station_id'] for c in units}
 for i,(aid,a) in enumerate(bodies):
  for bid,b in bodies[i+1:]:
   if aid.split(':')[0]==bid.split(':')[0]:continue
   if owners[aid.split(':')[0]]!=owners[bid.split(':')[0]]:continue
   if np.any(inside(samples(a),b,.001)) or np.any(inside(samples(b),a,.001)):collisions.append(dict(objects=[aid,bid],type='inter_unit_home_envelope_intersection'))
   elif np.any(inside(samples(a),b,-run['design'].get('minimum_clearance_mm',2))) or np.any(inside(samples(b),a,-run['design'].get('minimum_clearance_mm',2))):collisions.append(dict(objects=[aid,bid],type='inter_unit_home_clearance_warning'))
 for light in lights:
  if not light.get('head'):continue
  for hit in screen(light['head'],run['design'],mesh)['collisions']:
   if 'light' in hit['objects']:collisions.append(dict(hit,unit=light['id']))
 bom={}
 for c in units:
  for key in ('camera','lens'):
   row=bom.setdefault(c[key],dict(id=c[key],quantity=0,instance_ids=[],kind=key,status='conditional_candidate',source_url=items[c[key]]['source_url']))
   row['quantity']+=1;row['instance_ids'].append(c['id']+':'+key)
 for l in lights:
  id=l.get('item_id',l['id']);row=bom.setdefault(id,dict(id=id,quantity=0,instance_ids=[],kind='light',status=l['status']))
  row['quantity']+=1;row['instance_ids'].append(l['id'])
  if l.get('active_requirement_mm'):row['active_requirement_mm']=l['active_requirement_mm']
  if l.get('source_url'):row['source_url']=l['source_url']
 # Preserve depth devices, trial lights, controllers and unresolved integration components.
 for row in scheme['bom']:
  if row.get('kind') not in ('camera','lens') and row['id'] not in {h['light'] for h in heads.values()}:bom.setdefault(row['id'],copy.deepcopy(row))
 from .motion import screen_motion
 station_motion=[]
 for station_id in sorted({c['station_id'] for c in units}):
  local_units=[c for c in units if c['station_id']==station_id]
  local_lights=[l for l in lights if l.get('station_id')==station_id]
  report=screen_motion(local_units,heads,mesh,run['design'],local_lights)
  station_motion.append(dict(station_id=station_id,**report))
 motion=dict(cameras=[c for r in station_motion for c in r['cameras']],
             lights=[l for r in station_motion for l in r.get('lights',[])],
             potential_interference_count=sum(r['potential_interference_count'] for r in station_motion),
             stations=station_motion,status='per_station_screen_only',
             path_validated=False,simultaneous_motion_validated=False,fixture_complete=False,
             controller_program_generated=False,
             unresolved_unit_ids=[id for r in station_motion for id in r['unresolved_unit_ids']],
             limitations=list(dict.fromkeys(x for r in station_motion for x in r['limitations'])),
             inter_station_transfer_status='unverified')
 return dict(review=raw['review'],camera_units=units,light_units=lights,bom=list(bom.values()),exposure_count=len(heads),selected_fai=run['result']['selected_fai'],required_atom_count=len(run['result']['required_atoms']),collisions=collisions,motion_screen=motion,status='blocked_static_interference' if collisions else ('motion_changes_required' if motion['potential_interference_count'] else 'conditional_assembly'),metrology_accepted=False,ready_for_installation=False,depth_stations=[dict(id=s['id'],sensor_ids=s['sensor_ids'],search_regions=s['scan_regions']) for s in scheme['stations'] if s['sensor_ids']],gaps=(['大型光源仅为有效面积需求，尚未选定机械结构或验证透射光路'] if any(l['mode']=='merged_panel_requirement' for l in lights) else ['光源已选候选型号；实际光路、安装、亮度和触发待验证'])+['各相机组仍是分时移动采集，运动包络已自动筛查；完整工装、线缆和轴约束需验证','三维/共焦传感器尚无完整有效位姿，保留搜索区域而不编造安装'],parent_scheme_id=scheme['id'])

def assembly_context(run,scheme,groups):
 """Lossless shared scene dimensions; preserve every source pose and direction."""
 heads=run['result']['heads'];buckets={};templates={};rows=[]
 for h in heads:buckets.setdefault((h['camera'],h['lens'],h['light'],h['lighting_strategy']),[]).append(h)
 for group in buckets.values():
  shared={k:v for k,v in group[0]['scene'].items() if all(k in h['scene'] and h['scene'][k]==v for h in group[1:])}
  key='SCENE'+str(len(templates)+1);templates[key]=shared
  for h in group:rows.append({k:h[k] for k in ('id','camera','lens','light','lighting_strategy','fai_numbers')}|dict(scene_template=key,scene={k:v for k,v in h['scene'].items() if k not in shared}))
 return dict(groups=groups,stations=[dict(id=s['id'],head_ids=s['head_ids'],motion=s['motion']) for s in scheme['stations']],heads=rows,scene_templates=templates,selected_fai=run['result']['selected_fai'],design=run['design'])

class AssemblyService:
 def __init__(self,workflow):self.w=workflow
 def path(self,pid,sid):
  if sid not in ('shared','throughput','domestic'):raise ValueError('Unknown assembly scheme')
  return self.w.store._dir(pid)/('aoi-assembly-'+sid+'.json')
 def save(self,pid,sid,d):
  getattr(self.w.pool,'checkpoint',lambda _pid:None)(pid)
  with self.w.store.lock:
   p=self.path(pid,sid);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False));tmp.replace(p)
 def get(self,pid,sid):
  p=self.path(pid,sid)
  if not p.exists():return None
  d=json.loads(p.read_text());run=self.w.get(pid)['run'];d['stale']=not run or run.get('stale',True) or d['evidence_key']!=evidence_key(run,sid)
  return d
 def submit(self,pid,sid):
  with self.w.store.lock:
   run=self.w.get(pid)['run']
   if not run or run['status']!='complete' or run['stale']:raise ValueError('Complete current workflow required')
   scheme=next(s for s in run['result']['integrated_schemes'] if s['id']==sid)
   current=self.get(pid,sid)
   if current and current['status'] in ('queued','running'):return current
   d=dict(id=uuid.uuid4().hex,status='queued',created_at=now(),evidence_key=evidence_key(run,sid),parent_run_id=run['id'],model_identity=run['model_identity'],scheme_id=sid,progress='等待本地模型总装规划',attempts=[])
   if current and current['parent_run_id']==run['id'] and current.get('model_identity')==run['model_identity']:
    accepted=[a for a in current.get('attempts',[]) if not a.get('truncated') and not a.get('validation_error') and a.get('stage')!='narrative_audit' and a.get('text')]
    if current.get('status')=='complete' and accepted:
     try:
      import re
      d['cached_native_configuration']=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',accepted[-1]['text'].strip()))
      d['cached_native_attempt']=accepted[-1];d['reused_configuration_from']=current['id']
     except (ValueError,TypeError):pass
    valid=[a for a in current.get('attempts',[]) if not a.get('truncated') and a.get('validation_error') and a.get('text')]
    if valid:
     try:
      import re
      d['repair_seed']=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',valid[-1]['text'].strip()))
      d['repair_source_job']=current['id'];d['repair_seed_attempt']=valid[-1]
     except (ValueError,TypeError):pass
   self.save(pid,sid,d);self.w.pool.submit(self.plan,pid,sid,d,run,scheme);return d
 def plan(self,pid,sid,d,run,scheme):
  try:
   d.update(status='running',progress='本地模型分配多相机、合并照明和总装实例');self.save(pid,sid,d)
   heads=run['result']['heads'];groups=groups_for(heads)
   ctx=assembly_context(run,scheme,groups)
   ctx['shared_panel_catalog']=[i for i in library()['items'] if i['kind']=='light' and i['spec'].get('type')=='backlight' and i['spec'].get('active_mm') and i['spec'].get('body_mm')]
   prompt='为Viewer整机总装规划多个物理相机组和光源合并。每个head的完整姿态由scene_template引用scene_templates公共规格，再叠加head.scene得到；所有原始数值完整保留，公共规格不代表共同位置或观察方向。全部group必须分配且只分配给一个camera_unit，并分配且只分配给一个light_unit。尽量使用多个相机并行（2至4组，如几何允许），每组仍可分时移动。home_head_id必须属于组中已有head，挑选互不干涉的不同home（镜头有真实直径），不要挑贴近镜头的布局。station_id必须是源head拥有的工站。同一camera只用同型号镜头。分配不同观察方向或图像方向时，必须明确rotation_mode：移动相机为rotary_mount，移动工件为rotary_fixture；并在motion写清旋转/翻面与标定验证。仅XYZ平移不能看见背面，不能宣称同一固定朝向覆盖正反面。无旋转可不填rotation_mode。光源若同为transmitted_backlight且同向，可选择merged_panel_requirement覆盖多个group，工具按视场并集算需要有效面积；这只是大型发光面需求，未选厂商机械尺寸，不能声称有效或可采购。不能把环光/斜光/反射光硬合成背光，其他光源选择existing_moving_light并给home_head_id。优先合并同向背光为1块候选大光源，少重复光源。不改变原FAI位置。返回完整JSON {"review":"配置取舍与未验证问题","camera_units":[{"id":"CAM1","station_id":"S1","group_ids":["G1"],"home_head_id":"H1","motion_mode":"move_optics","rotation_mode":"rotary_mount","motion":"按原采集姿态移动和转动相机，行程、避障与标定待验证"}],"light_units":[{"id":"LIGHT1","station_id":"S1","group_ids":["G1"],"mode":"merged_panel_requirement"}]}。所有group/ID按源字段，不编造型号。review最多200字，用方案和机械人员熟悉的专业用语说明设备配置、动作和待确认条件，不使用内部字段名，不逐项复述FAI。'
   error=None;mesh=self.w.store.mesh(pid);previous=d.get('repair_seed')
   prompt+=' 如shared_panel_catalog中存在实际发光面积和机身尺寸完整、覆盖合并视场的型号，可以使用mode=catalog_panel并填写item_id；只允许原方向安装。无完整型号则仍用merged_panel_requirement，禁止由型号数字猜尺寸。'
   if previous is not None:
    try:compile_assembly(copy.deepcopy(previous),run,scheme,groups,mesh)
    except (ValueError,TypeError,KeyError) as e:error=str(e)
   for retry in range(5):
    try:
     repair=dict(ctx,previous_configuration=previous,validation_error=error) if previous is not None else ctx
     # Large projects can have dozens of exposure groups. The previous 3,200-token
     # ceiling repeatedly truncated valid camera/light allocations, leaving only
     # the first few groups in the model response. Reserve enough output for both
     # complete ownership lists plus the review; compile_assembly remains the
     # authority that rejects omissions, duplicates, and physically invalid merges.
     if retry==0 and d.get('cached_native_configuration'):
      raw=copy.deepcopy(d['cached_native_configuration']);attempt=copy.deepcopy(d['cached_native_attempt']);attempt['revalidated_by_current_engine']=True
     else:
      raw,attempt=generate_json(self.w.model,prompt+('修正：'+str(error)+'。previous_configuration为上一轮完整配置；只修正报错部分，保留全部拍摄组及有效配置，仍返回完整JSON，不能省略未改变的分组。' if error else ''),repair,10000)
     previous=copy.deepcopy(raw)
     d['attempts'].append(attempt);assembly=compile_assembly(raw,run,scheme,groups,mesh)
     if assembly['collisions'] and retry<4:raise ValueError('Home camera interference; choose different member home heads: '+json.dumps(assembly['collisions'])[:2200])
     break
    except DecisionError as e:d['attempts'].append(e.attempt);error=str(e)
    except (ValueError,TypeError,KeyError) as e:error=str(e);d['attempts'][-1]['validation_error']=error
    d['progress']='本地模型修正总装布局 · '+str(retry+1);self.save(pid,sid,d)
   else:raise ValueError('Assembly validation failed: '+str(error))
   audit,attempt=generate_json(self.w.model,'审校总装方案说明，返回JSON {"review":"最多180字"}。必须以校核事实为准：仅有限包络筛查，不得声称无物理干涉或安全安装；光源根据light_units逐项说明：只有merged_panel_requirement为面积需求；带item_id的是已有候选型号，不能称尚未选型。多相机各组仍需相对运动，未验证并行轨迹或海克斯康方法等效。不得改动配置。',dict(original_review=assembly['review'],camera_count=len(assembly['camera_units']),light_count=len(assembly['light_units']),light_units=[{k:l.get(k) for k in ('id','mode','item_id','status')} for l in assembly['light_units']],collisions=assembly['collisions'],gaps=assembly['gaps']),900)
   if not isinstance(audit.get('review'),str):raise ValueError('Invalid native assembly narrative audit')
   d['attempts'].append(dict(attempt,stage='narrative_audit'));assembly['raw_model_review']=assembly['review'];assembly['review']=audit['review']
   d.update(status='complete',result=assembly,finished_at=now(),progress='总装规划完成 · 安装与光学效果待验证');self.save(pid,sid,d)
  except Exception as e:d.update(status='failed',error=str(e),progress='总装规划失败');self.save(pid,sid,d)
