"""Source-wide local-model station/setup planning with explicit feasibility gaps."""
import copy
from collections import defaultdict
import hashlib
import json
import math
import uuid
from pathlib import Path

from .channel_routes import CHANNELS, propose
from .integrated_schemes import generate_json, compact_catalog_specs, DecisionError, shortlist, shortlist_sensors, table_evidence
from .workflow import compile_tasks, library, now, optical_pair_for_design, planning_evidence
from .placement import face_normals, plan_device_poses

VERSION='station-setup-v3'


class CoverageError(ValueError):
    def __init__(self,issues):
        self.issues=issues
        super().__init__('FAI ownership mismatch: '+json.dumps(issues,ensure_ascii=False,sort_keys=True))


def setup_scope(setup,routes):
    if 'fai_group_ids' not in setup:return setup.get('fai_numbers')
    refs=setup['fai_group_ids']
    allowed=sorted({r['channel'] for r in routes})
    if (not isinstance(refs,list) or not refs or any(not isinstance(ref,str) for ref in refs)
            or len(set(refs))!=len(refs) or any(ref not in allowed for ref in refs)):
        raise ValueError('Invalid source FAI group references in setup '+str(setup.get('id'))+
                         ': received '+str(refs)+'; allowed '+str(allowed))
    expanded=[r['number'] for ref in refs for r in routes if r['channel']==ref]
    explicit=setup.get('fai_numbers')
    if explicit:
        if (not isinstance(explicit,list) or any(type(n) is not int for n in explicit)
                or sorted(explicit)!=sorted(expanded)):
            raise ValueError('Conflicting FAI group and explicit numbers in setup '+str(setup.get('id'))+
                             '; group expands to '+str(expanded)+'; received '+str(explicit))
        return list(explicit)  # identical redundant source encoding, no model decision changed
    return expanded


def normalize_setup_order(raw):
    """Reconcile the display sequence with the setups the model actually emitted.

    Setup ownership and setup contents remain model decisions. The order is only
    a presentation/execution sequence, so discard dangling references and append
    omitted real setups deterministically while preserving the model's valid order.
    """
    repairs=[]
    for station in raw.get('stations',[]) if isinstance(raw,dict) else []:
        if not isinstance(station,dict) or not isinstance(station.get('setups'),list): continue
        ids=[setup.get('id') for setup in station['setups'] if isinstance(setup,dict) and isinstance(setup.get('id'),str)]
        if not ids: continue
        original=station.get('setup_order')
        proposed=original if isinstance(original,list) else []
        seen=set();order=[]
        for setup_id in proposed:
            if setup_id in ids and setup_id not in seen:
                order.append(setup_id);seen.add(setup_id)
        order.extend(setup_id for setup_id in ids if setup_id not in seen)
        if original != order:
            repairs.append(dict(station_id=station.get('id'),received=original,normalized=order,
                                removed_dangling=sum(x not in ids for x in proposed),
                                restored_missing=sum(x not in proposed for x in ids)))
            station['setup_order']=order
    return repairs


def coverage_issues(raw,tasks,routes):
    owners=defaultdict(list);known={t['number'] for t in tasks}
    for station in raw.get('stations',[]):
        for setup in station.get('setups',[]):
            numbers=setup_scope(setup,routes)
            if not isinstance(numbers,list) or any(type(n) is not int or n not in known for n in numbers):
                raise ValueError('Unknown or invalid source FAI in setup')
            for number in numbers:owners[number].append(setup.get('id'))
    return dict(missing=sorted(known-set(owners)),duplicates={str(n):ids for n,ids in sorted(owners.items()) if len(ids)>1})


def apply_ownership(raw,tasks,routes,assignments):
    """Compile explicit local-model choices; never choose an owner ourselves."""
    issues=coverage_issues(raw,tasks,routes)
    affected=set(issues['missing'])|{int(n) for n in issues['duplicates']}
    methods={r['number']:r['channel'] for r in routes}
    setups={u['id']:u for st in raw['stations'] for u in st['setups']}
    if len(setups)!=sum(len(st['setups']) for st in raw['stations']):raise ValueError('Duplicate setup IDs cannot be repaired by ownership assignment')
    if (not isinstance(assignments,list) or any(not isinstance(a,dict) or type(a.get('number')) is not int for a in assignments)
            or sorted(a['number'] for a in assignments)!=sorted(affected)):
        raise ValueError('Ownership repair must explicitly decide every affected FAI exactly once')
    for row in assignments:
        uid=row.get('setup_id')
        if uid not in setups or setups[uid]['channel']!=methods[row['number']]:
            raise ValueError('Ownership repair selected unknown or incompatible setup')
    result=copy.deepcopy(raw)
    for st in result['stations']:
        for setup in st['setups']:
            setup['fai_numbers']=[n for n in setup_scope(setup,routes) if n not in affected]
            setup.pop('fai_group_ids',None)
    target={u['id']:u for st in result['stations'] for u in st['setups']}
    for row in assignments:target[row['setup_id']]['fai_numbers'].append(row['number'])
    removed=[]
    for st in result['stations']:
        empty={u['id'] for u in st['setups'] if not u['fai_numbers']};removed.extend(sorted(empty))
        st['setups']=[u for u in st['setups'] if u['id'] not in empty]
        st['setup_order']=[uid for uid in st['setup_order'] if uid not in empty]
    result['stations']=[st for st in result['stations'] if st['setups']]
    return result,dict(issues=issues,assignments=assignments,removed_empty_setup_ids=removed)


def repair_duplicate_groups(model,raw,tasks,routes,attempts,progress,issues,setups):
    """Whole duplicate groups: model picks owners, compiler expands source IDs."""
    if issues['missing']:return None
    affected={int(n) for n in issues['duplicates']}
    groups={channel:[r['number'] for r in routes if r['channel']==channel] for channel in sorted({r['channel'] for r in routes})}
    if affected!={t['number'] for t in tasks}:return None
    choices={channel:[u['id'] for u in setups if u['channel']==channel and sorted(u['fai_numbers'])==sorted(numbers)]
             for channel,numbers in groups.items()}
    if any(not ids for ids in choices.values()):return None
    evidence=dict(source_groups=groups,allowed_setup_ids=choices,setups=setups,method_routes=routes,
        measurement_regions=[{k:t.get(k) for k in ('number','features','symbols','specifications','probes','anchors','location_status','pages')} for t in tasks])
    if len(json.dumps(table_evidence(evidence),ensure_ascii=False,separators=(',',':')))>63000:return None
    prompt='本地工站方案把每个完整测量通道组重复分配给了多个setup。请依据所有FAI的方法路由、测量区域及setup的装夹/运动安排，为source_groups的每个channel从allowed_setup_ids中选择唯一结果归属setup。原方法路由和设备配置不能更改；这是重复归属消歧，不是重新证明物理可行或计量通过。所有通道恰好一条，不能遗漏或重复。返回纯JSON {"group_assignments":[{"channel":"3d_surface","setup_id":"U1"}]}。编号无需重写，工具会无损展开原组完整编号；不输出思考过程。'
    error=None
    for retry in range(3):
        progress(retry,3,'本地模型核对重复测量组并确定最终归属')
        try:
            decision,attempt=generate_json(model,prompt+(f' 修正错误：{error}' if error else ''),evidence,1600)
            attempt['stage']='ownership_group_repair';attempts.append(attempt)
            rows=decision.get('group_assignments')
            if (not isinstance(rows,list) or any(not isinstance(row,dict) or not isinstance(row.get('channel'),str) for row in rows)
                    or sorted(row['channel'] for row in rows)!=sorted(groups)):
                raise ValueError('Every source group needs exactly one model ownership decision')
            assignments=[]
            for row in rows:
                if row.get('setup_id') not in choices[row['channel']]:raise ValueError('Group owner must be in allowed_setup_ids')
                assignments.extend(dict(number=n,setup_id=row['setup_id']) for n in groups[row['channel']])
            fixed,receipt=apply_ownership(raw,tasks,routes,assignments)
            receipt.update(model_group_assignments=rows,decision_encoding='complete_source_group_references')
            return fixed,receipt
        except DecisionError as exc:
            exc.attempt.update(stage='ownership_group_repair',validation_error=str(exc));attempts.append(exc.attempt);error=str(exc)
        except (ValueError,KeyError,TypeError) as exc:
            error=str(exc)
            if attempts and attempts[-1].get('stage')=='ownership_group_repair':attempts[-1]['validation_error']=error
    return None  # bounded group repair exhausted; retain errors and try per-FAI local decisions


def repair_ownership(model,raw,tasks,routes,attempts,progress):
    issues=coverage_issues(raw,tasks,routes)
    affected=set(issues['missing'])|{int(n) for n in issues['duplicates']}
    if not affected:return raw,None
    methods={r['number']:r for r in routes}
    setups=[dict(station_id=st['id'],station_name=st['name'],**{k:u.get(k) for k in
                 ('id','channel','device_ids','fixture_state','motion','sequence')},fai_numbers=setup_scope(u,routes))
            for st in raw['stations'] for u in st['setups']]
    grouped=repair_duplicate_groups(model,raw,tasks,routes,attempts,progress,issues,setups)
    if grouped is not None:return grouped
    for n in affected:
        if not any(u['channel']==methods[n]['channel'] for u in setups):
            raise ValueError(f'FAI {n} has no compatible setup; full local plan needs regeneration')
    prompt='修正本地模型工站方案中的FAI结果归属。仅决定本批affected_tasks中的漏项和重复项；正常归属不能改动。读取这些FAI的完整MSOP方法、基准和当前setup的装夹/动作安排，为每项仅从allowed_assignments给出的setup_ids选择一个现有且channel完全一致的setup。此阶段不能修改测量通道，跨坐标系测量也不能擅自把3d_surface改为hybrid_metrology。位置未确定仍须保留方案归属和验证缺口，不得据此声称安装或计量已通过。重复项必须选一个最终归属；编号不能省略、重复或新造。返回纯JSON {"assignments":[{"number":1,"setup_id":"U1"}]}，本批每个受影响FAI恰好一条。不重写工站方案、不输出思考过程。'
    ordered=sorted(affected);assignments=[];offset=0
    while offset<len(ordered):
        count=min(6,len(ordered)-offset)
        while True:
            batch=ordered[offset:offset+count]
            evidence=dict(ownership_errors=dict(missing=[n for n in issues['missing'] if n in batch],
                duplicates={n:owners for n,owners in issues['duplicates'].items() if int(n) in batch}),
                allowed_assignments=[dict(number=n,channel=methods[n]['channel'],setup_ids=[u['id'] for u in setups if u['channel']==methods[n]['channel']]) for n in batch],
                setups=[u for u in setups if u['channel'] in {methods[n]['channel'] for n in batch}],
                affected_tasks=[t for t in tasks if t['number'] in batch],affected_methods=[methods[n] for n in batch])
            if len(json.dumps(table_evidence(evidence),ensure_ascii=False,separators=(',',':')))<=63000:break
            if count==1:raise ValueError('Single ownership repair method exceeds context budget; no source truncated')
            count=max(1,count//2)
        error=None
        for retry in range(3):
            progress(offset,len(ordered),'本地模型修正FAI归属 · '+', '.join(str(n) for n in batch)+f'（尝试 {retry+1}/3）')
            try:
                decision,attempt=generate_json(model,prompt+(f' 上次修正错误：{error}' if error else ''),evidence,1600)
                attempt.update(stage='ownership_repair',numbers=batch);attempts.append(attempt)
                rows=decision.get('assignments')
                if (not isinstance(rows,list) or any(not isinstance(a,dict) or type(a.get('number')) is not int for a in rows)
                        or sorted(a['number'] for a in rows)!=batch):
                    raise ValueError('Ownership repair must cover this batch exactly once')
                choices={u['id']:u for u in setups}
                for row in rows:
                    uid=row.get('setup_id')
                    if not isinstance(uid,str) or uid not in choices or choices[uid]['channel']!=methods[row['number']]['channel']:
                        raise ValueError('FAI '+str(row['number'])+' selected '+str(uid)+' but requires '+methods[row['number']]['channel']+'; allowed setup IDs: '+str([u['id'] for u in setups if u['channel']==methods[row['number']]['channel']]))
                assignments.extend(rows);break
            except DecisionError as exc:
                exc.attempt.update(stage='ownership_repair',numbers=batch,validation_error=str(exc));attempts.append(exc.attempt);error=str(exc)
            except (ValueError,KeyError,TypeError) as exc:
                error=str(exc)
                if attempts and attempts[-1].get('stage')=='ownership_repair':attempts[-1]['validation_error']=error
        else:raise ValueError('Local ownership repair failed: '+str(error))
        offset+=count
    return apply_ownership(raw,tasks,routes,assignments)


def validate(raw,tasks,routes,cat,constraints,mesh_file=None):
    source_raw=raw
    raw=copy.deepcopy(raw)
    stations=raw.get('stations');items={i['id']:i for i in cat['items']}
    known={t['number']:t for t in tasks};methods={r['number']:r for r in routes}
    wanted_faces={a.get('face_id') for t in tasks for a in t.get('anchors',[]) if a.get('face_id')}
    pose_normals,mesh_center=face_normals(mesh_file,wanted_faces)
    if not isinstance(stations,list) or not 1<=len(stations)<=32:
        raise ValueError('Need 1..32 stations')
    if constraints.get('max_stations') and len(stations)>constraints['max_stations']:
        raise ValueError('Station count exceeds design constraint')
    station_ids=[];setup_ids=[];assigned=[];bom=[];instances={}
    for station in stations:
        sid=station.get('id');station_ids.append(sid)
        if not isinstance(sid,str) or not sid or not isinstance(station.get('name'),str):
            raise ValueError('Invalid station identity')
        setups=station.get('setups')
        if not isinstance(setups,list) or not setups:raise ValueError('Every station needs setups')
        for setup in setups:
            setup['fai_numbers']=setup_scope(setup,routes)
            uid=setup.get('id');setup_ids.append(uid);numbers=setup.get('fai_numbers')
            if not isinstance(uid,str) or not uid:raise ValueError('Invalid setup identity')
            if not isinstance(numbers,list) or not numbers:raise ValueError('Setup '+uid+' has no source FAI; do not create empty setups')
            if any(type(n) is not int or n not in known for n in numbers):
                raise ValueError('Unknown source FAI in setup')
            if len(set(numbers))!=len(numbers):raise CoverageError(coverage_issues(source_raw,tasks,routes))
            channel=setup.get('channel')
            if channel not in CHANNELS:raise ValueError('Unknown measurement channel')
            if any(methods[n]['channel']!=channel for n in numbers):
                raise ValueError('Setup '+uid+' channel '+str(channel)+' conflicts with source routes '+
                                 str({n:methods[n]['channel'] for n in numbers if methods[n]['channel']!=channel}))
            for field in ('fixture_state','motion','sequence'):
                if not isinstance(setup.get(field),str) or not setup[field].strip():
                    raise ValueError('Setup needs fixture state, motion and acquisition sequence')
            devices=setup.get('device_ids')
            if not isinstance(devices,list) or len(set(devices))!=len(devices) or any(d not in items for d in devices):
                raise ValueError('Unknown or duplicate catalogue device')
            kinds={items[d]['kind'] for d in devices};gaps=[]
            if channel=='2d_vision':
                cams=[items[d] for d in devices if items[d]['kind']=='camera']
                lenses=[items[d] for d in devices if items[d]['kind']=='lens']
                if len(cams)!=1 or len(lenses)!=1 or 'light' not in kinds:
                    gaps.append('二维 setup 尚缺相机、镜头或照明选型')
                else:
                    check=optical_pair_for_design(cams[0],lenses[0],constraints)
                    if not check['eligible']:raise ValueError('Camera/lens specification mismatch: '+str(check['reasons']))
                    setup['optical_pair']=check
            if channel in ('3d_surface','hybrid_metrology') and 'surface_sensor' not in kinds:
                gaps.append('尚缺三维测量传感器选型')
            if channel in ('confocal_height','hybrid_metrology') and 'confocal_sensor' not in kinds:
                gaps.append('尚缺共焦测量通道选型')
            regions=[]
            for n in numbers:
                task=known[n]
                if not task['anchors']:gaps.append(f'FAI {n} 没有可用测量位置')
                regions.append(dict(fai=n,face_ids=sorted({a['face_id'] for a in task['anchors'] if a.get('face_id')}),
                    points_mm=[a['position'] for a in task['anchors']],location_status=task['location_status'],
                    source_pages=task['pages'],specifications=task['specifications']))
            setup.update(station_id=sid,measurement_regions=regions,location_covered_fai=[n for n in numbers if known[n]['anchors']],
                measurement_requirements=[dict(fai=n,specifications=known[n]['specifications']) for n in numbers],
                accuracy_budget_status='per_FAI_uncertainty_budget_required',
                gaps=gaps+['夹具遮挡、装夹变形及实际工件基准需要验证','设备安装位姿、观察光路和动作轨迹尚未验证',
                           '采样间距不等于准确度；需逐 FAI 分配系统误差预算'],
                installation_status='unverified',metrology_accepted=False)
            setup['device_placement']=plan_device_poses(setup,[known[n] for n in numbers],list(items.values()),pose_normals,mesh_center)
            if setup['device_placement']['status']!='proposed_unvalidated':
                setup['gaps'].append('设备位姿未解算：'+setup['device_placement'].get('reason','CAD面法向或设备工作距离/FOV资料不足'))
            else:
                setup['gaps'].append('设备观察位姿已按CAD面法向生成候选；安装基准与夹具避碰仍待验证')
            # A setup is a hardware instance. Identical models in two setups
            # count twice; sharing must later identify an actual common unit.
            for device in devices:
                references=setup.get('device_instance_ids') or {}
                if not isinstance(references,dict) or set(references)-set(devices):
                    raise ValueError('Invalid setup hardware instance references')
                instance=references.get(device,uid+':'+device)
                if not isinstance(instance,str) or not instance or len(instance)>200:
                    raise ValueError('Invalid hardware instance ID')
                existing=instances.get(instance)
                if existing:
                    if existing['station_id']!=sid or existing['id']!=device:
                        raise ValueError('Hardware instance cannot cross stations or mix models')
                    existing['setup_ids'].append(uid)
                else:
                    row=dict(id=device,instance_id=instance,station_id=sid,setup_ids=[uid],
                        quantity=1,kind=items[device]['kind'],source_url=items[device]['source_url'],status='conditional_candidate')
                    instances[instance]=row;bom.append(row)
            assigned.extend(numbers)
        order=station.get('setup_order')
        ids=[s['id'] for s in setups]
        if not isinstance(order,list) or sorted(order)!=sorted(ids):raise ValueError('Station '+sid+' setup_order must list exactly '+str(ids)+' once; received '+str(order))
        station.update(ct=cycle_time(ids,constraints),physical_status='unverified',
            fixture_status='fixture_geometry_required',execution='sequential_candidate_order',
            parallel_execution_status='not_verified',fai_numbers=sorted({n for s in setups for n in s['fai_numbers']}))
    if len(set(station_ids))!=len(station_ids) or len(set(setup_ids))!=len(setup_ids):
        raise ValueError('Station and setup IDs must be unique')
    if sorted(assigned)!=sorted(known):raise CoverageError(coverage_issues(source_raw,tasks,routes))
    if set(constraints.get('setup_cycle_s') or {})-set(setup_ids):
        raise ValueError('Cycle timing references unknown setup IDs')
    unlocated=[n for n,t in known.items() if not t['anchors']]
    pose_count=sum(len(u.get('device_placement',{}).get('poses',[])) for st in stations for u in st['setups'])
    posed_setup_count=sum(u.get('device_placement',{}).get('status')=='proposed_unvalidated' for st in stations for u in st['setups'])
    return dict(version=VERSION,stations=stations,setup_count=len(setup_ids),station_count=len(stations),bom=bom,
        coverage=dict(source_total=len(known),assigned=len(assigned),located=len(known)-len(unlocated),
                      physically_validated=0,metrology_accepted=0,unlocated_fai=unlocated),
        placement=dict(proposed_device_poses=pose_count,pose_computed_setups=posed_setup_count),
        line_ct_s=None,line_ct_status='transfer_and_parallel_schedule_unverified',
        status='conditional_station_plan',ready_for_installation=False,review=raw.get('review',''))


def cycle_time(setup_ids,constraints):
    durations=constraints.get('setup_cycle_s') or {};overhead=constraints.get('station_overhead_s')
    if any(uid not in durations for uid in setup_ids) or overhead is None:
        return dict(total_s=None,target_s=constraints.get('target_ct_s'),status='timing_inputs_missing',
                    components=['上下料/定位夹紧','运动/转向及稳定等待','曝光/扫描','计算及结果交接'])
    values=[durations[uid] for uid in setup_ids]+[overhead]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError('Cycle durations must be finite nonnegative seconds')
    total=sum(values);target=constraints.get('target_ct_s')
    return dict(total_s=total,target_s=target,status='target_exceeded' if target is not None and total>target else 'estimated_unvalidated',
                authority='user_supplied_complete_setup_durations_sequential_sum')


class StationService:
    def __init__(self,store,model,pool):self.store,self.model,self.pool=store,model,pool
    def path(self,pid):return self.store._dir(pid)/'station-workflow.json'
    def save(self,pid,record):
        getattr(self.pool,'checkpoint',lambda _pid:None)(pid)
        with self.store.lock:
            path=self.path(pid);temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False));temp.replace(path)
    def source_key(self,pid):
        p=self.store.get(pid)
        from .workflow import planning_evidence
        return hashlib.sha256(json.dumps(dict(version=VERSION,engine=Path(__file__).read_text(),placement_engine=Path(__file__).with_name('placement.py').read_text(),
            route_engine=Path(__file__).with_name('channel_routes.py').read_text(),catalogue=library(),
            inference={k:self.model.info().get(k) for k in ('model_alias','inference_backend','base_model_path','adapter_dir')},
            documents=[{k:d.get(k) for k in ('sha256','role')} for d in p['documents']],
            model=p.get('model'),annotations=planning_evidence(p.get('annotations')),bindings=p.get('bindings')),
            sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    def get(self,pid):
        path=self.path(pid)
        if not path.exists():return None
        record=json.loads(path.read_text());record['stale']=record['source_key']!=self.source_key(pid)
        return record
    def submit(self,pid,constraints):
        with self.store.lock:
            prior=self.get(pid)
            if prior and prior['status'] in ('queued','running'):return prior
            record=dict(id=uuid.uuid4().hex,status='queued',created_at=now(),source_key=self.source_key(pid),
                constraints=constraints,progress=dict(done=0,total=0,message='等待本地模型'),attempts=[],routes=[])
            if prior and prior.get('routes') and prior.get('source_summary',{}).get('documents')==[
                    dict(role=d['role'],sha256=d['sha256'],filename=d['filename']) for d in self.store.get(pid)['documents']]:
                tasks=compile_tasks(self.store,self.store.get(pid))
                prior_identity=prior.get('model_identity')
                identity=self.model.info()
                same_model=bool(prior_identity) and all(prior_identity.get(k)==identity.get(k) for k in ('model_alias','inference_backend','base_model_path','adapter_dir'))
                same_tasks=planning_evidence(prior.get('task_snapshot'))==planning_evidence(tasks)
                same_route_engine=prior.get('route_engine_sha256')==hashlib.sha256(Path(__file__).with_name('channel_routes.py').read_bytes()).hexdigest()
                if same_model and same_tasks and same_route_engine and sorted(r.get('number') for r in prior['routes'])==sorted(t['number'] for t in tasks) and all(r.get('channel') in CHANNELS for r in prior['routes']):
                    record.update(cached_routes=prior['routes'],cached_route_attempts=[a for a in prior['attempts'] if 'encoded_chars' not in a and 'numbers' in a and 'input_sha256' in a],
                                  reused_routes_from=prior['id'])
                    if prior.get('status')=='failed' and (prior.get('raw_model_plan') or prior.get('repairable_model_plan')) and prior.get('constraints')==constraints:
                        # Revalidate the model's original complete decision first. The ownership
                        # repair draft may be an incomplete intermediate and can lose setup scope.
                        record['prior_failed_plan']=(prior.get('raw_model_plan') or prior.get('repairable_model_plan')
                                                     or prior.get('prior_failed_plan'))
            self.save(pid,record);self.pool.submit(self.run,pid,record);return record
    def run(self,pid,record):
        try:
            record['status']='running';record['model_identity']=self.model.info()
            tasks=compile_tasks(self.store,self.store.get(pid));cat=library()
            record['task_snapshot']=tasks
            record['route_engine_sha256']=hashlib.sha256(Path(__file__).with_name('channel_routes.py').read_bytes()).hexdigest()
            record['source_summary']=dict(total=len(tasks),located=sum(bool(t['anchors']) for t in tasks),
                documents=[dict(role=d['role'],sha256=d['sha256'],filename=d['filename']) for d in self.store.get(pid)['documents']])
            def progress(done,total,message,*partial):
                if partial:record['routes']=partial[0]
                record['progress']=dict(done=done,total=total,message=message);self.save(pid,record)
            routes,attempts=propose(self.model,tasks,progress,prior_routes=record.get('cached_routes'),
                                    prior_attempts=record.get('cached_route_attempts'))
            record.update(routes=routes,attempts=attempts);self.save(pid,record)
            # Lossless method proposals plus geometrical region identities;
            # raw MSOP methods were reviewed in bounded batches above.
            progress(0,1,'本地模型选择二维设备候选；三维通道按目录保留')
            def hardware_progress(values):
                record['hardware_model_attempts']=values
                retry=values[-1].get('retry',0)
                progress(retry+1,3,'本地模型设备候选已通过校验' if not values[-1].get('validation_error') else f'本地模型正在修正设备候选 · 第 {retry+1}/3 次校验未通过')
            choice,attempt=shortlist(self.model,cat,record['constraints'],hardware_progress)
            record['attempts'].append(dict(attempt,stage='hardware_shortlist'))
            chosen=set(choice['light_ids'])|set(choice['pair_ids'][0].split('|'))
            def sensor_attempts(values):
                record['sensor_model_attempts']=values
                self.save(pid,record)
            sensor_ids,sensor_reviews=shortlist_sensors(self.model,cat,routes,record['constraints'],progress,sensor_attempts)
            record.update(sensor_shortlist_ids=sensor_ids,sensor_model_attempts=sensor_reviews)
            chosen.update(sensor_ids)
            devices=[dict(id=i['id'],kind=i['kind'],vendor=i['vendor'],spec=i['spec']) for i in cat['items']
                     if i['id'] in chosen]
            devices,templates=compact_catalog_specs(devices)
            def region(task):
                points=[a['position'] for a in task['anchors']]
                return dict(number=task['number'],point_count=len(points),
                    face_ids=sorted({a['face_id'] for a in task['anchors'] if a.get('face_id')}),
                    bounds_mm=[[min(p[k] for p in points) for k in range(3)],
                               [max(p[k] for p in points) for k in range(3)]] if points else None,
                    location_status=task['location_status'],specifications=task['specifications'])
            context=dict(method_routes=routes,source_fai_groups={channel:[r['number'] for r in routes if r['channel']==channel] for channel in sorted({r['channel'] for r in routes})},tasks=[region(t) for t in tasks],
                catalogue=devices,spec_templates=templates,constraints=record['constraints'])
            prompt='规划完整 station/setup 方案。一个 station 可以有多组独立2D/3D setup；setup不是工站，也不是一次曝光。同一setup可含多个FAI，但必须同一channel；每个FAI恰好一个结果归属setup，未定位FAI也必须列入并保留位置缺口。所有setup编号在全部工站中全局唯一，不能每站重新从U1编号。硬件只从目录选ID，2d_vision需camera/lens/light，3d_surface需surface_sensor，共焦需confocal_sensor。优先合并相容的测量区域，但不同姿态必须明确运动、翻面与实际基准重建。无工装CAD、设备安装位姿或动作时间不能声称物理可行、CT达标或计量通过。时序先给候选串行顺序，不能凭空声称可并行。可提出多组独立setup放在一个station。返回完整JSON {"review":"方案取舍","stations":[{"id":"S1","name":"工站名称","setup_order":["U1"],"setups":[{"id":"U1","channel":"3d_surface","fai_group_ids":["3d_surface"],"device_ids":["目录ID"],"fixture_state":"定位与夹持姿态","motion":"设备与工件动作","sequence":"采集与计算顺序"}]}]}。不能省略FAI，不编造精度、姿态或动作时长。最多8个工站、每站最多8组setup。不输出空setup或空站。review不超过200字，装夹、动作和顺序每字段不超过120字，使用中文工程语言。'
            error=None;previous=None;seed=record.get('prior_failed_plan')
            prompt+=' 同工站内同型号可明确填写setup.device_instance_ids（目录型号到物理实例ID的对象）共用设备，动作顺序须包含切换与重定位。同型号不自动合并；不同工站不能共用同一物理实例。'
            prompt+=' 优先用fai_group_ids引用source_fai_groups（如["3d_surface"]）而不是手抄编号列表；channel必须与该组名完全一致。引用组会由工具无损展开完整编号。若要把一个组拆分到多个setup，可填写fai_numbers，但不能重复或遗漏；请只选一种写法；同时填写时两份编号必须完全一致，不允许分组与编号冲突。'
            prompt+=' 用户目标：在精度、镜头视野、景深、可见性和安装约束满足的前提下，尽量让同一个FAI在一次完整视野内采集，优先扩大有效视野、减少拆拍。不要为合并其他FAI而拆散可单视野完成的FAI。不同面朝向或遮挡必须分次观察；没有逐点投影证据不能承诺单视野覆盖，不能把多视角测量冒充图像拼接，不得用未经标定的拼图代替严格MSOP测量。'
            for retry in range(3):
                progress(retry,3,'本地模型分配 station / setup 并核对全项目覆盖')
                try:
                    repair_context=dict(context)
                    if previous is not None and len(json.dumps(context,ensure_ascii=False))+len(json.dumps(previous,ensure_ascii=False))<63000:
                        repair_context['previous_invalid_plan']=previous
                    if retry==0 and seed is not None:
                        raw=copy.deepcopy(seed)
                        record['attempts'].append(dict(stage='failed_plan_revalidation',source_job=record['reused_routes_from']))
                    else:
                        raw,attempt=generate_json(self.model,prompt+(f' 修正校验错误：{error}' if error else ''),repair_context,6000)
                        record['attempts'].append(attempt)
                    previous=copy.deepcopy(raw)
                    record['raw_model_plan']=copy.deepcopy(raw)
                    structural_repairs=normalize_setup_order(raw)
                    if structural_repairs:
                        record['attempts'][-1]['structural_repairs']=structural_repairs
                    try:result=validate(raw,tasks,routes,cat,record['constraints'],(self.store.get(pid).get('model') or {}).get('mesh_file'))
                    except CoverageError as exc:
                        record['attempts'][-1]['validation_error']=str(exc)
                        record['ownership_issues']=exc.issues
                        record['repairable_model_plan']=copy.deepcopy(raw)
                        progress(0,3,'检测到漏项或重复归属，正在交由本地模型修正')
                        fixed,receipt=repair_ownership(self.model,raw,tasks,routes,record['attempts'],progress)
                        result=validate(fixed,tasks,routes,cat,record['constraints'],(self.store.get(pid).get('model') or {}).get('mesh_file'))
                        record['ownership_repair']=receipt;record['raw_model_plan']=fixed
                    break
                except DecisionError as exc:
                    exc.attempt['validation_error']=str(exc);record['attempts'].append(exc.attempt);error=str(exc)
                except (ValueError,KeyError,TypeError) as exc:
                    error=str(exc)
                    record['attempts'].append(dict(stage='station_planning_validation',retry=retry,validation_error=error))
            else:raise ValueError('Station/setup validation failed: '+str(error))
            record.update(status='complete',result=result,finished_at=now(),progress=dict(done=len(tasks),total=len(tasks),message='工站方案已生成；安装、工装与节拍待验证'))
        except Exception as exc:
            record.update(status='failed',error=str(exc),finished_at=now())
            record['progress']=dict(record.get('progress') or {},message='方案未通过校验：'+str(exc))
        self.save(pid,record)
