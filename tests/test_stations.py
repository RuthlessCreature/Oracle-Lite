import copy
import pytest
from oracle_viewer.stations import validate,cycle_time,normalize_setup_order
from oracle_viewer.workflow import library


def fixture():
    tasks=[dict(number=n,anchors=[dict(face_id='face',position=[n,0,0])] if n!=3 else [],
                location_status='pending_review',pages=[n],specifications=[]) for n in (1,2,3)]
    routes=[dict(number=n,channel='3d_surface') for n in (1,2,3)]
    def setup(id,scope):return dict(id=id,channel='3d_surface',fai_numbers=scope,device_ids=['MV-DP3020-01H'],
                                  fixture_state='实际基准定位',motion='扫描移动待验证',sequence='定位、采集、拟合')
    raw=dict(review='多组独立 setup',stations=[dict(id='S1',name='站一',setup_order=['U1','U2'],
                                                 setups=[setup('U1',[1]),setup('U2',[2,3])])])
    return tasks,routes,raw


def test_normalize_setup_order_removes_cross_station_dangling_ids_and_keeps_real_order():
    raw={'stations':[
        {'id':'S1','setup_order':['U1','U2','U3','U4','U5','U6','U7','U8'],
         'setups':[{'id':f'U{i}'} for i in range(1,6)]},
        {'id':'S2','setup_order':['U6','U7','U8'],
         'setups':[{'id':f'U{i}'} for i in range(6,9)]},
    ]}
    repairs=normalize_setup_order(raw)
    assert raw['stations'][0]['setup_order']==['U1','U2','U3','U4','U5']
    assert raw['stations'][1]['setup_order']==['U6','U7','U8']
    assert repairs[0]['removed_dangling']==3
    assert len(repairs)==1


def test_one_station_has_multiple_setups_and_identical_devices_count_twice():
    tasks,routes,raw=fixture();r=validate(raw,tasks,routes,library(),{})
    assert r['station_count']==1 and r['setup_count']==2
    assert len(r['bom'])==2 and len({b['instance_id'] for b in r['bom']})==2
    assert r['coverage']==dict(source_total=3,assigned=3,located=2,physically_validated=0,metrology_accepted=0,unlocated_fai=[3])
    assert r['stations'][0]['ct']['total_s'] is None
    assert not r['ready_for_installation']


@pytest.mark.parametrize('failure',['missing','duplicate','wrong_route','unknown_hardware','bad_order'])
def test_native_station_decisions_reject_scope_and_resource_errors(failure):
    tasks,routes,raw=fixture();s=raw['stations'][0]
    if failure=='missing':s['setups'][1]['fai_numbers']=[2]
    if failure=='duplicate':s['setups'][1]['fai_numbers']=[1,2,3]
    if failure=='wrong_route':s['setups'][0]['channel']='2d_vision'
    if failure=='unknown_hardware':s['setups'][0]['device_ids']=['invented']
    if failure=='bad_order':s['setup_order']=['U1']
    with pytest.raises(ValueError):validate(raw,tasks,routes,library(),{})


def test_ct_checks_require_all_setup_durations_and_station_overhead():
    assert cycle_time(['U1','U2'],dict(target_ct_s=10,setup_cycle_s={'U1':2},station_overhead_s=1))['total_s'] is None
    r=cycle_time(['U1','U2'],dict(target_ct_s=10,setup_cycle_s={'U1':4,'U2':5},station_overhead_s=2))
    assert r['total_s']==11 and r['status']=='target_exceeded'
    with pytest.raises(ValueError):cycle_time(['U1'],dict(setup_cycle_s={'U1':float('nan')},station_overhead_s=0))


def test_partition_into_two_stations_keeps_distinct_instance_owners():
    tasks,routes,raw=fixture();st=raw['stations'][0];second=copy.deepcopy(st)
    second.update(id='S2',setups=[st['setups'][1]],setup_order=['U2'])
    st.update(setups=[st['setups'][0]],setup_order=['U1']);raw['stations'].append(second)
    with pytest.raises(ValueError,match='Station count'):validate(copy.deepcopy(raw),tasks,routes,library(),dict(max_stations=1))
    result=validate(raw,tasks,routes,library(),{})
    assert result['station_count']==2 and {b['station_id'] for b in result['bom']}=={'S1','S2'}


def test_explicit_same_station_device_sharing_is_counted_once():
    tasks,routes,raw=fixture()
    for s in raw['stations'][0]['setups']:s['device_instance_ids']={'MV-DP3020-01H':'DEPTH1'}
    result=validate(raw,tasks,routes,library(),{})
    assert len(result['bom'])==1 and result['bom'][0]['setup_ids']==['U1','U2']
    tasks,routes,raw=fixture();st=raw['stations'][0];second=copy.deepcopy(st)
    second.update(id='S2',setups=[st['setups'][1]],setup_order=['U2']);st.update(setups=[st['setups'][0]],setup_order=['U1']);raw['stations'].append(second)
    for station in raw['stations']:
        station['setups'][0]['device_instance_ids']={'MV-DP3020-01H':'DEPTH1'}
    with pytest.raises(ValueError,match='cross stations'):validate(raw,tasks,routes,library(),{})


def test_source_group_reference_expands_losslessly_and_rejects_duplicates():
    tasks,routes,raw=fixture();station=raw['stations'][0]
    setup=station['setups'][0];setup.pop('fai_numbers');setup['fai_group_ids']=['3d_surface']
    station.update(setups=[setup],setup_order=['U1'])
    r=validate(copy.deepcopy(raw),tasks,routes,library(),{})
    assert r['stations'][0]['setups'][0]['fai_numbers']==[1,2,3]
    setup['fai_group_ids']=['3d_surface','3d_surface']
    with pytest.raises(ValueError,match='group references'):validate(raw,tasks,routes,library(),{})


@pytest.mark.parametrize('changed',['none','identity','source_task','route_engine'])
def test_route_cache_requires_exact_source_tasks_model_and_route_engine(tmp_path,monkeypatch,changed):
    import hashlib,threading
    from pathlib import Path
    from oracle_viewer.stations import StationService
    import oracle_viewer.stations as module
    tasks,routes,_=fixture()
    class Store:
        root=tmp_path
        lock=threading.RLock()
        def _dir(self,pid):return tmp_path
        def get(self,pid):return dict(documents=[dict(role='msop',sha256='immutable-source',filename='source.pdf')])
    class Model:
        def info(self):return dict(model_alias='local-test',inference_backend='strata-native')
    class Pool:
        def submit(self,*args):pass
    monkeypatch.setattr(module,'compile_tasks',lambda *_:tasks)
    service=StationService(Store(),Model(),Pool())
    snapshot=copy.deepcopy(tasks)
    if changed=='source_task':snapshot[0]['specifications']=['different tolerance']
    identity=Model().info()
    if changed=='identity':identity=None
    record=dict(id='prior',status='complete',source_key=service.source_key('p'),routes=routes,attempts=[],
                source_summary=dict(documents=Store().get('p')['documents']),task_snapshot=snapshot,model_identity=identity,
                route_engine_sha256=hashlib.sha256(Path(module.__file__).with_name('channel_routes.py').read_bytes()).hexdigest())
    if changed=='route_engine':record['route_engine_sha256']='different'
    service.save('p',record)
    new=service.submit('p',{})
    assert bool(new.get('cached_routes'))==(changed=='none')


def test_coverage_diagnostics_and_local_ownership_choices_preserve_other_assignments():
    from oracle_viewer.stations import CoverageError,apply_ownership
    tasks,routes,raw=fixture()
    raw['stations'][0]['setups'][1]['fai_numbers']=[1,2]
    original=copy.deepcopy(raw)
    with pytest.raises(CoverageError) as exc:validate(raw,tasks,routes,library(),{})
    assert exc.value.issues==dict(missing=[3],duplicates={'1':['U1','U2']})
    assert raw==original  # validation must not contaminate the next model input
    fixed,receipt=apply_ownership(raw,tasks,routes,[dict(number=1,setup_id='U1'),dict(number=3,setup_id='U2')])
    assert fixed['stations'][0]['setups'][1]['fai_numbers']==[2,3]
    assert validate(fixed,tasks,routes,library(),{})['coverage']['assigned']==3
    assert receipt['assignments'][0]['setup_id']=='U1'
    with pytest.raises(ValueError):apply_ownership(raw,tasks,routes,[dict(number=3,setup_id='U2')])
    with pytest.raises(ValueError):apply_ownership(raw,tasks,routes,[dict(number=1,setup_id='unknown'),dict(number=3,setup_id='U2')])


def test_group_overlap_reports_precise_owners_without_mutating_native_plan():
    from oracle_viewer.stations import CoverageError,apply_ownership
    tasks,routes,raw=fixture()
    for u in raw['stations'][0]['setups']:
        u.pop('fai_numbers');u['fai_group_ids']=['3d_surface']
    original=copy.deepcopy(raw)
    with pytest.raises(CoverageError) as exc:validate(raw,tasks,routes,library(),{})
    assert exc.value.issues['duplicates']=={str(n):['U1','U2'] for n in (1,2,3)}
    fixed,receipt=apply_ownership(raw,tasks,routes,[dict(number=n,setup_id='U2') for n in (1,2,3)])
    assert receipt['removed_empty_setup_ids']==['U1']
    assert fixed['stations'][0]['setup_order']==['U2']
    assert validate(fixed,tasks,routes,library(),{})['coverage']['assigned']==3
    assert raw==original


def test_local_repair_retries_invalid_decision_and_uses_only_explicit_model_choices(monkeypatch):
    from oracle_viewer.stations import repair_ownership
    import oracle_viewer.stations as module
    tasks,routes,raw=fixture();raw['stations'][0]['setups'][1]['fai_numbers']=[1,2]
    seen=[]
    def generate(model,prompt,evidence,max_tokens):
        seen.append(evidence)
        assert [t['number'] for t in evidence['affected_tasks']]==[1,3]
        rows=[dict(number=1,setup_id='U2')]
        if len(seen)>1:rows.append(dict(number=3,setup_id='U1'))
        return dict(assignments=rows),dict(text='local test response',truncated=False)
    monkeypatch.setattr(module,'generate_json',generate)
    attempts=[]
    fixed,receipt=repair_ownership(object(),raw,tasks,routes,attempts,lambda *_:None)
    assert len(seen)==2 and attempts[0]['validation_error']
    assert receipt['assignments']==[dict(number=1,setup_id='U2'),dict(number=3,setup_id='U1')]
    assert validate(fixed,tasks,routes,library(),{})['coverage']['assigned']==3


def test_repair_does_not_reassign_to_incompatible_method_or_hide_same_setup_duplicates():
    from oracle_viewer.stations import apply_ownership,CoverageError
    tasks,routes,raw=fixture();raw['stations'][0]['setups'][0]['fai_numbers']=[1,1]
    with pytest.raises(CoverageError) as exc:validate(raw,tasks,routes,library(),{})
    assert exc.value.issues['duplicates']=={'1':['U1','U1']}
    fixed,_=apply_ownership(raw,tasks,routes,[dict(number=1,setup_id='U1')])
    assert validate(fixed,tasks,routes,library(),{})['coverage']['assigned']==3
    raw['stations'][0]['setups'][0]['channel']='2d_vision'
    with pytest.raises(ValueError,match='incompatible'):apply_ownership(raw,tasks,routes,[dict(number=1,setup_id='U1')])


def test_ownership_repair_batches_large_sets_without_omitting_any_decision(monkeypatch):
    import oracle_viewer.stations as module
    tasks,routes,raw=fixture()
    tasks=[dict(tasks[0],number=n) for n in range(1,10)]
    routes=[dict(number=n,channel='3d_surface') for n in range(1,10)]
    for setup in raw['stations'][0]['setups']:setup['fai_numbers']=[1]
    seen=[]
    def generate(model,prompt,evidence,max_tokens):
        numbers=[t['number'] for t in evidence['affected_tasks']];seen.append(numbers)
        assert all(r['setup_ids']==['U1','U2'] for r in evidence['allowed_assignments'])
        return dict(assignments=[dict(number=n,setup_id='U2') for n in numbers]),dict(text='local test',truncated=False)
    monkeypatch.setattr(module,'generate_json',generate)
    fixed,receipt=module.repair_ownership(object(),raw,tasks,routes,[],lambda *_:None)
    assert seen==[list(range(1,7)),list(range(7,10))]
    assert len(receipt['assignments'])==9
    assert module.validate(fixed,tasks,routes,library(),{})['coverage']['assigned']==9


def test_identical_group_and_number_encoding_is_accepted_but_conflicts_rejected():
    from oracle_viewer.stations import setup_scope
    tasks,routes,raw=fixture();u=raw['stations'][0]['setups'][0]
    u['fai_group_ids']=['3d_surface'];u['fai_numbers']=[3,1,2]
    assert setup_scope(u,routes)==[3,1,2]
    u['fai_numbers']=[1,2]
    with pytest.raises(ValueError,match='Conflicting FAI group'):setup_scope(u,routes)
    u['fai_numbers']=[];u['fai_group_ids']=['unknown']
    with pytest.raises(ValueError,match='allowed'):setup_scope(u,routes)


def test_duplicate_whole_groups_are_decided_by_local_model_and_expanded_losslessly(monkeypatch):
    import oracle_viewer.stations as module
    tasks,routes,raw=fixture()
    for u in raw['stations'][0]['setups']:
        u['fai_group_ids']=['3d_surface'];u['fai_numbers']=[1,2,3]
    calls=[]
    def generate(model,prompt,evidence,max_tokens):
        calls.append(evidence)
        assert evidence['source_groups']=={'3d_surface':[1,2,3]}
        assert evidence['allowed_setup_ids']=={'3d_surface':['U1','U2']}
        return dict(group_assignments=[dict(channel='3d_surface',setup_id='U2')]),dict(text='local test',truncated=False)
    monkeypatch.setattr(module,'generate_json',generate)
    attempts=[];fixed,receipt=module.repair_ownership(object(),raw,tasks,routes,attempts,lambda *_:None)
    assert len(calls)==1 and attempts[0]['stage']=='ownership_group_repair'
    assert receipt['assignments']==[dict(number=n,setup_id='U2') for n in (1,2,3)]
    assert receipt['removed_empty_setup_ids']==['U1']
    assert module.validate(fixed,tasks,routes,library(),{})['coverage']['assigned']==3
