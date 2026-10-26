"""An exported complete plan must not lose any physical setup view or FAI."""
import base64
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from oracle_viewer.station_presentation import frame_keys, validated_frames, StationPresentationService

PNG='data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\n').decode()

def record():
    return dict(id='a'*32,status='complete',stale=False,task_snapshot=[dict(number=13),dict(number=23)],result=dict(stations=[dict(id='S1',setups=[dict(id='U1',fai_numbers=[13]),dict(id='U2',fai_numbers=[23])])]))

def test_all_station_and_setup_views_are_required():
    r=record();assert frame_keys(r)=={'overview','station_S1','setup_U1','setup_U2'}
    images={k:PNG for k in frame_keys(r)};original=copy.deepcopy(r)
    assert set(validated_frames(images,r))==frame_keys(r);assert r==original
    del images['setup_U2']
    with pytest.raises(ValueError,match='不能漏项'):validated_frames(images,r)

def test_rejects_wrong_scheme_images_and_path_identifiers():
    r=record();images={k:PNG for k in frame_keys(r)};images['legacy_camera']=PNG
    with pytest.raises(ValueError):validated_frames(images,r)
    r['result']['stations'][0]['id']='../../other';images={k:PNG for k in frame_keys(r)}
    with pytest.raises(ValueError,match='identifier'):validated_frames(images,r)

def test_rejects_non_png_or_oversize():
    r=record();images={k:PNG for k in frame_keys(r)};images['overview']='data:image/jpeg;base64,AAAA'
    with pytest.raises(ValueError,match='PNG'):validated_frames(images,r)
    images['overview']='data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\n'+b'x'*5_000_000).decode()
    with pytest.raises(ValueError,match='oversized'):validated_frames(images,r)

def test_stale_or_duplicate_fai_cannot_be_exported():
    r=record();service=StationPresentationService(SimpleNamespace(get=lambda pid:r,store=None))
    r['stale']=True
    with pytest.raises(ValueError,match='当前版本'):service.submit('p',r['id'],{})
    r['stale']=False;r['result']['stations'][0]['setups'][1]['fai_numbers']=[13]
    with pytest.raises(ValueError,match='ownership'):service.submit('p',r['id'],{})

def test_status_rejects_changed_plan_and_exporter(tmp_path,monkeypatch):
    import oracle_viewer.station_presentation as module
    r=record();service=StationPresentationService(SimpleNamespace(get=lambda pid:r,store=SimpleNamespace(_dir=lambda pid:tmp_path)))
    monkeypatch.setattr(module,'generator_hash',lambda:'current');folder=service.path('p','b'*32);folder.mkdir(parents=True)
    job=dict(id='b'*32,plan_id=r['id'],generator_sha256='current',status='complete');(folder/'job.json').write_text(json.dumps(job))
    assert not service.get('p',job['id'])['stale'];r['id']='c'*32;assert service.get('p',job['id'])['stale']
    r['id']=job['plan_id'];job['generator_sha256']='old';(folder/'job.json').write_text(json.dumps(job));assert service.get('p',job['id'])['stale']

def test_node_cache_requires_same_plan_source_and_evidence(tmp_path,monkeypatch):
    import oracle_viewer.station_presentation as module
    calls=[]
    node=dict(id='requirements',**{k:['输入资料A','输入资料B'] for k in ('inputs','actions','outputs','checks')})
    def generate(model,prompt,context,max_tokens):
        calls.append((prompt,context));return copy.deepcopy(node),dict(truncated=False,input_sha256='receipt')
    monkeypatch.setattr(module,'generate_json',generate)
    args=(None,'requirements','prompt',dict(project='P',model_identity='strata'),tmp_path,'source-A','plan-A',lambda retry:None)
    first,receipts=module.explain_node(*args);assert len(calls)==1 and not receipts[0]['cache_hit']
    again,receipts=module.explain_node(*args);assert again==first and receipts[0]['cache_hit'] and len(calls)==1
    changed=list(args);changed[5]='source-B';module.explain_node(*changed);assert len(calls)==2
    changed=list(args);changed[6]='plan-B';module.explain_node(*changed);assert len(calls)==3
    changed=list(args);changed[3]=dict(project='P',model_identity='other-local-model');module.explain_node(*changed);assert len(calls)==4

def test_wrong_or_truncated_node_retries_without_caching_invalid_output(tmp_path,monkeypatch):
    import oracle_viewer.station_presentation as module
    good=dict(id='regions',**{k:['工程动作A','工程动作B'] for k in ('inputs','actions','outputs','checks')});calls=[];progress=[]
    def generate(*args):
        calls.append(1)
        if len(calls)==1:raise module.DecisionError('truncated',dict(truncated=True))
        if len(calls)==2:return good|dict(id='stations'),dict(truncated=False)
        return good,dict(truncated=False)
    monkeypatch.setattr(module,'generate_json',generate)
    node,attempts=module.explain_node(None,'regions','prompt',{},tmp_path,'source','plan',progress.append)
    assert node==good and len(attempts)==3 and attempts[0]['truncated'] and progress==[1,2]
    saved=json.loads(next(tmp_path.glob('*.json')).read_text());assert saved['node']==good and not saved['attempt']['truncated']

def test_persistent_invalid_node_fails_with_full_attempt_receipts(tmp_path,monkeypatch):
    import oracle_viewer.station_presentation as module
    monkeypatch.setattr(module,'generate_json',lambda *args:({},dict(truncated=False,text='{}')))
    with pytest.raises(module.DecisionError) as failure:
        module.explain_node(None,'regions','prompt',{},tmp_path,'source','plan',lambda n:None)
    assert len(failure.value.attempt['node_attempts'])==3 and not list(tmp_path.glob('*.json'))
