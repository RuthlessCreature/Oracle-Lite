import json
from oracle_viewer.local_matching import _candidate_prompt_payload, _normalize_model_source_aliases


def test_repeated_candidate_name_evidence_is_lossless_and_compact():
    names=[f'PART_P{i:03}' for i in range(40)]
    original=[dict(id=i,source_point_names=names,uv=[i/100,0.5],face_id=f'face{i}') for i in range(80)]
    packed=_candidate_prompt_payload(original)
    restored=[]
    for candidate in packed['candidates']:
        restored.append({**candidate,'source_point_names':packed['source_point_groups'][candidate['source_point_names']['group']]})
    assert restored==original
    assert len(json.dumps(packed)) < len(json.dumps(original))/3
    assert original[0]['source_point_names']==names


def test_only_unambiguous_allowed_msop_page_aliases_normalize():
    parsed={'source_pages':['MSOP p.19','MSOP page 20','drawing p.19','MSOP p.99'],
            'markers':[{'source_pages':['19'],'page':'MSOP p.20','source_point_names':['P_A']}]}
    result,_=_normalize_model_source_aliases(parsed,[{'name':'P_A'}],{19,20})
    assert result['source_pages']==[19,20,'drawing p.19','MSOP p.99']
    assert result['markers'][0]['source_pages']==[19]
    assert result['markers'][0]['page']==20
    assert parsed['source_pages'][0]=='MSOP p.19'


def test_marking_queue_reports_partial_coverage_as_partial(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from oracle_viewer.api import ViewerTaskQueue
    from oracle_viewer.store import ViewerStore
    store=ViewerStore(tmp_path);project=store.create('p','drawing','A')
    def run_marking(project_id):
        store.update_local_model_marking(project_id,status='partial',message='已标记 12/63 项；未标记 51 项')
    with ThreadPoolExecutor(max_workers=1) as executor:
        queue=ViewerTaskQueue(executor,store)
        queue.submit(run_marking,project['id']).result(timeout=5)
        row=queue.snapshot()['tasks'][0]
    assert row['status']=='partial'
    assert '12/63' in row['progress']


def test_marking_queue_snapshot_exposes_live_fai_classification_count(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from oracle_viewer.api import ViewerTaskQueue
    from oracle_viewer.store import ViewerStore
    store=ViewerStore(tmp_path);project=store.create('p','drawing','A')
    store.update_local_model_marking(project['id'],status='running',progress={
        'stage':'method_classification','done':0,
        'message':'复用已确认的方法分类，仅重分失败项'})
    state_dir=store._dir(project['id'])/'automatic-learning';state_dir.mkdir(parents=True)
    (state_dir/'fai-strategies.json').write_text(json.dumps({
        'status':'running','done':17,'total':63,'items':[{}]*17}),encoding='utf-8')
    release=Event()
    def run_marking(project_id): release.wait(5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        queue=ViewerTaskQueue(executor,store);queue.submit(run_marking,project['id'])
        try:
            row=queue.snapshot()['tasks'][0]
        finally:
            release.set()
    assert row['status']=='running'
    assert row['progress']=='本地模型方法分类 17/63'
