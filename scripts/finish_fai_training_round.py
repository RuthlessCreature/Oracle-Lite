"""Finish one authorized cloud training round, then retry all active FAI projects.

A task adapter can only serve source-manifest decisions after a complete paired
holdout passes; geometry remains with the configured self-hosted base and gates.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from talker.inference_backend import OpenAICompatibleBackend
from oracle_viewer.task_models import task_model
from types import SimpleNamespace
RUN='fai-manifest-contract-20261023'
CANDIDATE='oracle-source-'+RUN
REPORTS=ROOT/'reports/cloud-qwen/20261023/fai-manifest-round'
SSH=['ssh','-i','/home/cyg/.ssh/id_ed25519_oracle_training','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','ubuntu@119.91.74.163']


def write(value):
    REPORTS.mkdir(parents=True,exist_ok=True)
    temp=REPORTS/'round-status.tmp';temp.write_text(json.dumps(dict(value,updated_at=time.time()),ensure_ascii=False,indent=2));temp.replace(REPORTS/'round-status.json')


def get(path,post=False):
    req=urllib.request.Request('http://127.0.0.1:7870'+path,data=b'' if post else None)
    with urllib.request.urlopen(req,timeout=40) as r:return json.load(r)


def remote_receipt(relative):
    code=f'import pathlib; p=pathlib.Path("/home/ubuntu/wjkj/oracle-training/runs/{RUN}/{relative}"); print(p.read_text() if p.is_file() else "{{}}")'
    import shlex
    r=subprocess.run(SSH+['python3 -c '+shlex.quote(code)],text=True,capture_output=True,timeout=30,check=True)
    return json.loads(r.stdout)


def run():
    started=time.time();state=dict(status='running',stage='waiting_for_training_and_holdout',run_id=RUN,XML_used_as_input=False)
    write(state)
    while time.time()-started<10800:
        pipeline=remote_receipt('pipeline.json');state['cloud']=pipeline;write(state)
        if pipeline.get('status') in ('complete','failed'):break
        time.sleep(15)
    else:raise TimeoutError('Cloud training/holdout did not finish within 3 hours')
    evaluation=remote_receipt('evaluate-vllm/evaluation.json')
    local=REPORTS/'validation-report.json';local.write_text(json.dumps(evaluation,ensure_ascii=False,indent=2))
    state['adapter_deployed']=False
    if pipeline.get('source_gate_passed') is True:
        secret={}
        for line in Path('/home/cyg/.config/oracle-viewer/qwen-vllm.env').read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k,v=line.split('=',1);secret[k]=v.strip().strip('"').strip("'")
        backend=OpenAICompatibleBackend('http://127.0.0.1:59000','qwen3.8-27b',label='vllm-self-hosted',api_key=secret['ORACLE_VIEWER_MODEL_API_KEY'])
        route=dict(model=CANDIDATE,base_model=backend.model,endpoint=backend.base_url,validation_report=str(local),validation_sha256=hashlib.sha256(local.read_bytes()).hexdigest())
        path=ROOT/'talker/state/model-task-routes.json'
        existing=json.loads(path.read_text()) if path.is_file() else {}
        # Validate against the exact same gate used by inference before publishing.
        candidate_path=REPORTS/'candidate-routes.json';candidate_path.write_text(json.dumps(dict(existing,fai_manifest=route)))
        previous=os.environ.get('ORACLE_VIEWER_TASK_MODEL_ROUTES');os.environ['ORACLE_VIEWER_TASK_MODEL_ROUTES']=str(candidate_path)
        base=SimpleNamespace(openai_model=backend.model,openai_base_url=backend.base_url,openai_backend_label=backend.label,inference_backend=backend)
        selected=task_model(base,'fai_manifest')
        if previous is None:os.environ.pop('ORACLE_VIEWER_TASK_MODEL_ROUTES',None)
        else:os.environ['ORACLE_VIEWER_TASK_MODEL_ROUTES']=previous
        if selected is base:raise ValueError('Candidate failed task-routing validation gate')
        available=backend._request('/v1/models')['data']
        if CANDIDATE not in {r['id'] for r in available}:
            try:backend._request('/v1/load_lora_adapter',method='POST',payload=dict(lora_name=CANDIDATE,lora_path=f'/oracle-training/runs/{RUN}/train/adapter-final'),timeout=120)
            except json.JSONDecodeError:pass  # vLLM success may be plain text.
        if CANDIDATE not in {r['id'] for r in backend._request('/v1/models')['data']}:
            raise ValueError('Candidate is not registered on the live endpoint')
        temp=path.with_suffix('.tmp');temp.write_text(candidate_path.read_text());temp.replace(path)
        state['adapter_deployed']=True
    state.update(stage='waiting_for_current_queue');write(state)
    while True:
        queue=get('/api/viewer/task-queue')
        if not queue.get('active') and not queue.get('waiting'):break
        if time.time()-started>21600:raise TimeoutError('Existing marking queue exceeded 6 hours')
        time.sleep(15)
    # Load renderer/validator fixes only after the current task has completed.
    subprocess.run(['systemctl','--user','restart','oracle-viewer.service'],check=True)
    for _ in range(60):
        try:
            if get('/api/status').get('loaded'):break
        except Exception:pass
        time.sleep(2)
    else:raise RuntimeError('Viewer did not become ready after reload')
    projects=[('613-13599','9721ec1c27e045aa955028e8a2807016'),('806-69194','61502bd97bbb4d1ea2634cf21f858d03'),('613-16076','cc8e1c4224324fe69f771f4efbd4219d')]
    results=[]
    for name,pid in projects:
        state.update(stage='marking',project=name,results=results);write(state)
        get(f'/api/viewer/projects/{pid}/local-model-marking?retry=true',post=True)
        deadline=time.time()+21600
        while True:
            progress=get(f'/api/viewer/projects/{pid}/local-model-marking')
            state['project_progress']={k:progress.get(k) for k in ('status','progress','counts','coverage','error')};write(state)
            if progress.get('status') not in ('running','queued'):break
            if time.time()>deadline:raise TimeoutError('Project marking exceeded 6 hours: '+name)
            time.sleep(15)
        (REPORTS/f'{name}-after.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2))
        results.append(dict(project=name,status=progress.get('status'),coverage=progress.get('coverage'),error=progress.get('error')))
    full=all(r.get('coverage',{}).get('unmarked')==0 for r in results)
    state.update(status='complete' if full else 'partial',stage='finished',results=results,all_FAI_fully_marked=full)
    write(state)


if __name__=='__main__':
    try:run()
    except Exception as exc:
        write(dict(status='failed',stage='orchestration_failed',error=f'{type(exc).__name__}: {exc}'))
        raise
