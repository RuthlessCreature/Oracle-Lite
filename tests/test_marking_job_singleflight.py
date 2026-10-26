from pathlib import Path
from concurrent.futures import Future
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from oracle_viewer import api
from oracle_viewer.msop import parse_page


def test_reopening_project_cannot_queue_duplicate_or_reset_live_progress(tmp_path,monkeypatch):
    class Pool:
        def __init__(self,**kwargs):self.calls=[]
        def submit(self,*args):
            future=Future();self.calls.append((args,future));return future
    monkeypatch.setattr(api,'ThreadPoolExecutor',Pool)
    model=SimpleNamespace(loaded=True,alias='local-version-a')
    model.info=lambda:{'inference_backend':'strata-native','model_alias':model.alias}
    state=SimpleNamespace(cfg=SimpleNamespace(state_dir=tmp_path/'state',
        training_output_dir=tmp_path/'output',fai_matching_model_root=tmp_path),
        model=model,viewer_model_error=None)
    app=FastAPI();api.attach_viewer(app,state)
    project=state.viewer.create('singleflight','generic','A')
    project['documents']=[{'id':'doc','role':'msop','sha256':'b'*64,
        'filename':'generic.pdf','page_count':1,'method_parser_version':2,
        'method_pages':[parse_page('MSOP: FAI 1\nMeasure 1 vector point P01.',1)]}]
    project['model']={'status':'preview','source_sha256':'a'*64}
    state.viewer._write(project)
    pid=project['id'];endpoint=f'/api/viewer/projects/{pid}/local-model-marking'
    with TestClient(app) as client:
        assert client.post(endpoint).json()['status']=='queued'
        assert len(app.state.viewer_pool.calls)==1
        state.viewer.update_local_model_marking(pid,status='running',
            progress={'done':7,'current_fai':8,'message':'Local model is processing point group'})
        # A different fingerprint must not schedule a second job while the
        # first future is still live, or reset its progress back to zero.
        model.alias='local-version-b'
        repeated=client.post(endpoint).json()
        assert repeated['status']=='running' and repeated['progress']['done']==7
        assert len(app.state.viewer_pool.calls)==1
        app.state.viewer_pool.calls[0][1].set_result(None)
        state.viewer.update_local_model_marking(pid,status='partial',
            counts=dict(accepted=3,abstain=2,skipped=0,processed=5),items=[{'fai':1,'result':'old'}])
        queued=client.post(endpoint).json()
        assert queued['status']=='queued'
        assert queued['counts']==dict(accepted=0,abstain=0,skipped=0,processed=0)
        assert queued['items']==[] and queued['progress']['done']==0
        assert len(app.state.viewer_pool.calls)==2
        app.state.viewer_pool.calls[1][1].set_result(None)
        state.viewer.update_local_model_marking(pid,status='partial')
        original_read=Path.read_bytes
        def changed_on_disk(path):
            content=original_read(path)
            return content+b'\n# development edit not loaded by this process' if path.name=='local_matching.py' else content
        monkeypatch.setattr(Path,'read_bytes',changed_on_disk)
        unchanged_runtime=client.post(endpoint).json()
        assert unchanged_runtime['status']=='partial'
        assert len(app.state.viewer_pool.calls)==2
