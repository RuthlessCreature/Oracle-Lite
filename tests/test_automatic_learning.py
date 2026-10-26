import json
import os
import subprocess
import sys
from pathlib import Path
from oracle_viewer.store import ViewerStore
from oracle_viewer.adaptation import training_preflight
from oracle_viewer.automatic_learning import run_automatic_learning


class SourceStore:
    def listing(self):return [{'id':p} for p in ('A','B')]
    def get(self,pid):return dict(id=pid,drawing_no=pid)
    def active_msop(self,p):return dict(sha256='source-'+p['id'],filename='source.pdf',method_pages=[
        dict(page=1,text='X, Y, Z (mm) 1 2 3 POINT_P01'),
        dict(page=2,text='X, Y, Z (mm) 1 2 3 POINT_P01\nX, Y, Z (mm) 4 2 3 POINT_P01')])


def test_automatic_curriculum_needs_no_manual_labels_but_excludes_contradictions(tmp_path):
    state=run_automatic_learning(SourceStore(),tmp_path,dict(inference_backend='strata-native'))
    assert state['curriculum']['train_count']==1 and state['curriculum']['test_count']==1
    assert len(state['curriculum']['excluded_pages'])==2
    row=json.loads((tmp_path/'train.jsonl').read_text())
    assert row['automatic_label'] and not row['geometry_label']
    assert json.loads(row['answer'])['points'][0]['xyz_mm']==[1,2,3]
    assert not state['weight_training_started'] and not state['weights_changed']


def test_close_surface_fit_does_not_self_certify_training_label(tmp_path):
    registration=dict(frames=[dict(program='UP',coordinate_system='ALI_A',status='proposal',
        triangle_refinement=dict(status='surface_fit_unreviewed',best=dict(max_residual_mm=.01)))])
    state=run_automatic_learning(SourceStore(),tmp_path,{},registration_report=registration)
    assert not state['geometry_frames'][0]['training_eligible']


def test_repeated_input_receipt_and_data_are_reproducible(tmp_path):
    one=run_automatic_learning(SourceStore(),tmp_path,{'model_alias':'same'})
    two=run_automatic_learning(SourceStore(),tmp_path,{'model_alias':'same'})
    assert one['input_fingerprint']==two['input_fingerprint']
    assert one['curriculum']['file_sha256']==two['curriculum']['file_sha256']


def test_restart_preserves_receipts_and_makes_interrupted_job_resubmittable(tmp_path):
    store=ViewerStore(tmp_path);p=store.create('test')
    root=store._dir(p['id'])/'automatic-learning';root.mkdir()
    (root/'state.json').write_text(json.dumps(dict(status='running',weight_training_started=False)))
    (root/'registration.json').write_text('{"receipt":"preserve"}')
    store.recover_tasks()
    assert json.loads((root/'state.json').read_text())['status']=='failed'
    assert json.loads((root/'registration.json').read_text())['receipt']=='preserve'


def test_worker_scripts_import_without_inherited_pythonpath(tmp_path):
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    for name in ('run_automatic_learning.py','auto_register_msop.py'):
        result=subprocess.run([sys.executable,str(Path('scripts',name).resolve()),'--help'],cwd=tmp_path,
            env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_native_training_rejects_wrong_architecture_and_insufficient_disk(tmp_path,monkeypatch):
    (tmp_path/'config.json').write_text('{"model_type":"wrong"}')
    (tmp_path/'weights.safetensors').write_bytes(b'placeholder')
    checked=training_preflight(dict(expected_model_type='expected'),tmp_path)
    assert any('架构' in r for r in checked['reasons'])
    import shutil
    from types import SimpleNamespace
    monkeypatch.setattr(shutil,'disk_usage',lambda path:SimpleNamespace(free=10))
    checked=training_preflight(dict(training_checkpoint_bytes=1000,native_training_backend_available=False))
    assert checked['resource_check']['free_disk_bytes']==10
    assert any('资源条件' in r for r in checked['reasons'])
    assert any('训练后端' in r for r in checked['reasons'])
