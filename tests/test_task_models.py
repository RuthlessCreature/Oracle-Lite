import hashlib
import json
from types import SimpleNamespace
from oracle_viewer.task_models import task_model


def test_task_route_requires_complete_validated_receipt_and_never_mutates_shared_model(tmp_path,monkeypatch):
    model=SimpleNamespace(openai_model='base',openai_base_url='http://private',openai_backend_label='vllm-self-hosted',inference_backend=SimpleNamespace(model='base'))
    report=tmp_path/'validation.json'
    metrics=dict(pages=2,exact_pages=2,valid_json_pages=2,correct_points=3,invented_points=0)
    value=dict(complete=True,training_scope='FAI_direct_measurement_manifest',candidate='manifest',metrics=dict(base=metrics,adapter=metrics),items=[dict(truncated=False)])
    report.write_text(json.dumps(value))
    route=dict(model='manifest',base_model='base',endpoint='http://private',validation_report=str(report),validation_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    config=tmp_path/'routes.json';config.write_text(json.dumps(dict(fai_manifest=route)))
    monkeypatch.setenv('ORACLE_VIEWER_TASK_MODEL_ROUTES',str(config))
    selected=task_model(model,'fai_manifest')
    assert selected is not model and selected.inference_backend.model=='manifest'
    assert model.openai_model=='base' and model.inference_backend.model=='base'
    assert task_model(model,'geometry') is model
    value['metrics']['adapter']['exact_pages']=1;report.write_text(json.dumps(value))
    assert task_model(model,'fai_manifest') is model
