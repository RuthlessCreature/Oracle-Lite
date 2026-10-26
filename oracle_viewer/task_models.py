"""Use a validated self-hosted task adapter without mutating the shared model."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path


def task_model(model, task):
    if getattr(model,'openai_backend_label',None)!='vllm-self-hosted':return model
    route_path=Path(os.environ.get('ORACLE_VIEWER_TASK_MODEL_ROUTES',
        str(Path(__file__).resolve().parents[1]/'talker/state/model-task-routes.json')))
    if not route_path.is_file():return model
    try:
        route=json.loads(route_path.read_text()).get(task)
        if not route:return model
        report=Path(route['validation_report'])
        if hashlib.sha256(report.read_bytes()).hexdigest()!=route['validation_sha256']:return model
        value=json.loads(report.read_text());metrics=value.get('metrics',{})
        candidate=metrics.get('adapter',{});base=metrics.get('base',{})
        if (task!='fai_manifest' or value.get('training_scope')!='FAI_direct_measurement_manifest'
            or value.get('complete') is not True or value.get('candidate')!=route.get('model')
            or route.get('base_model')!=model.openai_model
            or route.get('endpoint')!=model.openai_base_url
            or any(r.get('error') or r.get('truncated') for r in value.get('items',[]))
            or candidate.get('exact_pages')!=candidate.get('pages')
            or not candidate.get('pages') or candidate.get('invented_points')!=0
            or candidate.get('valid_json_pages')!=candidate.get('pages')
            or candidate.get('correct_points',0)<base.get('correct_points',0)):
            return model
        selected=copy.copy(model)
        selected.inference_backend=copy.copy(model.inference_backend)
        selected.openai_model=route['model'];selected.inference_backend.model=route['model']
        # Shared RLock serializes the same endpoint, while the request alias is isolated.
        return selected
    except (ValueError,KeyError,OSError,TypeError):
        return model
