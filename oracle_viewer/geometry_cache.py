"""Content-addressed reuse of checked geometry, independent of orchestration."""
import ast
import hashlib
import json
from pathlib import Path


def geometry_key(data,tasks,catalog,design,choice):
    root=Path(__file__).parent
    tree=ast.parse((root/'workflow.py').read_text())
    names={'plan','lighting_layouts','optical_pair','optical_pair_for_design','maximum_sensor_roi',
           'fai_single_view_options','fai_capture_summary','recommend_fai_cover','compile_tasks','coordinate_tasks','solve_cover','bounded_model_candidates','archive_layout_candidates'}
    engine=''.join(ast.dump(node,include_attributes=False) for node in tree.body
                   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names).encode()
    for name in ('physical.py','visibility.py','msop_coordinates.py','candidate_mining.py','point_context.py',
                 'registration.py','section_geometry.py','msop.py','hardware_checks.py','channel_routes.py','geometry_cache.py'):
        engine+=(root/name).read_bytes()
    calibration=root.parent/'config'/'cad_frame_calibrations'/f"{data.get('drawing_no')}-{data.get('revision')}.json"
    if calibration.is_file():engine+=calibration.read_bytes()
    from .workflow import source_task_evidence,planning_evidence
    payload=dict(engine=hashlib.sha256(engine).hexdigest(),tasks=source_task_evidence(tasks),
                 model=data.get('model'),bindings=planning_evidence(data.get('bindings')),
                 documents=[{k:d.get(k) for k in ('id','sha256','role')} for d in data.get('documents',[])],
                 catalogue=catalog,design=design,selection={k:choice.get(k) for k in ('pair_ids','light_ids')})
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
