import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from talker.config import load_config
from talker.model import TalkerModel
from oracle_viewer.store import ViewerStore
from oracle_viewer.auto_registration import registration_algorithm_sha256
from oracle_viewer.msop_coordinates import _coordinate_rows
from oracle_viewer.candidate_mining import extract_named_points,extract_feature_labels,resolve_dominant_source_page_conflicts
from oracle_viewer.automatic_candidates import (_source_point_evidence,_required_marker_count,
    _registration_frame_hypotheses,_source_point_scopes)
from oracle_viewer.local_matching import (_planar_frame_hypotheses,_project_model_selected_xyz_group,
    _normal_origin_frame_hypotheses,_coalesce_source_point_definitions,
    _mark_model_selected_xyz_group,_is_local_inference_model,_validate_coordinate_variant_choices)

PROJECT=sys.argv[1]
BASE='http://127.0.0.1:7870/api/viewer'
store=ViewerStore(ROOT/'talker/state/viewer')
project=store.get(PROJECT);msop=store.active_msop(project);mesh=store.mesh(PROJECT)
registration=store._dir(PROJECT)/'automatic-learning/registration.json'
registration_hash=hashlib.sha256(registration.read_bytes()).hexdigest()
receipt=json.loads(registration.read_text())
if (receipt.get('cad_sha256')!=mesh['source_sha256'] or receipt.get('msop_sha256')!=msop['sha256']
        or receipt.get('algorithm_sha256')!=registration_algorithm_sha256()):
    raise RuntimeError('Current source-pinned registration evidence required')

def api(path,body=None):
    request=urllib.request.Request(BASE+path,data=json.dumps(body,ensure_ascii=False).encode() if body is not None else None,
        headers={'Content-Type':'application/json'},method='POST' if body is not None else 'GET')
    with urllib.request.urlopen(request,timeout=60) as response:
        return json.load(response)

cfg=load_config()
model=TalkerModel(cfg.training_output_dir,openai_base_url='http://127.0.0.1:8081',
    openai_model='qwen3.8-flash-next-iq2_xs',backend_label='strata-native')
model.load()
if not model.loaded or not _is_local_inference_model(model):
    raise RuntimeError('A verified local model is required')
identity=model.info()
source_choice_proofs={}

class AtomicAPIStore:
    def __getattr__(self,name):return getattr(store,name)
    def annotate(self,project_id,number,anchors,note,status,cad_sha256,document_id,
                 origin='manual',confidence=None,candidate_face_ids=None):
        if origin!='local_model_candidate' or status!='auto_marked':
            raise RuntimeError('Only validated local-model marking is permitted')
        current=store.get(project_id)
        if ((current.get('model') or {}).get('source_sha256')!=cad_sha256
                or store.active_msop(current)['sha256']!=msop['sha256']
                or hashlib.sha256(registration.read_bytes()).hexdigest()!=registration_hash):
            raise RuntimeError('Source or registration changed during model review')
        annotation=next(item for item in store.items(current) if item['number']==number).get('annotation') or {}
        state=api(f'/projects/{project_id}/local-model-marking')
        if annotation.get('status')=='auto_marked' or state.get('progress',{}).get('current_fai')==number:
            raise RuntimeError('Item already marked or currently in flight; defer to active batch')
        proof={'fai':number,'origin':origin,'status':status,'confidence':confidence,
               'cad_sha256':cad_sha256,'document_sha256':msop['sha256'],
               'registration_sha256':registration_hash,'model_identity':identity,
               'source_coordinate_choice_receipt':source_choice_proofs.get(number),
               'anchors':anchors,'full_model_and_coordinate_audit':note}
        content=json.dumps(proof,ensure_ascii=False,indent=2).encode()
        proof_path=store._dir(project_id)/'automatic-learning'/f'fai-shared-landmark-model-proof-{cad_sha256[:16]}-{number}.json'
        proof_path.write_bytes(content)
        digest=hashlib.sha256(content).hexdigest()
        compact=(f'本地 Strata 模型自动核准完整点组；FAI {number}，共 {len(anchors)} 点，显示置信度 {confidence}。'
                 f'完整模型决策、原文 XYZ 与 CAD 逐点审计：automatic-learning/{proof_path.name}；SHA256 {digest}。'
                 f'配准依据 SHA256 {registration_hash}；此为 Viewer 位置标记，不代表计量验收。')
        converted=[{**anchor,'position':[float(v) for v in anchor['position']],
                    'label_offset':[float(v) for v in anchor.get('label_offset',[32,-26])]} for anchor in anchors]
        return api(f'/projects/{project_id}/fai/{number}/annotation',{
            'anchors':converted,'note':compact,'status':status,'cad_sha256':cad_sha256,
            'document_id':document_id,'origin':origin,'confidence':confidence,
            'candidate_face_ids':candidate_face_ids or []})

initial=api(f'/projects/{PROJECT}/local-model-marking')
failed={item['fai'] for item in initial.get('items',[]) if item.get('result')=='abstain_or_invalid'}
from oracle_viewer.fai_strategy import strategy_state
failed.update(row['fai'] for row in strategy_state(store,PROJECT).get('items',[])
    if row.get('outcome_at_classification',{}).get('failure_reason'))
unmarked=set(initial['coverage']['unmarked_numbers'])
eligible=failed&unmarked
eligible.discard(initial.get('progress',{}).get('current_fai'))
named=extract_named_points(msop['method_pages'])
frames=[{**row['selected_transform'],'program':row['program'],'coordinate_system':row['coordinate_system'],
         'hypothesis_id':row.get('hypothesis_id'),'model_confidence':row.get('model_confidence')}
        for row in receipt.get('local_model_registration',{}).get('frames',[])
        if row.get('status')=='selected' and isinstance(row.get('selected_transform'),dict)]
results=[]
print('LOCAL_AUTO_RETRY',PROJECT,'completed_failed_items',sorted(eligible),flush=True)
for number in sorted(eligible):
    latest=store.get(PROJECT)
    item=next(item for item in store.items(latest) if item['number']==number)
    if (item.get('annotation') or {}).get('status')=='auto_marked':continue
    detail=store.detail(PROJECT,number)
    rows=_coordinate_rows(detail['methods'],[],extract_feature_labels(detail['methods'],item.get('features',[])),
                          msop['method_pages'],named)
    rows,_=resolve_dominant_source_page_conflicts(rows)
    points=_coalesce_source_point_definitions(_source_point_evidence(rows))
    if any(point.get('requires_variant_choice') for point in points):
        choice_path=store._dir(PROJECT)/'automatic-learning'/f'fai-coordinate-variants-{mesh["source_sha256"][:16]}-{number}.json'
        if not choice_path.is_file():continue
        choice_audit=json.loads(choice_path.read_text())
        if choice_audit.get('status')!='selected':continue
        valid_choices,choices=_validate_coordinate_variant_choices(points,choice_audit.get('selected_choices'))
        if not valid_choices:continue
        by_name={row['name']:row for row in choices}
        resolved=[]
        for point in points:
            selected_row=by_name.get(point['name'])
            row=dict(point)
            if selected_row:
                row.update(xyz=selected_row['xyz'],program=selected_row['program'],
                           coordinate_system=selected_row['coordinate_system'],pages=[selected_row['page']])
            row.update(requires_variant_choice=False,coordinate_variants=[])
            resolved.append(row)
        points=resolved
        source_choice_proofs[number]={'path':str(choice_path),
            'sha256':hashlib.sha256(choice_path.read_bytes()).hexdigest(),'audit':choice_audit,
            'all_choices_revalidated_against_current_source_options':True}
    required=_required_marker_count(detail['methods'],named)
    if (not points or required and len(points)!=required
            or any(point.get('requires_variant_choice') or not isinstance(point.get('xyz'),list)
                   or len(point['xyz'])!=3 for point in points)):
        continue
    scopes=_source_point_scopes(points)
    selected=[frame for frame in frames if (frame['program'],frame['coordinate_system']) in scopes]
    if {(frame['program'],frame['coordinate_system']) for frame in selected}!=scopes:continue
    from oracle_viewer.landmark_transfer import peer_landmark_frame_options
    proposals=peer_landmark_frame_options(store,latest,item,points,selected)
    if not proposals:continue
    valid=[]
    for proposal in proposals:
        projection,error=_project_model_selected_xyz_group(mesh,points,proposal,detail,[])
        if error is None:valid.append(proposal)
    if not valid:continue
    print('LOCAL_MODEL_REVIEW',number,'points',len(points),'valid_rigid_plane_candidates',len(valid),flush=True)
    candidate={'known_source_points':points,'required_marker_count':required,
        'evidence':{'known_source_points':points,'required_marker_count':required,
                    'registration_diagnostic':{'selected_frame_transforms':valid[0],
                        'frame_transform_hypotheses':[
                            *[{**frame,'candidate_source':'exact_shared_source_landmark_graph'}
                              for proposal in valid for frame in proposal]]}}}
    try:
        result=_mark_model_selected_xyz_group(AtomicAPIStore(),project,item,detail,model,mesh,candidate)
        result=result or {'fai':number,'result':'no_group_review_candidate'}
    except Exception as exc:
        result={'fai':number,'result':'deferred_or_failed','error':f'{type(exc).__name__}: {exc}'}
    results.append(result)
    print(json.dumps({key:result.get(key) for key in ('fai','result','reason','anchor_count','selected_local_frame_candidate','error')},ensure_ascii=False),flush=True)

final=api(f'/projects/{PROJECT}/local-model-marking')
report={'project':PROJECT,'model':identity,'eligible_completed_failed_items':sorted(eligible),
        'results':results,'coverage':final['coverage'],'human_selected_point_or_pose':False,
        'source_coordinates_modified':False,'metrology_accepted':False}
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=ROOT/f'reports/engineering-checks/20261016/shared-landmark-local-model-retry-{stamp}.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('COVERAGE',json.dumps(final['coverage'],ensure_ascii=False),'REPORT',out,flush=True)
