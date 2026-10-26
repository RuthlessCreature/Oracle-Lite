"""Test the largest unresolved XYZ group against historical frame seeds.

Historical registration is never upgraded to a current certificate. The fresh
local model must choose source rows and validate the full FAI with current gates.
"""
import argparse,hashlib,json,sys,urllib.request
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from oracle_viewer.store import ViewerStore
from oracle_viewer.fai_strategy import strategy_state,_write
from oracle_viewer.candidate_mining import extract_named_points,extract_feature_labels,resolve_dominant_source_page_conflicts
from oracle_viewer.msop_coordinates import _coordinate_rows
from oracle_viewer.automatic_candidates import _source_point_evidence,_required_marker_count
from oracle_viewer.local_matching import _coalesce_source_point_definitions,_mark_model_selected_xyz_group,_is_local_inference_model
from talker.model import TalkerModel
from talker.config import load_config
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('project');parser.add_argument('--fai',type=int)
parser.add_argument('--review-scopes',action='store_true',help='Native measurement/reporting-frame review plus new CAD rigid hypotheses')
args=parser.parse_args()
pid=args.project;store=ViewerStore(ROOT/'talker/state/viewer');project=store.get(pid);doc=store.active_msop(project);mesh=store.mesh(pid)
live=json.load(urllib.request.urlopen(f'http://127.0.0.1:7870/api/viewer/projects/{pid}/local-model-marking',timeout=15))
if live.get('status')=='running':
 raise SystemExit('Deferred: main local-model job is active; do not read an incomplete registration receipt or compete with its commits.')
registration=store._dir(pid)/'automatic-learning'/'registration.json';receipt=json.loads(registration.read_text());registration_sha=hashlib.sha256(registration.read_bytes()).hexdigest()
if receipt['cad_sha256']!=mesh['source_sha256'] or receipt['msop_sha256']!=doc['sha256']:raise RuntimeError('Historical seeds must match actual current CAD and MSOP')
frames=[{**r['selected_transform'],'program':r['program'],'coordinate_system':r['coordinate_system'],
    'historical_seed_only':True,'source_to_cad_registered':False}
    for r in receipt.get('local_model_registration',{}).get('frames',[]) if r.get('status')=='selected']
strategies={r['fai']:r for r in strategy_state(store,pid)['items']};named=extract_named_points(doc['method_pages']);eligible=[]
for item in store.items(project):
 if args.fai is not None and item['number']!=args.fai:continue
 if (item.get('annotation') or {}).get('status')=='auto_marked':continue
 strategy=strategies.get(item['number']) or {}
 if not strategy.get('evidence_class','').startswith('complete_xyz_'):continue
 detail=store.detail(pid,item['number']);rows=_coordinate_rows(detail['methods'],[],extract_feature_labels(detail['methods'],item.get('features',[])),doc['method_pages'],named)
 rows,_=resolve_dominant_source_page_conflicts(rows);points=_coalesce_source_point_definitions(_source_point_evidence(rows));required=_required_marker_count(detail['methods'],named)
 if required and len(points)==required:eligible.append((len(points),item,detail,points,strategy))
if not eligible:raise RuntimeError('No complete source group eligible')
_,item,detail,points,strategy=max(eligible,key=lambda row:row[0]);number=item['number']
model=TalkerModel(load_config().training_output_dir,openai_base_url='http://127.0.0.1:8081',openai_model='qwen3.8-flash-next-iq2_xs',backend_label='strata-native');model.load()
if not model.loaded or not _is_local_inference_model(model):raise RuntimeError('Local model only')
base='http://127.0.0.1:7870/api/viewer'
class AtomicStore:
 def __getattr__(self,key):return getattr(store,key)
 def annotate(self,project_id,fai,anchors,note,status,cad,document_id,origin='manual',confidence=None,**kwargs):
  current=store.get(project_id)
  if origin!='local_model_candidate' or status!='auto_marked':raise RuntimeError('Only native model may commit')
  if (current.get('model') or {}).get('source_sha256')!=cad or store.active_msop(current)['sha256']!=doc['sha256'] or hashlib.sha256(registration.read_bytes()).hexdigest()!=registration_sha:raise RuntimeError('Inputs changed')
  state=json.load(urllib.request.urlopen(base+f'/projects/{pid}/local-model-marking'))
  if state.get('status')=='running':raise RuntimeError('Defer commit while other marking is active')
  evidence={'fai':fai,'cad_sha256':cad,'msop_sha256':doc['sha256'],'historical_registration_seed_sha256':registration_sha,
      'registration_certificate_upgraded':False,'model':model.info(),'full_model_and_coordinate_audit':note,'anchors':anchors}
  raw=json.dumps(evidence,ensure_ascii=False,indent=2).encode();path=store._dir(pid)/'automatic-learning'/f'new-xyz-route-{hashlib.sha256(raw).hexdigest()}.json';path.write_bytes(raw)
  compact=f'本地模型按完整原文点组重新核准；FAI {fai}，共 {len(anchors)} 点。完整证据：automatic-learning/{path.name}；SHA256 {hashlib.sha256(raw).hexdigest()}。历史配准仅作候选种子，未提升计量认证；显示置信度 low。'
  payload={'anchors':anchors,'note':compact,'status':status,'cad_sha256':cad,'document_id':document_id,'origin':origin,'confidence':'low'}
  request=urllib.request.Request(base+f'/projects/{pid}/fai/{fai}/annotation',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
  with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
print('COMPLETE_CURRENT_SOURCE_REVIEW',number,'points',len(points),'source_groups_preserved',flush=True)
trial_path=store._dir(pid)/'automatic-learning'/'xyz-route-trials.json'
prior=json.loads(trial_path.read_text()) if trial_path.is_file() else {}
same_sources=prior.get('cad_sha256')==mesh['source_sha256'] and prior.get('msop_sha256')==doc['sha256']
trials=[row for row in prior.get('results',[]) if row['fai']!=number] if same_sources else []
trials.append({'fai':number,'route':strategy['next_route'],'status':'running',
    'stage':'原文坐标归属复核与全新刚体候选' if args.review_scopes else '完整 XYZ 点组复核',
    'source_point_count':len(points),'at':datetime.now(timezone.utc).isoformat()})
_write(trial_path,{'cad_sha256':mesh['source_sha256'],'msop_sha256':doc['sha256'],'results':trials})
candidate={'known_source_points':points,'required_marker_count':len(points),'evidence':{'fai_strategy':strategy,
    'source_scope_review_requested':args.review_scopes,
    'registration_diagnostic':{'selected_frame_transforms':frames}}}
result=_mark_model_selected_xyz_group(AtomicStore(),project,item,detail,model,mesh,candidate)
out=ROOT/'reports/engineering-checks/20261016'/f'complete-source-route-{pid[:8]}-{datetime.now(timezone.utc).strftime("%H%M%S")}.json'
out.write_text(json.dumps({'project':pid,'fai':number,'source_point_count':len(points),'model':model.info(),
    'all_source_coordinates_preserved':True,'historical_registration_is_seed_only':True,'result':result},ensure_ascii=False,indent=2))
trial_path=store._dir(pid)/'automatic-learning'/'xyz-route-trials.json'
prior=json.loads(trial_path.read_text()) if trial_path.is_file() else {}
same_sources=prior.get('cad_sha256')==mesh['source_sha256'] and prior.get('msop_sha256')==doc['sha256']
trials=[row for row in prior.get('results',[]) if row['fai']!=number] if same_sources else []
trials.append({'fai':number,'route':strategy['next_route'],'status':(result or {}).get('result'),
    'reason':(result or {}).get('reason'),'geometry_diagnostic':(result or {}).get('base_geometry_diagnostic'),
    'source_point_count':len(points),'report':str(out),'at':datetime.now(timezone.utc).isoformat()})
_write(trial_path,{'cad_sha256':mesh['source_sha256'],'msop_sha256':doc['sha256'],
    'classification_is_not_registration_accuracy':True,'results':trials})
print('RESULT',number,(result or {}).get('result'),(result or {}).get('reason'),'REPORT',out,flush=True)
