"""Source-only native adapter receipts; never create CAD labels or approve geometry."""
import hashlib,json,struct
from pathlib import Path
from .automatic_learning import write_json

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def refinement_status(root):
 """Read the configured refinement receipt; never imply geometry or deployment."""
 root=Path(root).resolve()
 try:
  profile=json.loads((root/'config/training_backends/strata-native.json').read_text())
  name=profile.get('source_output_refinement_report')
  if not name:return None
  path=(root/name).resolve()
  if not any(path.is_relative_to((root/p).resolve()) for p in ['reports','talker/state/viewer/source-head-training']):return None
  report=json.loads(path.read_text())
  if report.get('model_identity',{}).get('model_alias')!=profile['model_alias']:return None
  artifact=path.parent/'adapter.oha'
  if not artifact.is_file() or file_sha(artifact)!=report.get('adapter_sha256'):
   return dict(stage='stale_artifact',report=report,progress=None,default_enabled=False,is_project_geometry_training=False)
  benchmark=path.parent/'benchmark-report.json';progress=path.parent/'benchmark-progress.json'
  status='trained_not_evaluated'
  if progress.is_file() and (not benchmark.is_file() or progress.stat().st_mtime_ns>benchmark.stat().st_mtime_ns):
   status='evaluating_full_answers'
  elif benchmark.is_file():
   b=json.loads(benchmark.read_text());status='validated_source_candidate' if b.get('accepted_as_source_candidate') else 'retained_prior_after_validation'
  return dict(stage=status,report=report,progress=json.loads(progress.read_text()) if progress.is_file() else None,
   default_enabled=False,is_project_geometry_training=False)
 except (OSError,ValueError,KeyError,TypeError):return None

def backbone_fingerprint(paths):
 result=[]
 for name in paths:
  p=Path(name);st=p.stat()
  with p.open('rb') as f:header=hashlib.sha256(f.read(65536)).hexdigest()
  result.append(dict(path=str(p.resolve()),size=st.st_size,mtime_ns=st.st_mtime_ns,header_sha256=header))
 return result

def validated_source_receipt(state,bundle):
 bundle=Path(bundle)
 if not (bundle/'training-report.json').is_file():return None
 try:r=json.loads((bundle/'training-report.json').read_text())
 except (OSError,json.JSONDecodeError):return None
 if not isinstance(r,dict):return None
 adapter=bundle/'adapter.oha'
 if not r.get('adapter_trained') or not adapter.is_file():return None
 if r.get('source_curriculum_sha256')!=state['curriculum']['file_sha256']:return None
 if r.get('model_identity')!=state['model_identity']:return None
 if not r.get('backbone_fingerprint'):return None
 try:
  current=backbone_fingerprint([f['path'] for f in r['backbone_fingerprint']])
  if current!=r['backbone_fingerprint'] or file_sha(adapter)!=r.get('adapter_sha256'):return None
 except OSError:return None
 q=r.get('full_heldout',{}).get('adapter',{})
 if not q.get('pages') or q['pages']!=state['curriculum']['test_count'] or q.get('semantic_exact_pages')!=q['pages'] or q.get('correct_points')!=q.get('expected_points') or q.get('truncated_pages',1) or r.get('promotion_rejected'):return None
 return dict(artifact=str(adapter.resolve()),report=str((bundle/'training-report.json').resolve()),
  adapter_sha256=r['adapter_sha256'],train_samples=r['train_samples'],test_samples=r['test_samples'],
  parameters=r['parameters'],steps=r['steps'],evaluated_pages=q['pages'],correct_points=q['correct_points'],
  expected_points=q['expected_points'],schema_valid_pages=q['schema_valid_pages'],
  training_scope='frozen_backbone_source_xyz_output_head',geometry_training=False,default_enabled=False)

def attach_source_receipt(state,bundle,output):
 receipt=validated_source_receipt(state,bundle)
 if not receipt:return False
 state['source_adapter_receipt']=receipt
 state['phases'].update(source_adapter_training='complete',source_adapter_evaluation='complete')
 state['status']='source_adapter_validated_geometry_pending'
 state['adapter_training_started']=True;state['adapter_weights_changed']=True
 # These legacy flags refer to the full/base model, not the source-only adapter.
 state['weight_training_started']=False;state['weights_changed']=False
 state['note']='原文输出适配已训练并验证；主体权重未改变。几何匹配仍需独立核验，不能用原文输出准确率替代 FAI 放置准确率。'
 write_json(Path(output)/'state.json',state)
 return True

def automatic_source_adapter(state,output,profile,root):
 """Reuse verified artifacts or capture/train/evaluate a new source-only candidate.

 The previous active adapter is restored after testing. No CAD label is created,
 and no candidate is silently promoted for general Talker/Viewer requests.
 """
 import os,subprocess,sys,urllib.request,shutil,tempfile
 root=Path(root);output=Path(output)
 preferred=profile.get('source_adapter_receipt')
 if preferred and attach_source_receipt(state,root/preferred,output):return
 backend=profile.get('source_output_head_backend',{})
 if not backend.get('enabled'):return
 if not state['curriculum']['train_count'] or not state['curriculum']['test_count']:
  state['source_adapter_error']='原文输出适配需要独立项目训练和验证样本。';write_json(output/'state.json',state);return
 identity=state['model_identity']
 if identity.get('inference_backend')!='strata-native':return
 def resolve(name):return (root/Path(name)).resolve()
 capture=resolve(backend['capture_root']);active=resolve(backend['active_adapter'])
 key=hashlib.sha256(json.dumps(dict(files=state['curriculum']['file_sha256'],model=identity),sort_keys=True).encode()).hexdigest()
 candidate=resolve(backend['candidate_root'])/key;candidate.mkdir(parents=True,exist_ok=True)
 if attach_source_receipt(state,candidate,output):return
 old=active.read_bytes() if active.exists() else None
 if old is None:
  state['source_adapter_error']='原生适配器测试位尚未配置，未启动训练。';write_json(output/'state.json',state);return
 def persist(phase):
  state['status']='running';state['phases']['source_adapter_training']=phase;state['source_adapter_candidate']=str(candidate);write_json(output/'state.json',state)
 def command(args,name):
  with (candidate/name).open('w') as log:
   subprocess.run([sys.executable,*args],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','ORACLE_VIEWER_MODEL':identity['model_alias'],'ORACLE_NATIVE_TOKENIZER':backend.get('tokenizer_dir','/home/cyg/Strata-data/packs/iq2_xs/tokenizer')})
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open('http://127.0.0.1:8081/v1/models',timeout=10) as response:models=json.load(response)
  if not any(r.get('id')==identity['model_alias'] for r in models.get('data',[])):raise ValueError('运行中的本地模型与训练模型身份不符')
  original_backbone=backbone_fingerprint(backend['backbone_files'])
  persist('collecting_native_features')
  command(['scripts/evaluate_native_head_pilot.py','--all-test','--dataset',str(output),'--output',str(candidate)],'baseline.log')
  command(['scripts/train_native_head_pilot.py','--stage','all','--dataset',str(output),'--output',str(candidate),'--train-pages-per-project','0','--test-pages','0','--tokens-per-page','24','--capture-root',str(capture),'--capture-offset',str((int(key[:8],16)%16000)*50000)],'training.log')
  persist('evaluating_native_output')
  # Replace only the configured, source-only test adapter atomically.
  temporary=active.with_suffix('.tmp');shutil.copyfile(candidate/'adapter.oha',temporary);temporary.replace(active)
  command(['scripts/evaluate_native_head_pilot.py','--all-test','--adapter','--reload-adapter','--dataset',str(output),'--output',str(candidate)],'adapter-evaluation.log')
  sys.path.insert(0,str(root/'scripts'))
  from report_native_head_benchmark import metrics
  before=metrics(json.loads((candidate/'full-heldout-baseline.json').read_text()));after=metrics(json.loads((candidate/'full-heldout-adapter.json').read_text()))
  if backbone_fingerprint(backend['backbone_files'])!=original_backbone:raise ValueError('训练期间基座文件发生改变，候选未接受')
  report=json.loads((candidate/'training-report.json').read_text())
  report.update(model_identity=identity,source_curriculum_sha256=state['curriculum']['file_sha256'],
   backbone_fingerprint=backbone_fingerprint(backend['backbone_files']),adapter_sha256=file_sha(candidate/'adapter.oha'),
   full_heldout=dict(baseline={k:v for k,v in before.items() if k!='details'},adapter={k:v for k,v in after.items() if k!='details'}))
  accepted=after['pages']==state['curriculum']['test_count'] and after['semantic_exact_pages']==after['pages'] and after['truncated_pages']==0
  report['promotion_rejected']=not accepted;report['deployed']=False
  write_json(candidate/'training-report.json',report)
  write_json(candidate/'full-heldout-report.json',dict(baseline=before,adapter=after,accepted_as_source_candidate=accepted,default_enabled=False))
  if accepted:
   if not attach_source_receipt(state,candidate,output):raise ValueError('训练产物绑定校验未通过')
  else:
   state['status']='source_adapter_rejected_geometry_pending';state['phases'].update(source_adapter_training='complete',source_adapter_evaluation='rejected');state['source_adapter_error']='训练完成，但独立项目坐标校验未全部通过，未启用候选。';write_json(output/'state.json',state)
 except Exception as error:
  state['status']='prepared_training_blocked';state['phases']['source_adapter_training']='failed';state['source_adapter_error']=str(error);write_json(output/'state.json',state)
 finally:
  temporary=active.with_suffix('.restore');temporary.write_bytes(old);temporary.replace(active)
  # Restore the earlier adapter in the live engine as well as on disk.
  try:
   body={'model':identity['model_alias'],'messages':[{'role':'user','content':'Return OK.'}],'chat_template_kwargs':{'enable_thinking':False},'temperature':0,'max_tokens':1,'oracle_adapter':False,'oracle_adapter_reload':True}
   req=urllib.request.Request('http://127.0.0.1:8081/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=60) as response:json.load(response)
  except Exception as error:
   state['source_adapter_restore_error']=str(error);write_json(output/'state.json',state)
