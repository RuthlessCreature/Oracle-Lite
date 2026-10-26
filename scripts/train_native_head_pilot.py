"""Native Strata frozen-backbone output-head pilot; drawing-group heldout stays test only."""
import argparse,json,sys,time,struct,urllib.request,hashlib,shutil,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/home/cyg/Strata/tools')
from strata_tokenizer import Tokenizer
OUT=ROOT/'reports/engineering-checks/20261012/native-head-pilot'
DATA=ROOT/'reports/engineering-checks/20261012/automatic-source-curriculum'
TRAIN_PAGES=2; TEST_PAGES=2; TOKENS_PER_PAGE=24; CAPTURE_OFFSET=0
CAPTURE_ROOT=OUT/'captures'
ENDPOINT=os.environ.get('ORACLE_NATIVE_HEAD_ENDPOINT','http://127.0.0.1:8081').rstrip('/')
def request(body):
 req=urllib.request.Request(ENDPOINT+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 for attempt in range(120):
  try:
   with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=300) as r:return json.load(r)
  except urllib.error.URLError as err:
   if isinstance(err,urllib.error.HTTPError) and err.code not in (502,503):raise
   if attempt==119:raise
   time.sleep(2)
def tokenizer():
 p=Path(os.environ.get('ORACLE_NATIVE_TOKENIZER','/home/cyg/Strata-data/packs/iq2_xs/tokenizer'));v=json.loads((p/'vocab.json').read_text());tokens=['']*len(v)
 for t,i in v.items():tokens[i]=t
 return Tokenizer(tokens,(p/'merges.txt').read_text().split('\n'),json.loads((p/'token_type.json').read_text()))
def examples():
 rows={s:[json.loads(x) for x in (DATA/f'{s}.jsonl').read_text().splitlines()] for s in ('train','test')}
 chosen=[]
 for group in sorted({r['project_group'] for r in rows['train']}):
  part=[r for r in rows['train'] if r['project_group']==group]
  chosen += [('train',r) for r in (sorted(part,key=lambda x:len(str(x['answer'])))[:TRAIN_PAGES] if TRAIN_PAGES else sorted(part,key=lambda x:len(str(x['answer']))))]
 chosen += [('test',r) for r in (sorted(rows['test'],key=lambda x:len(str(x['answer'])))[:TEST_PAGES] if TEST_PAGES else sorted(rows['test'],key=lambda x:len(str(x['answer']))))]
 return chosen
def collect():
 tok=tokenizer();manifest=[];identifier=CAPTURE_OFFSET
 selected_examples=examples()
 for ex,(split,row) in enumerate(selected_examples):
  answer=row['answer'] if isinstance(row['answer'],str) else json.dumps(row['answer'],ensure_ascii=False,separators=(',',':'))
  ids=tok.encode(answer,parse_special=False)
  # Complete source remains in context; pilot supervises at most 24 evenly spaced response positions.
  # The server rejects oracle_teacher_tokens longer than 4096, so supervision stays within the first 4096 response tokens.
  positions=np.unique(np.linspace(0,min(len(ids)-1,4095),min(TOKENS_PER_PAGE,len(ids)),dtype=int)).tolist()
  for pos in positions:
   path=OUT/'captures'/f'{identifier}.bin'
   body={'model':os.environ.get('ORACLE_VIEWER_MODEL','qwen3.8-flash-next-iq2_xs'),'messages':[{'role':'system','content':'Extract only explicit source XYZ. Return JSON exactly, do not infer CAD.'},{'role':'user','content':row['question']+'\n'+row['context']}],'chat_template_kwargs':{'enable_thinking':False},'temperature':0,'max_tokens':1,'oracle_teacher_tokens':ids[:pos],'oracle_capture':identifier,'oracle_adapter':False}
   signature=hashlib.sha256(json.dumps({k:v for k,v in body.items() if k!='oracle_adapter'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
   receipt=path.with_suffix('.json')
   reusable=path.exists() and receipt.exists() and json.loads(receipt.read_text()).get('request_sha256')==signature
   if not reusable:
    result=request(body)
    native_path=CAPTURE_ROOT/f'{identifier}.bin'
    if not native_path.exists():raise RuntimeError(f'capture absent: {identifier}: {result}')
    if native_path.resolve()!=path.resolve():shutil.copyfile(native_path,path)
    receipt.write_text(json.dumps({'request_sha256':signature,'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
   manifest.append({'capture_id':identifier,'request_sha256':signature,'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'split':split,'project_group':row['project_group'],'page':row['page'],'example':ex,'position':pos,'label':ids[pos],'source_context':row['context'],'source_answer':answer})
   identifier+=1
   if len(manifest)%24==0:
    temp=OUT/'capture-manifest.json.tmp';temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));temp.replace(OUT/'capture-manifest.json')
   progress={'stage':'capture','samples_done':len(manifest),'example_done':ex+1,'examples_total':len(selected_examples),'split':split,'page':row['page']}
   temp=OUT/'capture-progress.json.tmp';temp.write_text(json.dumps(progress));temp.replace(OUT/'capture-progress.json')
   print(json.dumps({'stage':'capture','done':identifier,'split':split,'page':row['page']},ensure_ascii=False),flush=True)
 temp=OUT/'capture-manifest.json.tmp';temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));temp.replace(OUT/'capture-manifest.json')
 return manifest

def train():
 os.environ["CUDA_VISIBLE_DEVICES"]=""  # CPU adapter training must not initialize the occupied inference GPU.
 import torch
 torch.set_num_threads(4);torch.manual_seed(61316076)
 manifest=json.loads((OUT/'capture-manifest.json').read_text())
 labels=np.array([r['label'] for r in manifest]);tr=np.array([r['split']=='train' for r in manifest]);te=~tr
 assert tr.any() and te.any()
 assert set(r['project_group'] for r in manifest if r['split']=='train').isdisjoint(r['project_group'] for r in manifest if r['split']=='test')
 def read(row):
  b=(OUT/'captures'/f"{row['capture_id']}.bin").read_bytes()
  if row.get('capture_sha256'):assert hashlib.sha256(b).hexdigest()==row['capture_sha256'], 'Capture changed since collection'
  magic,width,vocab=struct.unpack('<III',b[:12]);assert magic==0x3143464f and len(b)==12+4*(width+vocab)
  arr=np.frombuffer(b[12:],dtype='<f4');assert np.isfinite(arr).all()
  return width,vocab,arr[:width],arr[width:]
 selected=set(int(t) for t in labels[tr])
 for row in manifest:
  if row['split']=='train':
   width,vocab,_,logits=read(row);selected.update(int(t) for t in np.argpartition(logits,-8)[-8:])
 subset=np.array(sorted(selected),dtype=np.uint32);assert len(subset)<=4096
 mask=np.ones(vocab,bool);mask[subset]=False
 features=[];subscores=[];partitions=[];targets=[];outside_best=[];outside_ids=[];baseline_tokens=[]
 for row in manifest:
  w,vc,x,logits=read(row);assert w==width and vc==vocab
  other=logits[mask];peak=float(other.max());partitions.append(peak+float(np.log(np.exp(other-peak).sum())))
  features.append(x);subscores.append(logits[subset]);targets.append(logits[row['label']]);baseline_tokens.append(int(logits.argmax()))
  outside_best.append(peak);outside_ids.append(int(np.flatnonzero(mask)[other.argmax()]))
 x=np.stack(features);mean=x[tr].mean(0);scale=np.maximum(x[tr].std(0),0.05)
 xt=torch.tensor((x-mean)/scale);bt=torch.tensor(np.stack(subscores));outside=torch.tensor(partitions,dtype=torch.float32)
 target_base=torch.tensor(targets,dtype=torch.float32)
 mapped={int(t):i for i,t in enumerate(subset)}
 target_index=torch.tensor([mapped.get(int(t),-1) for t in labels])
 rank=8;u=torch.nn.Parameter(torch.randn(width,rank)*0.001);v=torch.nn.Parameter(torch.zeros(len(subset),rank));bias=torch.nn.Parameter(torch.zeros(len(subset)))
 optim=torch.optim.AdamW([u,v,bias],lr=.003,weight_decay=.01)
 def scores():return bt+torch.clamp(xt@u@v.T+bias,-2,2)
 def loss(adapted,sel):
  z=torch.logaddexp(torch.logsumexp(adapted,1),outside)
  target=torch.where(target_index>=0,adapted[torch.arange(len(labels)),target_index.clamp(min=0)],target_base)
  return (z-target)[sel].mean()
 initial=float(loss(bt,tr));history=[]
 for step in range(120):
  optim.zero_grad();l=loss(scores(),tr);l.backward();torch.nn.utils.clip_grad_norm_([u,v,bias],1);optim.step()
  if step%20==0:history.append({'step':step,'train_nll':float(l.detach())})
 adapted=scores().detach().numpy();winning=adapted.argmax(1);best=adapted[np.arange(len(labels)),winning];changed_tokens=np.where(best>np.array(outside_best),subset[winning],outside_ids);baseline_tokens=np.array(baseline_tokens)
 report={'method':'frozen_native_backbone_low_rank_output_head','base_weights_changed':False,'adapter_trained':True,'steps':120,'train_samples':int(tr.sum()),'test_samples':int(te.sum()),'parameters':u.numel()+v.numel()+bias.numel(),'subset_tokens':len(subset),'rank':rank,'clip_logits':2,'train_nll_before':initial,'train_nll_after':float(loss(scores(),tr).detach()),'test_nll_before':float(loss(bt,te)),'test_nll_after':float(loss(scores(),te).detach()),'train_top1_before':float((baseline_tokens[tr]==labels[tr]).mean()),'train_top1_after':float((changed_tokens[tr]==labels[tr]).mean()),'test_top1_before':float((baseline_tokens[te]==labels[te]).mean()),'test_top1_after':float((changed_tokens[te]==labels[te]).mean()),'history':history,'deployed':False,'limitation':'Teacher-forced token pilot; not end-to-end FAI geometry or full-answer benchmark.'}
 with (OUT/'adapter.oha').open('wb') as f:
  f.write(struct.pack('<IIIIIf',0x3141484f,width,rank,len(subset),vocab,2))
  for arr in [subset,mean,scale,u.detach().numpy(),v.detach().numpy(),bias.detach().numpy()]:f.write(arr.astype('<u4' if arr.dtype==np.uint32 else '<f4').tobytes())
 (OUT/'training-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['collect','train','all'],default='all')
 p.add_argument('--output',type=Path,default=OUT);p.add_argument('--dataset',type=Path,default=DATA)
 p.add_argument('--train-pages-per-project',type=int,default=2);p.add_argument('--test-pages',type=int,default=2);p.add_argument('--tokens-per-page',type=int,default=24)
 p.add_argument('--capture-root',type=Path,default=CAPTURE_ROOT);p.add_argument('--capture-offset',type=int,default=0)
 a=p.parse_args();OUT=a.output;DATA=a.dataset;TRAIN_PAGES=a.train_pages_per_project;TEST_PAGES=a.test_pages;TOKENS_PER_PAGE=a.tokens_per_page;CAPTURE_ROOT=a.capture_root;CAPTURE_OFFSET=a.capture_offset
 if min(TRAIN_PAGES,TEST_PAGES,CAPTURE_OFFSET)<0 or not 1<=TOKENS_PER_PAGE<=256:p.error('Invalid sampling controls')
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'captures').mkdir(exist_ok=True)
 if a.stage in ('collect','all'):collect()
 if a.stage in ('train','all'):train()
