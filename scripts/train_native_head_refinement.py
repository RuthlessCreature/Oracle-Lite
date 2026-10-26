"""Train a syntax-only residual on a frozen, previously checked native adapter.

Source captures are reused only after SHA validation. Test captures are never read.
Page-grouped dev selection excludes the warm-start adapter's training pages.
"""
import argparse, hashlib, json, os, re, struct, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_native_head_pilot import tokenizer

def read_adapter(path):
    data=Path(path).read_bytes();magic,width,rank,count,vocab,clip=struct.unpack('<IIIIIf',data[:24]);offset=24
    if magic!=0x3141484f or not 0<rank<=64 or not 0<count<=4096 or not 0<clip<=2:
        raise ValueError('Invalid native adapter')
    arrays=[]
    for shape,dtype in [((count,),'<u4'),((width,),'<f4'),((width,),'<f4'),((width,rank),'<f4'),((count,rank),'<f4'),((count,),'<f4')]:
        n=int(np.prod(shape));arrays.append(np.frombuffer(data,dtype=dtype,count=n,offset=offset).copy().reshape(shape));offset+=n*4
    if offset!=len(data) or len(set(arrays[0].tolist()))!=count or max(arrays[0])>=vocab or any(not np.isfinite(x).all() for x in arrays[1:]) or (arrays[2]<=0).any():
        raise ValueError('Corrupt native adapter')
    return width,rank,vocab,clip,arrays

def page_partition(rows,prior_rows):
    prior={(r['project_group'],r['page']) for r in prior_rows if r['split']=='train'}
    keys=sorted({(r['project_group'],r['page']) for r in rows})
    dev=set()
    for group in sorted({k[0] for k in keys}):
        eligible=[k for k in keys if k[0]==group and k not in prior]
        eligible.sort(key=lambda k:hashlib.sha256(f'native-source-dev-v1:{k[0]}:{k[1]}'.encode()).hexdigest())
        dev.update(eligible[:max(1,len(eligible)//5)])
    return np.array([(r['project_group'],r['page']) not in dev for r in rows]),dev

def syntax_token_ids(tok):
    candidates=set()
    for text in ['{"points":[{"name":"","xyz_mm":[]}]}','```json\n','\n```',' : ', '\n  ', '}]', ',\n', '\n{', 'points','name','xyz_mm']:
        candidates.update(tok.encode(text,parse_special=False))
    # Avoid learning new scores for point names, coordinates, digits or x/y/z.
    def allowed(t):
        s=tok.decode([t]);return bool(s) and (not re.search(r'[A-Za-z0-9]',s) or s.strip() in {'points','name','xyz','_mm','xyz_mm','json'})
    return sorted(t for t in candidates if allowed(t))

def run(a):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.captures/'capture-manifest.json').read_text())
    rows=[r for r in manifest if r['split']=='train']
    if not rows or not {r['project_group'] for r in rows}.isdisjoint({r['project_group'] for r in manifest if r['split']=='test'}):
        raise ValueError('Drawing-group train/test leakage')
    prior=json.loads((a.prior.parent/'capture-manifest.json').read_text())
    fit,dev=page_partition(rows,prior)
    width,rank,vocab,clip,(ids,mean,scale,old_u,old_v,old_b)=read_adapter(a.prior)
    syntax=syntax_token_ids(tokenizer());subset=np.array(sorted(set(ids.tolist())|set(syntax)),dtype=np.uint32)
    lookup={int(t):i for i,t in enumerate(subset)};syntax_ix=np.array([lookup[t] for t in syntax])
    v0=np.zeros((len(subset),rank),np.float32);b0=np.zeros(len(subset),np.float32)
    for i,t in enumerate(ids):v0[lookup[int(t)]]=old_v[i];b0[lookup[int(t)]]=old_b[i]
    mask=np.ones(vocab,bool);mask[subset]=False
    xs=[];bs=[];parts=[];targets=[];labels=[];out_max=[];out_ids=[]
    for n,r in enumerate(rows):
        raw=(a.captures/'captures'/f'{r["capture_id"]}.bin').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['capture_sha256']:raise ValueError('Capture SHA mismatch')
        magic,w,vc=struct.unpack('<III',raw[:12])
        if magic!=0x3143464f or w!=width or vc!=vocab or len(raw)!=12+4*(w+vc):raise ValueError('Capture dimensions mismatch')
        arr=np.frombuffer(raw[12:],dtype='<f4');x=arr[:w];logits=arr[w:]
        if not np.isfinite(arr).all():raise ValueError('Nonfinite capture')
        other=logits[mask];peak=float(other.max());parts.append(peak+float(np.log(np.exp(other-peak).sum())))
        xs.append((x-mean)/scale);bs.append(logits[subset]);targets.append(float(logits[r['label']]));labels.append(int(r['label']));out_max.append(peak);out_ids.append(int(np.flatnonzero(mask)[other.argmax()]))
        if n%500==0:print(json.dumps({'stage':'loading_checked_train_captures','done':n,'total':len(rows)}),flush=True)
    x=torch.tensor(np.stack(xs));base=torch.tensor(np.stack(bs));outside=torch.tensor(parts);target_base=torch.tensor(targets)
    indices=torch.tensor([lookup.get(t,-1) for t in labels]);sel=torch.tensor(fit)
    old_raw=x@torch.tensor(old_u)@torch.tensor(v0).T+torch.tensor(b0)
    old_scores=base+old_raw.clamp(-clip,clip)
    u=torch.nn.Parameter(torch.randn(width,a.rank)*0.001);v=torch.nn.Parameter(torch.zeros(len(syntax),a.rank));bias=torch.nn.Parameter(torch.zeros(len(syntax)))
    optimizer=torch.optim.AdamW([u,v,bias],lr=a.learning_rate,weight_decay=.01)
    def scores():
        delta=(x@u@v.T+bias).clamp(-a.residual_budget,a.residual_budget)
        raw=old_raw.clone();raw[:,syntax_ix]+=delta
        return base+raw.clamp(-clip,clip),delta
    def nll(s,subset_mask):
        z=torch.logaddexp(torch.logsumexp(s,1),outside)
        target=torch.where(indices>=0,s[torch.arange(len(rows)),indices.clamp(min=0)],target_base)
        return (z-target)[subset_mask].mean()
    history=[];best=(float(nll(old_scores,~sel)),0,{k:p.detach().clone() for k,p in [('u',u),('v',v),('bias',bias)]})
    for step in range(1,a.steps+1):
        optimizer.zero_grad();s,delta=scores();loss=nll(s,sel)+a.regularization*delta[sel].square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_([u,v,bias],1);optimizer.step()
        if step%5==0:
            with torch.no_grad():
                s,delta=scores();d=float(nll(s,~sel));train=float(nll(s,sel))
            history.append(dict(step=step,fit_nll=train,dev_nll=d))
            if d<best[0]-1e-6:best=(d,step,{k:p.detach().clone() for k,p in [('u',u),('v',v),('bias',bias)]})
    with torch.no_grad():
        for k,p in [('u',u),('v',v),('bias',bias)]:p.copy_(best[2][k])
    # Bound the exported raw residual too: no training-only clipping mismatch.
    with torch.no_grad():raw_delta=x@u@v.T+bias;maximum=float(raw_delta.abs().max());factor=min(1,a.residual_budget/max(maximum,1e-12));v.mul_(factor);bias.mul_(factor)
    vu=np.zeros((len(subset),a.rank),np.float32);bu=b0.copy();vu[syntax_ix]=v.detach().numpy();bu[syntax_ix]+=bias.detach().numpy()
    new_u=np.concatenate([old_u,u.detach().numpy()],axis=1);new_v=np.concatenate([v0,vu],axis=1)
    exported=base+torch.tensor(np.stack(xs)@new_u@new_v.T+bu).clamp(-clip,clip)
    semantic_ix=[i for i,t in enumerate(subset) if int(t) not in syntax]
    max_semantic_change=float((exported[:,semantic_ix]-old_scores[:,semantic_ix]).abs().max())
    if max_semantic_change>2e-5:raise ValueError('Semantic token scores changed')
    with (a.output/'adapter.oha').open('wb') as f:
        f.write(struct.pack('<IIIIIf',0x3141484f,width,rank+a.rank,len(subset),vocab,clip))
        for arr in [subset,mean,scale,new_u,new_v,bu]:f.write(arr.astype('<u4' if arr.dtype==np.uint32 else '<f4').tobytes())
    def accuracy(s,selection):
        val,ix=s.max(1);pred=np.where(val.detach().numpy()>np.array(out_max),subset[ix.numpy()],out_ids)
        return float((pred[selection]==np.array(labels)[selection]).mean())
    report=dict(method='frozen_native_backbone_syntax_residual_on_verified_adapter',adapter_trained=True,base_weights_changed=False,geometry_training=False,
        seed=a.seed,steps=a.steps,selected_step=best[1],train_samples=int(fit.sum()),dev_samples=int((~fit).sum()),test_samples_used=0,
        train_pages=len({(r['project_group'],r['page']) for r in rows})-len(dev),dev_pages=[list(k) for k in sorted(dev)],
        heldout_policy='No test captures read; page-grouped dev excludes prior adapter training pages. Final benchmark project has been evaluated in prior rounds and is not a fresh blind test.',
        prior_adapter_sha256=hashlib.sha256(a.prior.read_bytes()).hexdigest(),capture_manifest_sha256=hashlib.sha256((a.captures/'capture-manifest.json').read_bytes()).hexdigest(),
        rank=rank+a.rank,new_trainable_parameters=u.numel()+v.numel()+bias.numel(),syntax_token_ids=syntax,residual_budget=a.residual_budget,
        residual_export_scale=factor,semantic_logits_max_change=max_semantic_change,
        fit_nll_before=float(nll(old_scores,sel)),fit_nll_after=float(nll(exported,sel)),dev_nll_before=float(nll(old_scores,~sel)),dev_nll_after=float(nll(exported,~sel)),
        fit_top1_before=accuracy(old_scores,fit),fit_top1_after=accuracy(exported,fit),dev_top1_before=accuracy(old_scores,~fit),dev_top1_after=accuracy(exported,~fit),
        history=history,deployed=False,adapter_sha256=hashlib.sha256((a.output/'adapter.oha').read_bytes()).hexdigest())
    (a.output/'training-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='history'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--captures',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=160);p.add_argument('--rank',type=int,default=4);p.add_argument('--seed',type=int,default=61313599)
    p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--residual-budget',type=float,default=.5);p.add_argument('--regularization',type=float,default=.05)
    a=p.parse_args()
    if not 0<a.rank<=32 or not 0<a.steps<=1000 or not 0<a.residual_budget<=1:p.error('Invalid training controls')
    run(a)
