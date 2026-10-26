"""Repeat changed-schema pages and rescore saved complete answers transparently."""
import argparse,hashlib,json,sys
from pathlib import Path
from benchmark_native_head_refinement import acceptance,evaluate
from report_native_head_benchmark import metrics
from train_native_head_pilot import request

def run(a):
    out=a.output
    before_rows=json.loads((out/'full-heldout-prior.json').read_text());after_rows=json.loads((out/'full-heldout-candidate.json').read_text())
    before=metrics(before_rows);after=metrics(after_rows)
    changed={x['page'] for x,y in zip(before['details'],after['details']) if x['schema_valid']!=y['schema_valid']}
    dataset=[json.loads(s) for s in (a.dataset/'test.jsonl').read_text().splitlines()]
    if [r['page'] for r in before_rows]!=[r['page'] for r in dataset] or [r['page'] for r in after_rows]!=[r['page'] for r in dataset]:raise ValueError('Regression page scope mismatch')
    for answers in [before_rows,after_rows]:
        for response,source in zip(answers,dataset):
            expected=json.loads(source['answer']) if isinstance(source['answer'],str) else source['answer']
            if response['expected']!=expected:raise ValueError('Benchmark reference changed')
    original=a.active.read_bytes();prior_sha=hashlib.sha256(a.prior.read_bytes()).hexdigest()
    if hashlib.sha256(original).hexdigest()!=prior_sha:raise ValueError('Active adapter must be restored before recheck')
    def install(data):
        p=a.active.with_suffix('.recheck.tmp');p.write_bytes(data);p.replace(a.active)
    repeats=[]
    try:
        selected=[r for r in dataset if r['page'] in changed]
        for n in range(a.rounds):
            install(a.prior.read_bytes());old=evaluate(selected,out,f'repeat-prior-{n+1}') if selected else None
            install(a.candidate.read_bytes());new=evaluate(selected,out,f'repeat-candidate-{n+1}') if selected else None
            repeats.append(dict(round=n+1,prior=old,candidate=new))
    finally:
        install(original)
        request({'model':'qwen3.8-flash-next-iq2_xs','messages':[{'role':'user','content':'Return OK.'}],'chat_template_kwargs':{'enable_thinking':False},'temperature':0,'max_tokens':1,'oracle_adapter':False,'oracle_adapter_reload':True})
        (out/'restore-receipt.json').write_text(json.dumps(dict(restored=True,sha256=hashlib.sha256(a.active.read_bytes()).hexdigest(),default_enabled=False)))
    synthetic_before=metrics(json.loads((out/'synthetic-prior.json').read_text()));synthetic_after=metrics(json.loads((out/'synthetic-candidate.json').read_text()))
    ok,better=acceptance(before,after);ok_extra,better_extra=acceptance(synthetic_before,synthetic_after)
    repeated_ok=all(r['candidate']['semantic_exact_pages']==r['candidate']['pages'] and r['candidate']['schema_valid_pages']>=r['prior']['schema_valid_pages'] for r in repeats if r['candidate'])
    repeat_delta=sum(r['candidate']['schema_valid_pages']-r['prior']['schema_valid_pages'] for r in repeats if r['candidate'])
    accepted=ok and ok_extra and (better or better_extra) and repeated_ok and (not changed or repeat_delta>0)
    report=dict(metrics_version='2-coordinate-content-distinct-from-json-root-contract',prior=before,candidate=after,synthetic_prior=synthetic_before,synthetic_candidate=synthetic_after,
        changed_schema_pages=sorted(changed),repeated_changed_pages=repeats,repeated_schema_delta=repeat_delta,
        accepted_as_source_candidate=accepted,default_enabled=False,geometry_improvement_measured=False,
        promotion_reason='完整回答无坐标回退，结构合格页增加，变化页复测通过' if accepted else '未满足完整回答及复测要求，保留现有版本',
        benchmark_scope='Original 817-07781 pages are repeated regression, not a fresh blind project. Eight renamed/shifted synthetic tables test source reading only. Changed pages selected after full comparison are repeatability checks, not independent generalization.',
        prior_sha256=prior_sha,candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),dataset_test_sha256=hashlib.sha256((a.dataset/'test.jsonl').read_bytes()).hexdigest())
    (out/'benchmark-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    training=json.loads((out/'training-report.json').read_text());training.update(promotion_rejected=not accepted,promotion_reason=report['promotion_reason'],metrics_version=report['metrics_version'],full_heldout=dict(prior={k:v for k,v in before.items() if k!='details'},adapter={k:v for k,v in after.items() if k!='details'}),full_synthetic=dict(prior={k:v for k,v in synthetic_before.items() if k!='details'},adapter={k:v for k,v in synthetic_after.items() if k!='details'}),repeatability=dict(pages=sorted(changed),rounds=a.rounds,schema_delta=repeat_delta,passed=repeated_ok))
    (out/'training-report.json').write_text(json.dumps(training,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(dict,list))},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--active',type=Path,default=Path('/home/cyg/Strata/data/oracle-source-head-active.oha'));p.add_argument('--rounds',type=int,default=3);run(p.parse_args())
