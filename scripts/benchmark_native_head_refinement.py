"""Full-answer regression plus separately labelled synthetic source-table checks."""
import argparse, fcntl, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from train_native_head_pilot import request
from report_native_head_benchmark import metrics

def challenges(dataset):
    rows=[json.loads(s) for s in (dataset/'test.jsonl').read_text().splitlines()]
    result=[]
    for index,row in enumerate(rows[::max(1,len(rows)//8)][:8]):
        expected=json.loads(row['answer']) if isinstance(row['answer'],str) else row['answer']
        points=[]
        for n,p in enumerate(expected['points']):
            points.append({'name':f'CHK_{index:02d}_PT_{n:03d}', 'xyz_mm':[round(float(v)+(axis+1)*.137,6) for axis,v in enumerate(p['xyz_mm'])]})
        context='Synthetic coordinate-reading test, not a project geometry label.\nLocation Definitions\nNAME | X (mm) | Y (mm) | Z (mm)\n'
        context+='\n'.join(f'{p["name"]} | '+ ' | '.join(str(v) for v in p['xyz_mm']) for p in points[::-1])
        result.append(dict(project_group='synthetic-source-format-robustness',page=index,question='Extract every listed point. Return only JSON {"points":[{"name":"...","xyz_mm":[0,0,0]}]}. Preserve names and coordinates; no markdown.',context=context,answer={'points':points}))
    return result

def evaluate(rows,output,tag,reload=True):
    records=[]
    for n,row in enumerate(rows):
        expected=json.loads(row['answer']) if isinstance(row['answer'],str) else row['answer']
        body={'model':os.environ.get('ORACLE_VIEWER_MODEL','qwen3.8-flash-next-iq2_xs'),'messages':[{'role':'system','content':'Extract only explicit source XYZ. Return JSON exactly, do not infer CAD.'},{'role':'user','content':row['question']+'\n'+row['context']}],'chat_template_kwargs':{'enable_thinking':False},'temperature':0,'max_tokens':2048,'oracle_adapter':True,'oracle_adapter_reload':reload and n==0}
        start=time.monotonic();response=request(body);choice=response['choices'][0];text=choice['message'].get('content','')
        try:actual=json.loads(text);valid=True
        except (ValueError,TypeError):actual=None;valid=False
        records.append(dict(project_group=row['project_group'],page=row['page'],output=text,expected=expected,valid_json=valid,exact_structured_answer=actual==expected,finish_reason=choice['finish_reason'],elapsed_seconds=time.monotonic()-start,usage=response.get('usage')))
        (output/f'{tag}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
        progress=dict(stage=tag,done=n+1,total=len(rows),page=row['page']);(output/'benchmark-progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    return metrics(records)

def acceptance(before,after):
    semantic=after['semantic_exact_pages']==after['pages'] and after['correct_points']==after['expected_points'] and after['truncated_pages']==0
    nonregression=all(after[k]>=before[k] for k in ['semantic_exact_pages','correct_points','schema_valid_pages']) and after['truncated_pages']<=before['truncated_pages']
    better=after['schema_valid_pages']>before['schema_valid_pages'] or after['correct_points']>before['correct_points'] or after['truncated_pages']<before['truncated_pages']
    return semantic and nonregression,better

def run(a):
    a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'benchmark.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active=a.active;original=active.read_bytes();original_sha=hashlib.sha256(original).hexdigest()
    if original_sha!=hashlib.sha256(a.prior.read_bytes()).hexdigest():raise ValueError('Active adapter differs from declared prior')
    (a.output/'active-adapter-backup.oha').write_bytes(original)
    full=[json.loads(s) for s in (a.dataset/'test.jsonl').read_text().splitlines()];synthetic=challenges(a.dataset)
    (a.output/'synthetic-test-cases.json').write_text(json.dumps(synthetic,ensure_ascii=False,indent=2))
    def install(data):
        temp=active.with_suffix('.refinement.tmp');temp.write_bytes(data);temp.replace(active)
    try:
        before=evaluate(full,a.output,'full-heldout-prior');before_extra=evaluate(synthetic,a.output,'synthetic-prior',reload=False)
        install(a.candidate.read_bytes());after=evaluate(full,a.output,'full-heldout-candidate');after_extra=evaluate(synthetic,a.output,'synthetic-candidate',reload=False)
        valid,improved=acceptance(before,after);extra_valid,extra_improved=acceptance(before_extra,after_extra)
        accepted=valid and extra_valid and (improved or extra_improved)
        report=dict(prior=before,candidate=after,synthetic_prior=before_extra,synthetic_candidate=after_extra,
            accepted_as_source_candidate=accepted,default_enabled=False,geometry_improvement_measured=False,
            promotion_reason='完整回答无回退，且至少一项完整回答指标改善' if accepted else '未满足完整回答无回退及可测改善要求，保留现有版本',
            benchmark_scope='Original 817-07781 source pages are repeated regression, not a fresh blind project. Eight renamed/shifted synthetic tables test source reading only, not CAD.',
            prior_sha256=original_sha,candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest())
        (a.output/'benchmark-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        training=json.loads((a.output/'training-report.json').read_text());training.update(promotion_rejected=not accepted,full_heldout=dict(prior={k:v for k,v in before.items() if k!='details'},adapter={k:v for k,v in after.items() if k!='details'}),full_synthetic=dict(prior={k:v for k,v in before_extra.items() if k!='details'},adapter={k:v for k,v in after_extra.items() if k!='details'}),promotion_reason=report['promotion_reason'])
        (a.output/'training-report.json').write_text(json.dumps(training,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in report.items() if not isinstance(v,dict)},ensure_ascii=False),flush=True)
    finally:
        install(original)
        request({'model':os.environ.get('ORACLE_VIEWER_MODEL','qwen3.8-flash-next-iq2_xs'),'messages':[{'role':'user','content':'Return OK.'}],'chat_template_kwargs':{'enable_thinking':False},'temperature':0,'max_tokens':1,'oracle_adapter':False,'oracle_adapter_reload':True})
        if hashlib.sha256(active.read_bytes()).hexdigest()!=original_sha:raise RuntimeError('Adapter restore mismatch')
        (a.output/'restore-receipt.json').write_text(json.dumps(dict(restored=True,sha256=original_sha,default_enabled=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--active',type=Path,default=Path('/home/cyg/Strata/data/oracle-source-head-active.oha'));run(p.parse_args())
