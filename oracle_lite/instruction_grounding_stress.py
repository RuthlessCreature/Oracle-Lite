"""Held-out values for missing facts, scoped values and field associations."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from .instruction_fixtures import build_fixtures
from .grounding_stress import parse_answer, score_json_mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=202609303)
    parser.add_argument('--count', type=int, default=32)
    args=parser.parse_args()
    if args.count < 1: parser.error('--count must be positive')
    candidates=build_fixtures(seed=args.seed, count=args.count*3+3)
    cases=[c for c in candidates if parse_answer(c['answer']) is not None][:args.count]
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir/'cases.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    from talker.model import TalkerModel
    model=TalkerModel(args.output_dir);model.load()
    model.retriever.context=lambda *a,**k:''
    model.retriever.curated_context=lambda *a,**k:None
    model.retriever.visual_context=lambda *a,**k:[]
    results=[]
    with (args.report_dir/'answers.jsonl').open('w') as stream:
        for index, case in enumerate(cases):
            start=time.monotonic()
            answer=model.answer([{'role':'user','content':case['question'],'attachments':[{
                'filename':case['source_path'],'parsed_text':case['context'],'images':[]}]}])
            expected=parse_answer(case['answer'])
            score=score_json_mapping(answer, expected)
            result={'id':index+1,'kind':case['fixture_kind'],'expected':expected,'answer':answer,
                    'exact_match':score['json_only'],'score':score,
                    'seconds':round(time.monotonic()-start,2)}
            results.append(result);stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
            print(json.dumps(result,ensure_ascii=False),flush=True)
    report={'seed':args.seed,'cases':len(results),'passed':sum(r['exact_match'] for r in results),
            'model':model.info(),'by_kind':{str(kind):{
                'cases':sum(r['kind']==kind for r in results),
                'passed':sum(r['kind']==kind and r['exact_match'] for r in results)} for kind in range(10)},
            'note':'Held-out random values in known training template families, with exact JSON and retrieval disabled. Does not establish unseen-template or general-domain performance.'}
    (args.report_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
