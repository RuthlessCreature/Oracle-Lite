"""Actual-model checks on unseen fictional images, isolated from corpus retrieval."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from .visual_instruction_fixtures import build_cases
from .grounding_stress import score_json_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=202609302)
    parser.add_argument('--count', type=int, default=16)
    args = parser.parse_args()
    cases = build_cases(args.report_dir/'images', seed=args.seed, count=args.count)
    (args.report_dir/'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in cases))
    from talker.model import TalkerModel
    model = TalkerModel(args.output_dir)
    model.load()
    model.retriever.context = lambda *a, **k: ''
    model.retriever.curated_context = lambda *a, **k: None
    model.retriever.visual_context = lambda *a, **k: []
    results = []
    with (args.report_dir/'answers.jsonl').open('w') as stream:
        for index, case in enumerate(cases):
            start = time.monotonic()
            answer = model.answer([{'role':'user', 'content':case['question'], 'attachments':[{
                'filename':Path(case['source_path']).name, 'parsed_text':case['context'],
                'images':case['images']}]}])
            score = score_json_mapping(answer, case['expected_mapping'])
            result = {'id':index+1,'family':case['fixture_family'],'answer':answer,
                      'expected':case['expected_mapping'],
                      'exact_match':score['json_only'],'score':score,
                      'seconds':round(time.monotonic()-start,2)}
            results.append(result)
            stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
            print(json.dumps(result,ensure_ascii=False),flush=True)
    report={'seed':args.seed,'cases':len(results),'passed':sum(r['exact_match'] for r in results),
            'model':model.info(),'by_family':{str(family):{
                'cases':sum(r['family']==family for r in results),
                'passed':sum(r['family']==family and r['exact_match'] for r in results)} for family in range(4)},
            'note':'New random values and image instances in four known layout families. Does not establish accuracy on all drawings or unseen layout families.'}
    (args.report_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
