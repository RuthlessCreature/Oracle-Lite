"""Run the isolated local FAI-match adapter over the held-out FAI examples."""
from __future__ import annotations
import json
import re
import argparse
from pathlib import Path
from talker.model import TalkerModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True,
                        help='isolated experiment root containing oracle.yaml and data/heldout.jsonl')
    args = parser.parse_args()
    root = args.root
    holdout = root / 'data/heldout.jsonl'
    rows = [json.loads(line) for line in holdout.read_text(encoding='utf-8').splitlines() if line.strip()]
    model = TalkerModel(root)
    bundle = model.load()
    outputs = []
    for row in rows:
        messages = [{'role':'user','content':row['question'],
                     'attachments':[{'filename':row['source_path'],'parsed_text':row['context'],'images':[]}]}]
        with model._lock:
            text, limited, count = model._generate_once(messages,max_context_chars=5000,
                                                          max_new_tokens=256,max_images=0)
        cleaned=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip(),flags=re.I)
        try:
            parsed=json.loads(cleaned); valid=isinstance(parsed,dict)
        except json.JSONDecodeError:
            parsed=None;valid=False
        expected=json.loads(row['answer'])
        outputs.append({'fai':row['fai'],'expected_decision':expected['decision'],
                        'expected_face_ids':expected['face_ids'],'valid_json':valid,
                        'decision':parsed.get('decision') if parsed else None,
                        'face_ids':parsed.get('face_ids') if parsed else None,
                        'decision_exact':bool(parsed and parsed.get('decision')==expected['decision']),
                        'face_ids_exact':bool(parsed and parsed.get('face_ids')==expected['face_ids']),
                        'truncated':limited,'generated_tokens':count,'output':text})
        print(f"FAI {row['fai']}: json={valid}, decision={outputs[-1]['decision']}, faces_exact={outputs[-1]['face_ids_exact']}",flush=True)
    report={'adapter':str(bundle.adapter_dir),'base_model':str(bundle.base_model_path),
            'heldout_count':len(outputs),'valid_json_count':sum(x['valid_json'] for x in outputs),
            'decision_exact_count':sum(x['decision_exact'] for x in outputs),
            'face_ids_exact_count':sum(x['face_ids_exact'] for x in outputs),
            'label_quality':'weak/unreviewed; this is a format/consistency check, not an accuracy estimate',
            'items':outputs}
    path=root/'evaluation-heldout.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='items'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
