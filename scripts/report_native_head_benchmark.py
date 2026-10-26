"""Separate strict output contract from semantic coordinate correctness."""
import json,re,math
from pathlib import Path
from train_native_head_pilot import OUT

def metrics(rows):
 results=[]
 for r in rows:
  raw=r['output'].strip();plain=re.sub(r'^```(?:json)?\s*|\s*```$','',raw,flags=re.I)
  schema_valid=False
  try:
   raw_parsed=json.loads(raw)
   schema_valid=isinstance(raw_parsed,dict) and set(raw_parsed)=={'points'} and isinstance(raw_parsed['points'],list) and all(isinstance(p,dict) and set(p)=={'name','xyz_mm'} and isinstance(p['name'],str) and isinstance(p['xyz_mm'],list) and len(p['xyz_mm'])==3 and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in p['xyz_mm']) for p in raw_parsed['points'])
  except Exception:pass
  try:
   parsed=json.loads(plain)
   # A bare point array violates the output contract but may preserve all source
   # names/numbers. Do not report its correct coordinates as lost coordinates.
   point_rows=parsed if isinstance(parsed,list) else parsed['points']
   actual={p['name']:p.get('xyz_mm',[p.get('x'),p.get('y'),p.get('z')]) for p in point_rows};expected={p['name']:p['xyz_mm'] for p in r['expected']['points']}
   valid_numbers=lambda v:isinstance(v,list) and len(v)==3 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in v)
   exact_semantic=actual==expected and len(actual)==len(point_rows) and all(valid_numbers(v) for v in actual.values());correct=sum(actual.get(k)==v and valid_numbers(actual.get(k)) for k,v in expected.items());extra=len(set(actual)-set(expected))
  except Exception:exact_semantic=False;correct=0;extra=None;expected={p['name']:p['xyz_mm'] for p in r['expected']['points']}
  results.append({k:r[k] for k in ['project_group','page','valid_json','exact_structured_answer','finish_reason']}|{'schema_valid':schema_valid,'semantic_exact':exact_semantic,'correct_points':correct,'expected_points':len(expected),'extra_points':extra})
 return {'pages':len(results),'strict_json_pages':sum(x['valid_json'] for x in results),'exact_order_pages':sum(x['exact_structured_answer'] for x in results),'schema_valid_pages':sum(x['schema_valid'] for x in results),'semantic_exact_pages':sum(x['semantic_exact'] for x in results),'correct_points':sum(x['correct_points'] for x in results),'expected_points':sum(x['expected_points'] for x in results),'truncated_pages':sum(x['finish_reason']=='length' for x in results),'details':results}
if __name__=='__main__':
 b=metrics(json.loads((OUT/'full-heldout-baseline.json').read_text()));a=metrics(json.loads((OUT/'full-heldout-adapter.json').read_text()));assert [r['page'] for r in b['details']]==[r['page'] for r in a['details']]
 report={'training':'96 token positions on four pages from 613-13599 and 613-16076','heldout':'All 26 source-coordinate examples from 817-07781; no training gradient uses heldout','baseline':b,'adapter':a,'default_enabled':False,'rollout_gate':'Full-source generalization and downstream integration still required; geometry unverified.'}
 (OUT/'full-heldout-report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:{n:v for n,v in row.items() if n!='details'} for k,row in [('baseline',b),('adapter',a)]},indent=2))
