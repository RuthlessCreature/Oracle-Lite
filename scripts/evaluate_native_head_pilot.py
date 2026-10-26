import argparse,json,os
from train_native_head_pilot import OUT,DATA,examples,request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--adapter',action='store_true');p.add_argument('--all-test',action='store_true');p.add_argument('--output',type=Path,default=OUT);p.add_argument('--dataset',type=Path,default=DATA);p.add_argument('--reload-adapter',action='store_true');a=p.parse_args();OUT=a.output;DATA=a.dataset;OUT.mkdir(parents=True,exist_ok=True);rows=[]
selected=[('test',json.loads(x)) for x in (DATA/'test.jsonl').read_text().splitlines()] if a.all_test else examples()
for split,row in selected:
 if split!='test':continue
 body={'model':os.environ.get('ORACLE_VIEWER_MODEL','qwen3.8-flash-next-iq2_xs'),'messages':[{'role':'system','content':'Extract only explicit source XYZ. Return JSON exactly, do not infer CAD.'},{'role':'user','content':row['question']+'\n'+row['context']}],'chat_template_kwargs':{'enable_thinking':False},'temperature':0,'max_tokens':int(os.environ.get('ORACLE_EVAL_MAX_TOKENS','8192')),'oracle_adapter':a.adapter,'oracle_adapter_reload':a.reload_adapter and len(rows)==0}
 result=request(body);text=result['choices'][0]['message'].get('content','');expected=row['answer'];expected=json.loads(expected) if isinstance(expected,str) else expected
 try:actual=json.loads(text);valid=True
 except Exception:actual=None;valid=False
 rows.append({'project_group':row['project_group'],'page':row['page'],'valid_json':valid,'exact_structured_answer':actual==expected,'finish_reason':result['choices'][0]['finish_reason'],'output':text,'expected':expected,'usage':result.get('usage')})
 print(json.dumps({'page':row['page'],'valid_json':valid,'exact':actual==expected}),flush=True)
(OUT/(('full-heldout-' if a.all_test else 'end-to-end-')+('adapter.json' if a.adapter else 'baseline.json'))).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
