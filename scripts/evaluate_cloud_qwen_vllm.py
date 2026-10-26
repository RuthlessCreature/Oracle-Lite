"""Cloud-host paired source benchmark on the same vLLM runtime as Viewer.

Candidate registration leaves the application's default base alias unchanged.
Existing XML is never supplied to inference or labels.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import subprocess
from pathlib import Path
import time
import urllib.request

from train_cloud_qwen import load_rows, score_answer, save

SYSTEM = 'Read measurement evidence exactly. Output the requested JSON only. Never infer missing coordinate axes or CAD positions.'


def api_key(root):
    saved = json.loads((root/'original-inference-container.private.json').read_text())
    cmd = saved['Config']['Cmd']
    for index, arg in enumerate(cmd):
        if arg.startswith('--api-key='):
            return arg.split('=', 1)[1]
        if arg == '--api-key':
            return cmd[index+1]
    raise ValueError('Private inference authentication configuration missing')


def request(path, key, payload=None, endpoint='http://127.0.0.1:55556'):
    headers = {'Authorization': 'Bearer ' + key}
    data = None
    if payload is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint.rstrip('/')+path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as response:
        raw = response.read().decode()
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def evaluate(row, mode, args, key, budgets):
    alias = args.candidate if mode == 'adapter' else 'qwen3.8-27b'
    image = base64.b64encode((args.root/getattr(args,'dataset_name','source-reading')/row['images'][0]).read_bytes()).decode()
    content = [
        {'type': 'text', 'text': row['question']+'\nSource: '+row['source_file']+
         f" page {row['page']}\nNative page text:\n"+row['context']},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+image}},
    ]
    body = {'model': alias, 'messages': [{'role':'system','content':SYSTEM},
                                        {'role':'user','content':content}],
            'temperature':0, 'max_tokens':getattr(args,'max_output_tokens',8192),
            'repetition_penalty':1.05, 'chat_template_kwargs':{'enable_thinking':False}}
    started = time.time()
    output = request('/v1/chat/completions',key,body,getattr(args,'endpoint','http://127.0.0.1:55556'))
    choice = output['choices'][0]
    text = choice['message'].get('content') or ''
    score = score_answer(text,row['answer'])
    truncated = choice.get('finish_reason')=='length'
    if truncated:
        score['exact']=False
    return dict(id=row['id'],mode=mode,**score,output=text,truncated=truncated,
                seconds=time.time()-started,usage=output.get('usage'),
                runtime_endpoint=getattr(args,'endpoint','http://127.0.0.1:55556'))


def metrics_for(results):
    return {mode:{'pages':sum(r['mode']==mode for r in results),
                  'exact_pages':sum(r['mode']==mode and r['exact'] for r in results),
                  'valid_json_pages':sum(r['mode']==mode and r['valid_json'] for r in results),
                  'schema_valid_pages':sum(r['mode']==mode and r.get('schema_valid',False) for r in results),
                  'correct_points':sum(r['correct_points'] for r in results if r['mode']==mode),
                  'expected_points':sum(r['expected_points'] for r in results if r['mode']==mode),
                  'invented_points':sum(r['invented_points'] for r in results if r['mode']==mode)}
            for mode in ('base','adapter')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/ubuntu/wjkj/oracle-training'))
    parser.add_argument('--run-id',default='source-reading-20261023')
    parser.add_argument('--candidate',default='oracle-source-candidate-20261023')
    parser.add_argument('--pages',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dataset-name',default='source-reading')
    parser.add_argument('--endpoint',default='http://127.0.0.1:55556')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--stop-evaluator-after',action='store_true')
    parser.add_argument('--max-output-tokens',type=int,default=8192)
    args = parser.parse_args()
    if args.stop_evaluator_after and args.endpoint.rstrip('/')!='http://127.0.0.1:59020':
        parser.error('Cleanup is only allowed for the dedicated private evaluation endpoint')
    run = args.root/'runs'/args.run_id
    out = run/'evaluate-vllm';out.mkdir(parents=True,exist_ok=True)
    pipeline = dict(status='running',stage='evaluate-vllm',model='qwen3.8-27b',
                    scope='MSOP证据读取与FAI测量清单',geometry_training=False,XML_used_as_input=False,
                    XML_role='虚拟拍摄图与测量方案生成的最终产物',deployment_status='not_deployed')
    status = dict(status='running',stage='evaluate-vllm',done=0,total=0,started_at=time.time())
    save(run/'pipeline.json',dict(pipeline,updated_at=time.time()))
    try:
        manifest,rows = load_rows(args.root/args.dataset_name)
        pipeline['scope']=manifest.get('training_scope',pipeline['scope'])
        preflight = json.loads((run/'preflight/preflight.json').read_text())
        budgets = {r['id']:r['supervised_tokens'] for r in preflight['lengths']}
        key = api_key(args.root)
        available = {r['id'] for r in request('/v1/models',key,endpoint=args.endpoint)['data']}
        if args.candidate not in available:
            request('/v1/load_lora_adapter',key,{'lora_name':args.candidate,
                'lora_path':f'/oracle-training/runs/{args.run_id}/train/adapter-final'},endpoint=args.endpoint)
        items = sorted(rows['validation'],key=lambda r:len(r['answer']))
        if args.pages>0:items=items[:args.pages]
        status['total']=len(items)
        save(out/'status.json',status)
        results=[]
        existing=out/'evaluation.json'
        by_id={r['id']:r for r in items}
        if args.resume and existing.is_file():
            previous=json.loads(existing.read_text())
            if previous.get('dataset_sha256')!=manifest['sha256'] or previous.get('candidate')!=args.candidate:
                raise ValueError('Existing evaluation source/candidate fingerprint differs')
            retry_ids={r['id'] for r in previous.get('items',[]) if r.get('truncated') or r.get('error')}
            status['retest_pages']=len(retry_ids)
            for old in previous.get('items',[]):
                if old['id'] in by_id and old['id'] not in retry_ids and not old.get('error'):
                    old.update(score_answer(old['output'],by_id[old['id']]['answer']))
                    if old.get('truncated'):old['exact']=False
                    results.append(old)
        completed={(r['id'],r['mode']) for r in results}
        resumed={mode:{r['id'] for r in results if r['mode']==mode} for mode in ('base','adapter')}
        status.update(done=len(resumed['base']&resumed['adapter']),requests_done=len(results),
                      metrics=metrics_for(results),updated_at=time.time())
        save(out/'status.json',status)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(evaluate,row,mode,args,key,budgets):(row,mode) for row in items for mode in ('base','adapter') if (row['id'],mode) not in completed}
            for future in as_completed(pending):
                try:
                    results.append(future.result())
                except Exception as error:
                    row,mode=pending[future]
                    results.append(dict(id=row['id'],mode=mode,valid_json=False,exact=False,
                        schema_valid=False,correct_points=0,expected_points=len(json.loads(row['answer']).get('points',json.loads(row['answer']).get('source_point_names',[]))),
                        invented_points=0,error=f'{type(error).__name__}: {error}',output=''))
                grouped={mode:{r['id'] for r in results if r['mode']==mode} for mode in ('base','adapter')}
                metrics=metrics_for(results)
                status.update(done=len(grouped['base']&grouped['adapter']),requests_done=len(results),
                              metrics=metrics,updated_at=time.time())
                save(out/'status.json',status)
                save(out/'evaluation.json',dict(complete=len(results)==len(items)*2,metrics=metrics,
                    candidate=args.candidate,dataset_sha256=manifest['sha256'],items=results,
                    runtime='vllm_bfloat16_same_base_and_adapter',XML_used_as_input=False,
                    training_scope=manifest.get('training_scope'),
                    max_output_tokens=args.max_output_tokens,
                    scoring='semantic_coordinates_independent_of_JSON_fences; raw_JSON_compliance_separate',
                    scope='source_coordinates_only_not_FAI_geometry_or_XML'))
        metrics=metrics_for(results)
        errors=sum(bool(r.get('error')) for r in results)
        status.update(status='failed' if errors else 'complete',done=len(items),metrics=metrics,finished_at=time.time())
        save(out/'status.json',status)
        save(out/'evaluation.json',dict(complete=True,metrics=metrics,candidate=args.candidate,
             dataset_sha256=manifest['sha256'],items=results,runtime='vllm_bfloat16_same_base_and_adapter',
             XML_used_as_input=False,training_scope=manifest.get('training_scope'),scoring='semantic_coordinates_or_manifest_decisions; raw_JSON_compliance_separate',
             max_output_tokens=args.max_output_tokens,
             scope='source_coordinates_only_not_FAI_geometry_or_XML'))
        gate=(not errors and
              metrics['adapter']['exact_pages']==len(items) and
              metrics['adapter']['correct_points']==metrics['adapter']['expected_points'] and
              metrics['adapter']['invented_points']==0 and
              metrics['adapter']['valid_json_pages']==len(items) and
              metrics['adapter']['correct_points']>=metrics['base']['correct_points'] and
              metrics['adapter']['invented_points']<=metrics['base']['invented_points'] and
              metrics['adapter']['valid_json_pages']>=metrics['base']['valid_json_pages'])
        save(run/'pipeline.json',dict(pipeline,status='failed' if errors else 'complete',
             stage='failed' if errors else 'evaluated',metrics=metrics,error_count=errors,
             source_gate_passed=gate,updated_at=time.time(),
             limitation='Source reading only; FAI geometry and virtual-shot-to-XML remain separate tasks'))
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}',finished_at=time.time())
        save(out/'status.json',status)
        save(run/'pipeline.json',dict(pipeline,status='failed',stage='failed',error=status['error'],updated_at=time.time()))
        raise
    finally:
        if args.stop_evaluator_after:
            subprocess.run(['docker','stop','oracle-qwen-evaluator'],capture_output=True)


if __name__=='__main__':main()
