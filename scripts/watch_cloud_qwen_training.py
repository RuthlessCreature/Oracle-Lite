"""Mirror cloud training receipts into the local Viewer without credentials."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def snapshot(host, key, root, run_id='source-reading-20261023', dataset_name='source-reading'):
    code = """import json,subprocess,time
from pathlib import Path
root=Path(ROOT_VALUE);run=root/'runs'/RUN_VALUE
def read(path):
 return json.loads(path.read_text()) if path.is_file() else {}
pipeline=read(run/'pipeline.json');stage=pipeline.get('stage')
details=read(run/stage/'status.json') if stage in ('train','evaluate','evaluate-vllm','preflight') else {}
if stage=='failed':
 candidates=[p for p in [run/'train/status.json',run/'evaluate/status.json',run/'evaluate-vllm/status.json'] if p.is_file()]
 if candidates:details=read(max(candidates,key=lambda p:p.stat().st_mtime))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
trained=read(run/'train/status.json');receipt=read(run/'train/training-report.json')
training_summary=dict(status=trained.get('status'),steps=trained.get('global_step'),max_steps=trained.get('max_steps'),parameters=trained.get('trainable_parameters'),adapter_sha256=receipt.get('adapter_sha256'),metrics=receipt.get('metrics'))
print(json.dumps(dict(pipeline=pipeline,progress=details,training=training_summary,dataset=read(root/DATASET_VALUE/'manifest.json'),gpu=gpu,synced_at=time.time()),ensure_ascii=False))
""".replace('ROOT_VALUE', repr(root)).replace('RUN_VALUE', repr(run_id)).replace('DATASET_VALUE',repr(dataset_name))
    remote = 'python3 -c ' + shlex.quote(code)
    result = subprocess.run(['ssh', '-i', key, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                             '-o', 'ConnectTimeout=8', host, remote], text=True,
                            capture_output=True, check=True, timeout=25)
    return json.loads(result.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--remote-root', default='/home/ubuntu/wjkj/oracle-training')
    parser.add_argument('--run-id', default='source-reading-20261023')
    parser.add_argument('--dataset-name',default='source-reading')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round-status',type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            value = snapshot(args.host, args.key, args.remote_root, args.run_id, args.dataset_name)
            value['connection_error'] = None
            if args.round_status and args.round_status.is_file():
                value['round'] = json.loads(args.round_status.read_text())
        except Exception as error:
            value = json.loads(args.output.read_text()) if args.output.is_file() else {}
            value['connection_error'] = type(error).__name__ + ': cloud receipt unavailable'
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        tmp.replace(args.output)
        if args.once:
            break
        time.sleep(10)
