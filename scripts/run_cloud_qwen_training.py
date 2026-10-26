"""Cloud-host job: train a separate adapter, then compare with frozen base."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/ubuntu/wjkj/oracle-training')
RUN = ROOT / 'runs/source-reading-20261023'
DATASET_NAME = 'source-reading'


def status(**values):
    values.update(updated_at=time.time(), model='qwen3.8-27b',
                  scope=('FAI直接测量清单确认' if DATASET_NAME=='fai-manifest' else 'MSOP图文坐标读取'), geometry_training=False,
                  XML_role='虚拟拍摄图与测量方案生成的最终产物', XML_used_as_input=False,
                  deployment_status='not_deployed')
    temp = RUN / 'pipeline.tmp'
    temp.write_text(json.dumps(values, ensure_ascii=False, indent=2))
    temp.replace(RUN / 'pipeline.json')


def execute(stage, extra):
    status(status='running', stage=stage)
    args = ['docker', 'exec', 'oracle-qwen-trainer', 'python3', '-u', '/training/train_cloud_qwen.py',
            '--base', '/models/base', '--dataset', f'/training/{DATASET_NAME}',
            '--output', f'/training/runs/{RUN.name}/{stage}', '--stage', stage, *extra]
    with (RUN / f'{stage}.log').open('w') as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run-id', default='source-reading-20261023')
    parser.add_argument('--dataset-name', default=DATASET_NAME)
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('run-id must be a simple directory name')
    ROOT = args.root
    RUN = ROOT / 'runs' / args.run_id
    if Path(args.dataset_name).name != args.dataset_name or args.dataset_name in ('', '.', '..'):
        parser.error('dataset-name must be a simple directory name')
    DATASET_NAME = args.dataset_name
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        preflight = json.loads((RUN / 'preflight/preflight.json').read_text())
        if not preflight.get('passed'):
            raise ValueError('Source/image preflight not passed')
        execute('train', [])
        status(status='running',stage='evaluate-vllm')
        with (RUN/'vllm-evaluation.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'start_cloud_qwen_evaluator.py'),
                '--root',str(ROOT),'--dataset-name',DATASET_NAME],stdout=log,stderr=subprocess.STDOUT,check=True)
            layout=json.loads((ROOT/'evaluator-layout.json').read_text())
            subprocess.run([sys.executable,str(ROOT/'evaluate_cloud_qwen_vllm.py'),
                '--root',str(ROOT),'--run-id',RUN.name,'--dataset-name',DATASET_NAME,
                '--candidate','oracle-source-'+RUN.name,'--endpoint',layout['endpoint'],
                '--workers',str(layout['max_sequences']),'--stop-evaluator-after'],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    except Exception as error:
        status(status='failed', stage='failed', error=f'{type(error).__name__}: {error}')
        raise
