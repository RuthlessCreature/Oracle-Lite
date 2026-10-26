"""Run on the cloud host: retain original container, isolate inference on GPU 1.

Reuses existing read-only checkpoint and image. On startup failure it restores
the original service. This script never prints authentication configuration.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def command(*args):
    result = subprocess.run(list(args), text=True, capture_output=True)
    if result.returncode:
        # Container arguments can contain the existing API credential. Keep
        # full command lines out of exceptions and Viewer-visible receipts.
        raise RuntimeError(f'{args[0]} {args[1]} failed with exit code {result.returncode}')
    return result.stdout.strip()


def set_option(args, name, value=None):
    result = []
    skip = False
    for item in args:
        if skip:
            skip = False
            continue
        if item == name:
            skip = value is not None
            continue
        if item.startswith(name + '='):
            continue
        result.append(item)
    if value is not None:
        result.append(name + '=' + str(value))
    return result


def main():
    root = Path('/home/ubuntu/wjkj/oracle-training')
    root.mkdir(parents=True, exist_ok=True)
    old = json.loads(command('docker', 'inspect', 'qwen38-vllm'))[0]
    backup = root / 'original-inference-container.private.json'
    if not backup.exists():
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(old, stream)
    args = old['Config']['Cmd'][:]
    for name, value in {
        '--tensor-parallel-size': 1, '--max-model-len': 65536,
        '--gpu-memory-utilization': .87, '--max-num-seqs': 2,
        '--max-num-batched-tokens': 4096,
    }.items():
        args = set_option(args, name, value)
    args = set_option(args, '--speculative-config')
    args += ['--enforce-eager', '--enable-lora', '--max-lora-rank=16']
    mounts = []
    for mount in old['Mounts']:
        suffix = '' if mount['RW'] else ':ro'
        mounts += ['-v', mount['Source'] + ':' + mount['Destination'] + suffix]
    mounts += ['-v', str(root) + ':/oracle-training:ro']
    name = 'oracle-qwen-inference'
    existing = subprocess.run(['docker', 'inspect', name], capture_output=True)
    if existing.returncode != 0:
        command('docker', 'create', '--name', name, '--gpus', 'device=1', '--ipc=host',
                '--restart', 'unless-stopped', '-p', '127.0.0.1:55556:8000',
                '-e', 'VLLM_ALLOW_RUNTIME_LORA_UPDATING=1', *mounts, old['Image'], *args)
    command('docker', 'stop', '--time', '60', 'qwen38-vllm')
    try:
        command('docker', 'start', name)
        for _ in range(600):
            state = json.loads(command('docker', 'inspect', name))[0]['State']
            if not state['Running']:
                raise RuntimeError('Isolated inference exited; see protected cloud logs')
            try:
                with urllib.request.urlopen('http://127.0.0.1:55556/health', timeout=2) as response:
                    if response.status == 200:
                        report = {'status': 'ready', 'inference_gpu': 1, 'training_gpu': 0,
                                  'model_downloaded': False, 'base_checkpoint_read_only': True,
                                  'original_container_retained': True, 'max_model_len': 65536,
                                  'port': 55556, 'updated_at': time.time()}
                        (root / 'gpu-layout.json').write_text(json.dumps(report, indent=2))
                        print(json.dumps(report), flush=True)
                        return
            except OSError:
                pass
            time.sleep(1)
        raise TimeoutError('Isolated inference startup timed out')
    except Exception:
        subprocess.run(['docker', 'stop', name], capture_output=True)
        command('docker', 'start', 'qwen38-vllm')
        raise


if __name__ == '__main__':
    main()
