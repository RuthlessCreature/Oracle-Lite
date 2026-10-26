"""Use the released training GPU for batched validation; leave Viewer online."""
import json
import argparse
import math
from pathlib import Path
import struct
import subprocess
import time
import urllib.request

from cloud_qwen_gpu_layout import command, set_option


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/ubuntu/wjkj/oracle-training'))
    parser.add_argument('--dataset-name',default='source-reading')
    options=parser.parse_args()
    root=options.root
    max_area=0
    for image in (root/options.dataset_name/'images').glob('*.png'):
        width,height=struct.unpack('>II',image.read_bytes()[16:24])
        max_area=max(max_area,width*height)
    if not max_area:raise ValueError('Evaluation source images missing')
    processor=json.loads((Path('/home/ubuntu/wjkj/models/Qwen3.8-27B')/'preprocessor_config.json').read_text())
    original_bound=processor.get('size',{}).get('longest_edge',16777216)
    pixel_bound=min(original_bound,max(2097152,math.ceil(max_area*1.1/1024)*1024))
    sequences=max(2,min(8,int(16777216/pixel_bound)))
    free=int(command('nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits').splitlines()[0])
    if free<65000:
        raise ValueError('Training GPU is still occupied; evaluator must not compete with training')
    old=json.loads((root/'original-inference-container.private.json').read_text())
    args=old['Config']['Cmd'][:]
    for name,value in {'--tensor-parallel-size':1,'--max-model-len':65536,
                       '--gpu-memory-utilization':.90,'--max-num-seqs':sequences,
                       '--max-num-batched-tokens':8192}.items():args=set_option(args,name,value)
    args=set_option(args,'--speculative-config')
    args+=['--enforce-eager','--enable-lora','--max-lora-rank=16',
           '--limit-mm-per-prompt={"image":1,"video":0}',
           '--mm-processor-kwargs='+json.dumps({'max_pixels':pixel_bound})]
    mounts=[]
    for m in old['Mounts']:mounts+=['-v',m['Source']+':'+m['Destination']+('' if m['RW'] else ':ro')]
    mounts+=['-v',str(root)+':/oracle-training:ro']
    name='oracle-qwen-evaluator'
    found=subprocess.run(['docker','inspect',name],capture_output=True)
    create=bool(found.returncode)
    if not create:
        existing=json.loads(found.stdout)[0]
        if existing['State']['Running']:
            raise RuntimeError('A dedicated evaluator is already running; leave its job untouched')
        if existing['Config']['Cmd']!=args or existing['Image']!=old['Image']:
            command('docker','rm',name)
            create=True
    if create:
        command('docker','create','--name',name,'--gpus','device=0','--ipc=host',
                '-p','127.0.0.1:59020:8000','-e','VLLM_ALLOW_RUNTIME_LORA_UPDATING=1',
                *mounts,old['Image'],*args)
    command('docker','start',name)
    for _ in range(600):
        info=json.loads(command('docker','inspect',name))[0]
        if not info['State']['Running']:raise RuntimeError('Isolated evaluator failed to start')
        try:
            with urllib.request.urlopen('http://127.0.0.1:59020/health',timeout=2) as response:
                if response.status==200:
                    (root/'evaluator-layout.json').write_text(json.dumps({'max_sequences':sequences,
                        'pixel_bound':pixel_bound,'endpoint':'http://127.0.0.1:59020'},indent=2))
                    print('dedicated-evaluator-ready; Viewer remains on GPU 1',flush=True)
                    break
        except OSError:pass
        time.sleep(1)
    else:raise TimeoutError('Evaluation server startup timed out')
