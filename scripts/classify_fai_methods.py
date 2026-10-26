import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from talker.config import load_config
from talker.model import TalkerModel
from oracle_viewer.store import ViewerStore
from oracle_viewer.fai_strategy import classify_project
model=TalkerModel(load_config().training_output_dir,openai_base_url='http://127.0.0.1:8081',openai_model='qwen3.8-flash-next-iq2_xs',backend_label='strata-native');model.load()
store=ViewerStore(ROOT/'talker/state/viewer')
import argparse
parser=argparse.ArgumentParser(description='Classify MSOP FAI methods using only the local model')
parser.add_argument('projects',nargs='+')
parser.add_argument('--batch-size',type=int,default=5,choices=range(1,11))
args=parser.parse_args()
for pid in args.projects:
 print('CLASSIFY_START',pid,flush=True)
 result=classify_project(store,pid,model,batch_size=args.batch_size)
 print('CLASSIFY_END',pid,result['status'],'classified',len(result['items']),'errors',len(result['errors']),flush=True)
