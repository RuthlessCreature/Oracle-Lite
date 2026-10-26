import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from talker.model import TalkerModel
from talker.config import load_config
from oracle_viewer.store import ViewerStore
from oracle_viewer.fai_strategy import strategy_state
from oracle_viewer.local_matching import _method_page_images,_is_local_inference_model
from oracle_viewer.part_identity import select_physical_parts
model=TalkerModel(load_config().training_output_dir,openai_base_url='http://127.0.0.1:8081',openai_model='qwen3.8-flash-next-iq2_xs',backend_label='strata-native');model.load()
if not _is_local_inference_model(model):raise RuntimeError('Local model only')
import argparse
parser=argparse.ArgumentParser(description='Validate the local-model body identity stage for classified unresolved FAI methods')
parser.add_argument('project');parser.add_argument('--max-trials',type=int,default=2)
args=parser.parse_args()
store=ViewerStore(ROOT/'talker/state/viewer');pid=args.project;project=store.get(pid);mesh=store.mesh(pid)
strategies=strategy_state(store,pid);tried=set();results=[]
for row in strategies['items']:
 if len(results)>=args.max_trials:break
 route=row['next_route']
 if route not in ('feature_topology_graph','drawing_pattern_to_part') or route in tried:continue
 item=store.detail(pid,row['fai'])['item']
 if (item.get('annotation') or {}).get('status')=='auto_marked':continue
 tried.add(route);detail=store.detail(pid,row['fai']);images=_method_page_images(store,project,detail,model,row['fai'])
 print('LOCAL_PART_IDENTITY_ROUTE',row['fai'],route,flush=True)
 parts,audit=select_physical_parts(store,project,item,detail,model,mesh,images)
 results.append({'fai':row['fai'],'route':route,'parts':parts,'audit':audit})
 output=store._dir(pid)/'automatic-learning'/'strategy-route-trials.json'
 output.write_text(json.dumps({'model':model.info(),'human_chosen_points':False,'results':results},ensure_ascii=False,indent=2))
 print('RESULT',row['fai'],audit['status'],audit.get('decision'),flush=True)
print('DONE',len(results),flush=True)
