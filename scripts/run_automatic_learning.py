"""Build the automatic curriculum and verify native training prerequisites."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from oracle_viewer.store import ViewerStore
from oracle_viewer.automatic_learning import run_automatic_learning


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--registration',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--project');p.add_argument('--resume',action='store_true')
    p.add_argument('--training-profile',type=Path)
    args=p.parse_args()
    backend=os.environ.get('ORACLE_VIEWER_MODEL_BACKEND','strata-native')
    if args.training_profile is None:
        profile_name='strata-native' if backend=='strata-native' else 'qwen-cloud'
        args.training_profile=ROOT/f'config/training_backends/{profile_name}.json'
    if args.project:
        args.output.mkdir(parents=True,exist_ok=True)
        args.registration=args.output/'registration.json'
        cmd=[sys.executable,'scripts/auto_register_msop.py','--project',args.project,'--output',str(args.registration)]
        if args.resume:cmd.append('--resume')
        subprocess.run(cmd,check=True)
    report=json.loads(args.registration.read_text()) if args.registration else None
    identity=dict(inference_backend=backend,model_alias=os.environ.get('ORACLE_VIEWER_MODEL','qwen3.8-flash-next-iq2_xs'))
    if args.training_profile.is_file():
        profile=json.loads(args.training_profile.read_text())
        if profile.get('model_alias')==identity['model_alias']:identity.update(profile['identity'])
    result=run_automatic_learning(ViewerStore(Path('talker/state/viewer')),args.output,identity,
        checkpoint=args.checkpoint or os.environ.get('ORACLE_STRATA_TRAIN_CHECKPOINT'),registration_report=report)
    if args.training_profile.is_file():
        profile=json.loads(args.training_profile.read_text())
        if (backend=='strata-native' and profile.get('model_alias')==identity['model_alias']
                and (profile.get('source_adapter_receipt') or profile.get('source_output_head_backend'))):
            from oracle_viewer.native_head_training import automatic_source_adapter
            automatic_source_adapter(result,args.output,profile,ROOT)
    print(json.dumps(dict(phases=result['phases'],curriculum=result['curriculum'],preflight=result['preflight']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
