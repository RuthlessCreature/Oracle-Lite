"""Run all-project engineering checks locally and freeze reproducible reports."""
import argparse
import json
from pathlib import Path
from oracle_viewer.store import ViewerStore
from oracle_viewer.workflow import compile_tasks,coordinate_tasks,library
from oracle_viewer.engineering import build_report,csv_report,audit_library
from oracle_viewer.adaptation import export_adaptation
from oracle_viewer.automatic_learning import run_automatic_learning


def main():
    p=argparse.ArgumentParser();p.add_argument('--store',type=Path,default=Path('talker/state/viewer'))
    p.add_argument('--output',type=Path,default=Path('reports/engineering-checks'));args=p.parse_args()
    store=ViewerStore(args.store);args.output.mkdir(parents=True,exist_ok=True);reports=[]
    for summary in store.listing():
        pid=summary['id'];data=store.get(pid)
        print('Checking',summary['drawing_no'],flush=True)
        mesh=store.mesh(pid) if (data.get('model') or {}).get('status') in ('ready','preview') else None
        tasks=coordinate_tasks(store,data,compile_tasks(store,data))
        run_path=store._dir(pid)/'aoi-workflow.json'
        run=json.loads(run_path.read_text()) if run_path.is_file() else None
        if run:
            from oracle_viewer.workflow import fingerprint
            run['stale']=run.get('input_fingerprint')!=fingerprint(data,library())
        report=build_report(store,data,tasks,mesh,library(),run);reports.append(report)
        (args.output/(pid+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        (args.output/(pid+'.csv')).write_text(csv_report(report))
        print(json.dumps(report['coverage'],ensure_ascii=False),flush=True)
    cat=audit_library(library(),Path('oracle_viewer/catalog/assets'))
    (args.output/'library-audit.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2))
    manifest=export_adaptation(store,reports,args.output/'adaptation',dict(model_alias='qwen3.8-flash-next-iq2_xs',inference_backend='strata-native'))
    automatic=run_automatic_learning(store,args.output/'automatic-source-curriculum',dict(model_alias='qwen3.8-flash-next-iq2_xs',inference_backend='strata-native'))
    (args.output/'summary.json').write_text(json.dumps(dict(projects=[dict(project_id=r['project_id'],drawing=r['drawing'],coverage=r['coverage'],geometry=r['geometry']) for r in reports],adaptation=manifest,automatic_source_curriculum=automatic,invalid_library_assets=cat['invalid_assets']),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
