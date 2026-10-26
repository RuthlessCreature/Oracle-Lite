"""Generate source-pinned, program-separated automatic CAD registration proposals."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from oracle_viewer.store import ViewerStore
from oracle_viewer.candidate_mining import extract_named_points
from oracle_viewer.auto_registration import (automatic_fit,refine_surface_fit,
    registration_algorithm_sha256,surface_seed_candidates)
from oracle_viewer.surface_index import SurfaceIndex
from oracle_viewer.adaptation import training_preflight
from oracle_viewer.automatic_learning import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--state-root',type=Path,default=ROOT/'talker/state/viewer');args=parser.parse_args()
    store=ViewerStore(args.state_root);p=store.get(args.project)
    msop=store.active_msop(p);mesh=store.mesh(args.project)
    algorithm_hash=registration_algorithm_sha256(ROOT)
    if args.resume and args.output.is_file():
        old=json.loads(args.output.read_text())
        if (old.get('project_id')==args.project and old.get('cad_sha256')==mesh['source_sha256']
                and old.get('msop_sha256')==msop['sha256'] and old.get('algorithm_sha256')==algorithm_hash):
            print('Reusing source-and-algorithm-matched registration',args.output,flush=True);return
    samples=[];triangles=[];excluded_faces=[]
    for face in mesh['faces']:
        if face.get('occt_face_valid') is False:
            excluded_faces.append(face['id']);continue
        xyz=np.asarray(face['positions'],float).reshape(-1,3)
        tri=xyz[np.asarray(face['indices'],int).reshape(-1,3)]
        samples.extend([xyz,tri.mean(1)])
        triangles.append(tri)
    samples=np.unique(np.concatenate(samples),axis=0)
    triangles=np.concatenate(triangles)
    surface_index=SurfaceIndex(triangles)
    grouped={};omitted=[]
    for name,value in extract_named_points(msop['method_pages']).items():
        for row in value.get('occurrences',[]):
            key=(row.get('program'),row.get('coordinate_system'))
            if key[1] is None or (key[0] is None and (row.get('coordinate_frame_source')!='explicit_output_system_table'
                    or any(page.get('program') for page in msop['method_pages']))):
                omitted.append(dict(name=name,reason='missing_program_or_frame'));continue
            grouped.setdefault(key,{}).setdefault(name,[]).append(row)
    results=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    progress_path=args.output.with_name('registration-progress.json')
    for (program,frame),by_name in sorted(grouped.items()):
        write_json(progress_path,dict(status='running',done=len(results),total=len(grouped),message=f'正在配准 {program} / {frame}'))
        rows=[];conflicts=[]
        for name,values in sorted(by_name.items()):
            if any(v.get('frame_assignment_conflict') for v in values) or any(not np.allclose(values[0]['xyz'],v['xyz'],atol=1e-6,rtol=0) for v in values[1:]):
                conflicts.append(dict(name=name,occurrences=values));continue
            rows.append(dict(name=name,xyz=values[0]['xyz'],pages=sorted({v['page'] for v in values})))
        print(program,frame,'points',len(rows),'conflicts',len(conflicts),flush=True)
        result=automatic_fit([r['xyz'] for r in rows],samples)
        if result.get('best'):
            seeds=surface_seed_candidates([r['xyz'] for r in rows],triangles,index=surface_index)
            result['triangle_refinement']=refine_surface_fit([r['xyz'] for r in rows],triangles,
                [result['best']]+result['alternatives'][:1]+seeds,index=surface_index)
            result['surface_seed_count']=len(seeds)
            # A coordinate system can contain a few unrelated point families
            # (e.g. frame perimeter plus nut centers). Such outliers can pull
            # unpaired ICP away from the dominant physical feature. Keep
            # family fits as explicitly labeled alternatives; the local model
            # still chooses the transform and every FAI is checked separately.
            families={}
            for row in rows:
                family=row['name'].split('_',1)[0] if '_' in row['name'] else row['name']
                families.setdefault(family,[]).append(row)
            family_hypotheses=[]
            for family,family_rows in sorted(families.items()):
                if len(family_rows)<5:
                    continue
                family_source=[row['xyz'] for row in family_rows]
                family_fit=automatic_fit(family_source,samples)
                if not family_fit.get('best'):
                    continue
                refined=refine_surface_fit(family_source,triangles,
                    [family_fit['best'],*(family_fit.get('alternatives') or [])[:1]],index=surface_index)
                family_hypotheses.append(dict(
                    feature_group=family,
                    support_point_names=[row['name'] for row in family_rows],
                    support_point_count=len(family_rows),
                    triangle_refinement=refined))
            result['feature_family_hypotheses']=family_hypotheses
        results.append(dict(program=program,coordinate_system=frame,source_points=rows,conflicts=conflicts,**result))
    report=dict(project_id=args.project,drawing_no=p.get('drawing_no'),cad_sha256=mesh['source_sha256'],
        msop_sha256=msop['sha256'],algorithm_sha256=algorithm_hash,
        cad_sample_count=len(samples),frames=results,omitted=omitted,weights_changed=False,
        excluded_invalid_display_face_ids=excluded_faces,
        training_preflight=training_preflight(dict(inference_backend='strata-native',model_alias='qwen3.8-flash-next-iq2_xs')))
    write_json(args.output,report)
    write_json(progress_path,dict(status='complete',done=len(results),total=len(grouped),message='坐标系逐组检查完成'))
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
