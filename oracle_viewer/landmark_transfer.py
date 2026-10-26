"""Geometry-only hypotheses from exact landmarks previously reviewed locally.

Prior annotations are low-confidence evidence, never ground truth. The current
whole FAI still needs a fresh local-model choice and all ordinary geometry gates.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def peer_landmark_frame_options(store,project,item,source_points,selected_frames):
    from .local_matching import _rigid_correspondence
    cad=(project.get('model') or {}).get('source_sha256');msop=store.active_msop(project) or {}
    peers=[];key='逐点原文 XYZ 与 CAD 投影：'
    for peer in store.items(project):
        if peer['number']==item['number']:continue
        annotation=peer.get('annotation') or {};note=annotation.get('note') or ''
        if (annotation.get('status')!='auto_marked' or annotation.get('origin')!='local_model_candidate'
            or annotation.get('source_sha256')!=cad or annotation.get('document_sha256')!=msop.get('sha256') or key not in note):continue
        try:rows,_=json.JSONDecoder().raw_decode(note.split(key,1)[1])
        except (ValueError,TypeError):continue
        if not isinstance(rows,list):continue
        peers.append((peer['number'],rows,hashlib.sha256(json.dumps(annotation,sort_keys=True).encode()).hexdigest()))
    scopes={ (row.get('program'),row.get('coordinate_system')) for row in source_points }
    results=[]
    for peer_number,rows,digest in peers:
        trial=[];changed=False
        for frame in selected_frames:
            scope=(frame.get('program'),frame.get('coordinate_system'))
            current=[p for p in source_points if (p.get('program'),p.get('coordinate_system'))==scope]
            if not current:trial.append(frame);continue
            common=[]
            for point in current:
                candidates=[row for row in rows if row.get('name')==point.get('name')
                    and (row.get('program'),row.get('coordinate_system'))==scope
                    and row.get('source_xyz')==point.get('xyz')
                    and isinstance(row.get('transformed_xyz'),list) and len(row['transformed_xyz'])==3]
                if len(candidates)==1:common.append((point,candidates[0]))
            if len(common)<2:trial.append(frame);continue
            source=np.asarray([pair[0]['xyz'] for pair in common],float)
            target=np.asarray([pair[1]['transformed_xyz'] for pair in common],float)
            if not np.isfinite(source).all() or not np.isfinite(target).all():trial.append(frame);continue
            rank=np.linalg.matrix_rank(source-source.mean(axis=0),tol=1e-6)
            fit=_rigid_correspondence(source,target,max_residual_mm=.05) if rank>=2 else None
            if fit:
                rotation=fit['rotation'];translation=fit['translation'];basis='noncollinear_exact_shared_source_landmarks'
            elif rank==1:
                # A line cannot authorize a frame for off-line source points.
                all_source=np.asarray([p['xyz'] for p in current],float)
                if np.linalg.matrix_rank(all_source-source[0],tol=1e-6)!=1:trial.append(frame);continue
                pair=np.unravel_index(np.linalg.norm(source[:,None]-source[None,:],axis=2).argmax(),(len(source),len(source)))
                a=source[pair[1]]-source[pair[0]];b=target[pair[1]]-target[pair[0]]
                rotation=np.asarray(frame.get('rotation'),float)
                if rotation.shape!=(3,3):trial.append(frame);continue
                a=rotation@a;a/=np.linalg.norm(a);b/=np.linalg.norm(b)
                cross=np.cross(a,b);sine=float(np.linalg.norm(cross));cosine=float(a@b)
                if cosine<np.cos(np.deg2rad(30)):trial.append(frame);continue
                delta=Rotation.from_rotvec(cross/sine*np.arctan2(sine,cosine)).as_matrix() if sine>1e-10 else np.eye(3)
                rotation=delta@rotation;translation=np.median(target-source@rotation.T,axis=0)
                basis='collinear_source_group_only_with_prior_orientation'
            else:trial.append(frame);continue
            residual=float(np.linalg.norm(source@rotation.T+translation-target,axis=1).max())
            if residual>.05 or not np.isclose(np.linalg.det(rotation),1,atol=1e-5):trial.append(frame);continue
            trial.append({**frame,'rotation':rotation.tolist(),'translation_mm':translation.tolist(),
                'peer_landmark_graph':{'source_fai':peer_number,'annotation_sha256':digest,
                    'shared_names':[p['name'] for p,_ in common],'basis':basis,'fit_residual_mm':residual,
                    'prior_annotation_is_ground_truth':False,'requires_fresh_local_model_selection':True}})
            changed=True
        if changed and {(f.get('program'),f.get('coordinate_system')) for f in trial}>=scopes:results.append(trial)
    return results[:8]
