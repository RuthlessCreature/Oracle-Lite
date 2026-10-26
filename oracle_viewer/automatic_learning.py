"""Automatically sourced curriculum, separate from unverified geometric labels."""
import hashlib
import json
import re
from pathlib import Path
from .store import now
from .candidate_mining import COORDINATE_POINT
from .adaptation import training_preflight


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def source_curriculum(store):
    """Train source reading from exact native text; never invent CAD answers."""
    rows=[];excluded=[]
    for summary in store.listing():
        p=store.get(summary['id']);msop=store.active_msop(p)
        if not msop:continue
        for page in msop.get('method_pages',[]):
            text=page.get('text','');points=[]
            seen={};conflicts=[]
            for match in COORDINATE_POINT.finditer(text):
                x,y,z,name=match.groups();xyz=[float(x),float(y),float(z)]
                if name in seen and seen[name]!=xyz:conflicts.append(name)
                if name not in seen:
                    points.append(dict(name=name,xyz_mm=xyz,source_quote=match.group(0)))
                seen[name]=xyz
            if not points:continue
            if conflicts:
                excluded.append(dict(project_id=p['id'],page=page['page'],reason='same_page_coordinate_conflict',names=conflicts));continue
            # Raw text transcription has no CAD/frame identity claim. Program and
            # registration examples need their own explicit provenance checks.
            answer=dict(points=[dict(name=x['name'],xyz_mm=x['xyz_mm']) for x in points])
            row=dict(project_id=p['id'],project_group=p.get('drawing_no') or p['id'],page=page['page'],
                mode='instruction',question='逐项提取本页 Location Definitions 的点名与 XYZ（mm），输出 JSON points 数组；不转换坐标，不推断 CAD 位置。',
                context=text,answer=json.dumps(answer,ensure_ascii=False),source_path=msop.get('filename') or msop.get('name') or 'MSOP',
                source_hashes=dict(msop=msop['sha256'],page_text=hashlib.sha256(text.encode()).hexdigest()),
                label_quality='exact_native_source_transcription_not_geometry',source_quotes=points,
                automatic_label=True,geometry_label=False,metrology_accepted=False)
            row['example_sha256']=digest(row);rows.append(row)
    return rows,excluded


def run_automatic_learning(store,output,model_identity,checkpoint=None,registration_report=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows,excluded=source_curriculum(store)
    groups=sorted({r['project_group'] for r in rows})
    test_groups=groups[-1:] if len(groups)>1 else []
    splits={'train':[r for r in rows if r['project_group'] not in test_groups],
            'test':[r for r in rows if r['project_group'] in test_groups]}
    hashes={}
    for split,values in splits.items():
        data=''.join(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n' for v in values).encode()
        (output/(split+'.jsonl')).write_bytes(data);hashes[split]=hashlib.sha256(data).hexdigest()
    preflight=training_preflight(model_identity,checkpoint)
    geometry=[]
    if registration_report:
        for frame in registration_report.get('frames',[]):
            q=frame.get('triangle_refinement') or {}
            # A close surface distance alone is not semantic verification.
            geometry.append(dict(program=frame['program'],frame=frame['coordinate_system'],
                status=q.get('status',frame['status']),training_eligible=False,
                reason='independent_feature_identity_not_verified',max_residual_mm=(q.get('best') or {}).get('max_residual_mm')))
    state=dict(schema='oracle-automatic-learning-v1',status='prepared_training_blocked',updated_at=now(),model_identity=model_identity,
        input_fingerprint=digest(dict(examples=[r['example_sha256'] for r in rows],model=model_identity,
            registration=digest(registration_report) if registration_report else None)),
        phases=dict(source_labels='complete',project_split='complete',geometry_labels='pending_verified_geometry',
                    weight_training='waiting_trainable_backend',evaluation='not_started',deployment='not_started'),
        curriculum=dict(task='msop_native_xyz_transcription',train_count=len(splits['train']),test_count=len(splits['test']),
            train_groups=[g for g in groups if g not in test_groups],test_groups=test_groups,
            by_project={g:sum(r['project_group']==g for r in rows) for g in groups},excluded_pages=excluded,
            split_unit='drawing_no_all_revisions',file_sha256=hashes),
        geometry_frames=geometry,preflight=preflight,weight_training_started=False,weights_changed=False,
        note='自动原文标签不需要人工逐点标注；未核验的CAD匹配不进入位置训练。原文读取数据集不代表几何训练完成。')
    # Persist a full source-pinned receipt so a later run can determine exactly
    # which curriculum and checkpoint it is resuming rather than reuse stale data.
    write_json(output/'state.json',state)
    return state
