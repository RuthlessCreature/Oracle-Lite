"""Project-separated, provenance-preserving adaptation data and deployment preflight."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from .store import now


def training_preflight(model_identity, checkpoint=None):
    path=Path(checkpoint).expanduser() if checkpoint else None
    config=None;reasons=[]
    if path and path.is_dir() and (path/'config.json').is_file():
        config=json.loads((path/'config.json').read_text())
        if model_identity.get('expected_model_type') and config.get('model_type')!=model_identity['expected_model_type']:
            reasons.append('可训练checkpoint架构与当前推理基座不匹配')
        if not any(path.glob('*.safetensors')) and not any(path.glob('*.bin')):
            reasons.append('未找到可训练权重文件')
    else: reasons.append('当前推理模型尚无已配置且匹配的可训练原始checkpoint')
    if model_identity.get('inference_backend')=='strata-native':
        reasons.append('尚未验证Strata主体参数微调后的原生导出及加载链路；已验证的冻结主体输出层适配试验单列')
    if model_identity.get('native_training_backend_available') is False:
        reasons.append('当前Oracle尚未接通该架构的主体参数训练后端；冻结主体输出层适配后端已完成试验')
    required=model_identity.get('training_checkpoint_bytes')
    resource_check=None
    if required:
        target=path if path and path.exists() else Path.cwd()
        free=shutil.disk_usage(target).free
        resource_check=dict(checkpoint_bytes=required,free_disk_bytes=free,checkpoint_present=bool(config))
        if not config and required>free:
            reasons.append(f'完整可训练checkpoint至少需要{required/1024**3:.1f} GiB，当前磁盘空闲{free/1024**3:.1f} GiB；整模型下载/训练路径不满足资源条件')
    return dict(created_at=now(),model_identity=model_identity,checkpoint=str(path) if path else None,
                checkpoint_model_type=(config or {}).get('model_type'),
                resource_check=resource_check,
                status='blocked_prerequisites' if reasons else 'ready_for_resource_check',
                reasons=reasons,weight_training_started=False,weights_changed=False,
                fallback_to_other_base=False)


def export_adaptation(store, reports, output, model_identity):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    grouped={};quarantine=[];project_readiness=[]
    for report in reports:
        project=store.get(report['project_id']);items={i['number']:i for i in store.items(project)}
        project_key=project.get('drawing_no') or project['id']
        counts=dict(project_id=project['id'],drawing_no=project_key,total_fai=len(report['fai']),
                    eligible_labels=0,quarantined_labels=0,quarantine_reasons={})
        for audit in report['fai']:
            item=items[audit['fai']];ann=item.get('annotation') or {}
            row=dict(project_id=project['id'],project_group=project_key,fai=audit['fai'],
                     source_hashes=report['source_hashes'],context=audit['contract'],
                     geometry_checks=audit['anchors'],model_confidence=audit['model_confidence'],
                     reviewed=ann.get('status')=='reviewed')
            reasons=[]
            if not row['reviewed']:reasons.append('annotation_not_reviewed')
            if not audit['anchors']:reasons.append('no_cad_anchors')
            if any(a['status']!='on_display_surface' for a in audit['anchors']):reasons.append('anchors_not_on_display_surface')
            if audit.get('evidence_grade')=='conflicting' or audit.get('source_conflicts') or audit.get('point_scope_conflict_count',0):reasons.append('source_evidence_conflicting')
            if reasons:
                row['quarantine_reasons']=reasons
                counts['quarantined_labels']+=1
                for reason in reasons:counts['quarantine_reasons'][reason]=counts['quarantine_reasons'].get(reason,0)+1
                row['label_quality']='unreviewed_do_not_train_as_ground_truth';quarantine.append(row);continue
            row.update(label_quality='human_reviewed_display_location_not_metrology',
                       answer=dict(fai=audit['fai'],decision='propose',face_ids=[a['face_id'] for a in ann['anchors']],
                                   source_pages=audit['pages'],review_required=True))
            grouped.setdefault(project_key,[]).append(row)
            counts['eligible_labels']+=1
        counts['status']='awaiting_verified_correspondences' if not counts['eligible_labels'] else 'labels_available_training_not_started'
        project_readiness.append(counts)
    keys=sorted(grouped)
    # All revisions of a drawing stay together; no modulo-FAI split leaks the
    # same part geometry into both train and evaluation sets.
    test_groups=keys[-1:] if len(keys)>=2 else []
    train=[r for k in keys if k not in test_groups for r in grouped[k]]
    test=[r for k in test_groups for r in grouped[k]]
    hashes={}
    for name,rows in [('train.jsonl',train),('test.jsonl',test),('quarantine.jsonl',quarantine)]:
        blob=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows).encode()
        (output/name).write_bytes(blob);hashes[name]=hashlib.sha256(blob).hexdigest()
    manifest=dict(created_at=now(),train_count=len(train),test_count=len(test),quarantine_count=len(quarantine),
                  split_unit='drawing_no_all_revisions',train_groups=[k for k in keys if k not in test_groups],test_groups=test_groups,
                  cross_project_evaluation_ready=bool(test),source_projects=[r['project_id'] for r in reports],
                  project_readiness=project_readiness,
                  file_sha256=hashes,preflight=training_preflight(model_identity),
                  status='awaiting_reviewed_labels' if not train else 'dataset_ready_training_not_started')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest
