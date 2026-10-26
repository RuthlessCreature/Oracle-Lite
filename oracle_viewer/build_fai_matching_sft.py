"""Build weakly supervised, project-local FAI candidate-review instruction data.

The current pins are unreviewed proposals, so this builder labels their provenance
and never upgrades them to authoritative associations. A deterministic held-out FAI
split is written separately and must not be supplied to training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle_viewer.store import ViewerStore


def build(project_id: str, viewer_root: Path, output: Path, holdout_modulus: int = 5):
    store = ViewerStore(viewer_root)
    project = store.get(project_id)
    mesh = store.mesh(project_id)
    pdf_doc = next(d for d in project['documents'] if d['role'] == 'msop')
    faces = {f['id']: f for f in mesh['faces']}
    output.mkdir(parents=True, exist_ok=True)
    train_rows, heldout_rows = [], []
    for item in store.items(project):
        detail = store.detail(project_id, item['number'])
        if not detail['methods']:
            continue
        pages = detail['methods']
        page_no = pages[0]['page']
        text_blocks = []
        for source_page in pages[:1]:
            excerpt = source_page['text'].split('Location Definitions', 1)[0].strip()
            excerpt = re.sub(r'\s+', ' ', excerpt)[:200]
            text_blocks.append(f"[MSOP page {source_page['page']} verified excerpt; location table represented by candidate coordinates below]\n{excerpt}")
        text = '\n\n'.join(text_blocks)
        annotation = item.get('annotation')
        if annotation:
            all_anchors = annotation['anchors']
            anchors = all_anchors[:1]
            face_ids = sorted({a['face_id'] for a in anchors})
            face_catalog = [{
                'face_id': fid,
                'instance': faces[fid].get('instance_path'),
                'surface_type': faces[fid].get('surface'),
                'center_mm': [round(float(v), 3) for v in faces[fid].get('center', [])],
            } for fid in face_ids if fid in faces]
            proposal = {'candidate_face_catalog': face_catalog,
                        'candidate_anchors_mm': [{'face_id': a['face_id'], 'position': [round(float(v), 3) for v in a['position']]} for a in anchors],
                        'candidate_count': len(all_anchors), 'omitted_anchor_count': max(0, len(all_anchors)-len(anchors)), 'provenance': annotation['origin'],
                        'surface_validation': 'Each stored anchor is a vertex on its listed active CAD face.',
                        'review_state': 'automatic candidate; confidence and evidence retained',
                        'note': annotation.get('note', '')[:100]}
            answer_obj = {
                'decision': 'AUTO_MARK', 'fai': item['number'],
                'face_ids': face_ids, 'anchors': proposal['candidate_anchors_mm'],
                'source_pages': item['pages'], 'evidence': annotation.get('note', '')[:100],
                'confidence': 'low',
            }
        else:
            proposal = {'candidate_face_catalog': [], 'candidate_anchors_mm': [],
                        'candidate_count': 0, 'omitted_anchor_count': 0,
                        'provenance': 'geometric_preprocessor_abstained',
                        'reason': ('测量方法没有可直接配准的二维采点，坐标系混合，或最近网格残差超出筛选阈值。')}
            answer_obj = {
                'decision': 'abstain', 'fai': item['number'], 'face_ids': [],
                'anchors': [], 'source_pages': item['pages'],
                'evidence': proposal['reason'], 'confidence': 'low',
            }
        question = (
            '自动判定该 FAI→CAD 标记。候选预处理已验证 face_id 和锚点确实位于当前 CAD 显示网格。'
            '若 candidate_count>0 且 MSOP 特征或采点与候选证据一致，decision=AUTO_MARK。'
            '证据缺失或冲突时 decision=ABSTAIN。字段：decision,fai,face_ids,anchors,source_pages,evidence,confidence。')
        context = (
            f"Drawing {project.get('drawing_no')} Rev {project.get('revision')}; FAI {item['number']}\n"
            f"MSOP extracted fields: {json.dumps({'headings':item['headings'],'features':item['features'],'specifications':item['specifications'],'warnings':item['warnings']},ensure_ascii=False)}\n\n"
            f"Geometry proposal and provenance: {json.dumps(proposal,ensure_ascii=False)}\n\n{text}")
        answer = json.dumps(answer_obj, ensure_ascii=False, separators=(',', ':'))
        row = {'mode':'instruction','text':answer,'answer':answer,'question':question,
               'context':context,'source_path':pdf_doc['filename'],
               'images':[],
               'label_source':'weak_unreviewed_geometry_or_abstention',
               'fai':item['number'],'source_pages':item['pages']}
        (heldout_rows if item['number'] % holdout_modulus == 0 else train_rows).append(row)
    for name, rows in [('train.jsonl',train_rows),('heldout.jsonl',heldout_rows)]:
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest = {'project_id':project_id,'drawing':project['drawing_no'],'revision':project['revision'],
                'train_records':len(train_rows),'heldout_records':len(heldout_rows),
                'train_fai':[r['fai'] for r in train_rows],'heldout_fai':[r['fai'] for r in heldout_rows],
                'label_quality':'weak/unreviewed; derived from geometry proposals and conservative abstentions',
                'input_sha256':hashlib.sha256((output/'train.jsonl').read_bytes()).hexdigest(),
                'holdout_sha256':hashlib.sha256((output/'heldout.jsonl').read_bytes()).hexdigest(),
                'heldout_included_in_training':False,
                'excerpt_policy':'Text-only; normalized one-page excerpt capped at 200 characters; one candidate anchor per example, omission is disclosed.',
                'visual_training':'none; initialized adapter retains its prior visual capability, this run adapts text-grounded decision format only.',
                'warning':'Training this data teaches proposal formatting and weak source/geometry alignment. It does not establish matching accuracy; collect human-reviewed FAI labels before promotion.'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True)
    p.add_argument('--viewer-root',type=Path,default=Path('talker/state/viewer'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.project,a.viewer_root,a.output),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
