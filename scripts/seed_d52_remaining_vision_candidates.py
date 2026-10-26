"""Create reviewable visual candidates from explicitly named MSOP vision circles."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle_viewer.store import ViewerStore

VISION = re.compile(r'\b(CIR_N[1-4]_VISION)\b')
AXIS = re.compile(r'Dimension\s+([XY])\s+Value', re.I)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--origin', default='http://127.0.0.1:7870')
    args = parser.parse_args()
    store = ViewerStore(Path('talker/state/viewer'))
    data = store.get(args.project)
    mesh = store.mesh(args.project)
    items = store.items(data)
    detail = {i['number']: store.detail(args.project, i['number']) for i in items}
    feature_coords = {}
    for item in items:
        n = item['number']
        if not detail[n]['methods']:
            continue
        text = '\n'.join(m['text'] for m in detail[n]['methods'])
        feature = next(iter(VISION.findall(text)), None)
        axis = AXIS.search(text)
        if feature and axis and item['specifications']:
            try:
                feature_coords.setdefault(feature, {})[axis.group(1).upper()] = float(item['specifications'][0]['nominal'])
            except (TypeError, ValueError):
                pass
    vertices = []
    for face in mesh['faces']:
        coords = face['positions']
        for i in range(0, len(coords), 3):
            vertices.append((face['id'], coords[i:i+3]))
    existing = {i['number']: i for i in items}
    created = []
    for item in items:
        n = item['number']
        if item['annotation'] or item['binding']:
            continue
        text = '\n'.join(m['text'] for m in detail[n]['methods'])
        feature = next(iter(VISION.findall(text)), None)
        if not feature:
            continue
        anchors = None
        evidence = ''
        # The dimension pages identify the same vision feature by name. Pair its
        # X and Y dimensions, then apply the measured X sign convention established
        # by the already source-linked N1/N4 candidates.
        xy = feature_coords.get(feature, {})
        if 'X' in xy and 'Y' in xy:
            target = (-xy['X'], xy['Y'])
            distance, (face_id, point) = min(
                (math.hypot(p[0]-target[0], p[1]-target[1]), (fid, p)) for fid, p in vertices)
            if distance <= 0.65:
                anchors = [{'face_id': face_id, 'position': point, 'label_offset': [34, -30]}]
                evidence = (f'MSOP 同名特征 {feature} 的 X/Y 维度页配对；以 N1/N4 已核对候选推定 X 镜像后，'
                           f'标称平面坐标映射到 CAD 显示网格最近点，XY 残差 {distance:.3f} mm。')
        # Repeated dimension rows use an explicitly identical named physical feature.
        if anchors is None and n in (80, 81):
            donor = 24 if n == 80 else 23
            source = existing.get(donor, {}).get('annotation')
            if source:
                anchors = source['anchors']
                evidence = f'MSOP 明确命名为 {feature}，复用同名已关联候选 FAI {donor} 的 CAD 显示位置。'
        if anchors is None:
            continue
        payload = {'anchors': anchors, 'status': 'pending_review',
                   'note': evidence + ' 自动候选，仅供位置复核；不是坐标配准、测量结果或计量验收。',
                   'origin': 'geometry_candidate', 'cad_sha256': mesh['source_sha256'],
                   'document_id': item['document_id']}
        req = urllib.request.Request(
            f"{args.origin}/api/viewer/projects/{args.project}/fai/{n}/annotation",
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            saved = json.load(response)
        created.append({'fai': n, 'feature': feature, 'status': saved['status'], 'note': evidence})
    print(json.dumps({'created': created, 'count': len(created)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
