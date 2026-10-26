"""Seed pending-review CAD pins where MSOP point locations map cleanly in XY."""
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

POINT = re.compile(r'X,\s*Y,\s*Z\s*\(mm\)\s*([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([A-Z0-9_]+)', re.I)
SYSTEM = re.compile(r'Measured System\s+([A-Z0-9_&~]+)', re.I)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--origin', default='http://127.0.0.1:7870')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    store = ViewerStore(Path('talker/state/viewer'))
    project = store.get(args.project)
    mesh = store.mesh(args.project)
    vertices = []
    for face in mesh['faces']:
        coords = face['positions']
        vertices.extend((face['id'], coords[i:i+3]) for i in range(0, len(coords), 3))
    items = store.items(project)
    output = []
    for item in items:
        if item['annotation'] or item['binding']:
            continue
        pages = store.detail(args.project, item['number'])['methods']
        text = '\n'.join(page['text'] for page in pages)
        points = POINT.findall(text)
        if not points or len(points) > 100:
            continue
        systems = set(SYSTEM.findall(text))
        if systems == {'ALI_ABC'} or systems == {'ALI_ABC_A&ALI_ABC'}:
            y_offset = -64.5
            transform_name = 'ALI_ABC→CAD: X 镜像，Y 平移 -64.5 mm（由 MSOP/CAD 同名基准位置拟合）'
        elif systems == {'ALI_ABD'} or systems == {'ALI_ITER'}:
            y_offset = 0.0
            transform_name = 'ALI_ABD/ALI_ITER→CAD: X 镜像，Y 零偏移（由 MSOP/CAD 特征位置拟合）'
        else:
            continue
        anchors = []
        residuals = []
        for x_text, y_text, _z_text, name in points:
            target_x, target_y = -float(x_text), float(y_text) + y_offset
            distance, (face_id, position) = min(
                (math.hypot(point[0]-target_x, point[1]-target_y), (fid, point))
                for fid, point in vertices)
            residuals.append(distance)
            anchors.append({'face_id': face_id, 'position': position, 'label_offset': [32, -26]})
        maximum = max(residuals)
        if maximum > 1.0:
            continue
        names = ', '.join(dict.fromkeys(p[3] for p in points))
        evidence = (f'MSOP 第 {", ".join(map(str, item["pages"]))} 页的 {len(points)} 个明确采点 {names}；'
                    f'{transform_name}；逐点投影到 CAD 网格最近表面位置，XY 残差最大 {maximum:.3f} mm。'
                    'Z 未用于定位，需在对照视图人工复核遮挡面/实际采点面。')
        if not args.dry_run:
            payload = {'anchors': anchors, 'status': 'pending_review', 'note': evidence,
                       'origin': 'geometry_candidate', 'cad_sha256': mesh['source_sha256'],
                       'document_id': item['document_id']}
            req = urllib.request.Request(
                f"{args.origin}/api/viewer/projects/{args.project}/fai/{item['number']}/annotation",
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=60) as response:
                json.load(response)
        output.append({'fai': item['number'], 'page': item['pages'], 'points': len(points),
                       'max_xy_residual_mm': round(maximum, 3), 'status': 'pending_review'})
    print(json.dumps({'dry_run': args.dry_run, 'count': len(output), 'candidates': output}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
