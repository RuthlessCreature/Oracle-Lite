"""Freeze MSOP image/text supervision for cloud Qwen; XML is never input.

This first stage trains source coordinate reading. It deliberately does not
turn model-generated CAD annotations or virtual shots into geometry truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oracle_viewer.candidate_mining import COORDINATE_POINT
from oracle_viewer.msop_tables import feature_coordinate_rows
from oracle_viewer.store import ViewerStore


def sha(data):
    return hashlib.sha256(data).hexdigest()


def coordinate_rows(page):
    rows = []
    for x, y, z, name in COORDINATE_POINT.findall(page.get('text', '')):
        rows.append({'name': name, 'axes': ['X', 'Y', 'Z'],
                     'values_mm': [float(x), float(y), float(z)]})
    for row in feature_coordinate_rows(page):
        if row['axes'] in (['X', 'Y'], ['X', 'Y', 'Z']):
            rows.append({'name': row['name'], 'axes': row['axes'],
                         'values_mm': [float(v) for v in row['values']]})
    unique = {}
    for row in rows:
        key = row['name'], tuple(row['axes'])
        if key in unique and unique[key] != row:
            raise ValueError('conflicting same-page point: ' + row['name'])
        unique[key] = row
    return list(unique.values())


def prepare(store, output, holdout, excluded):
    import fitz
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    splits = {'train': [], 'validation': []}
    skipped = []
    for summary in store.listing():
        project = store.get(summary['id'])
        group = project.get('drawing_no') or project['id']
        if group in excluded:
            continue
        source = store.active_msop(project)
        if not source:
            continue
        pdf_path = Path(source['path'])
        if sha(pdf_path.read_bytes()) != source['sha256']:
            raise ValueError('MSOP content fingerprint changed: ' + group)
        with fitz.open(pdf_path) as pdf:
            for page in source.get('method_pages', []):
                try:
                    points = coordinate_rows(page)
                except ValueError as error:
                    skipped.append({'project': group, 'page': page['page'], 'reason': str(error)})
                    continue
                if not points:
                    continue
                image = Path('images') / f"{source['sha256'][:16]}-{page['page']:04d}.png"
                pix = pdf[page['page'] - 1].get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
                pix.save(output / image)
                answer = json.dumps({'points': points}, ensure_ascii=False, separators=(',', ':'))
                row = {
                    'id': f"{group}:msop:{source['sha256'][:16]}:page:{page['page']}",
                    'project_group': group, 'page': page['page'],
                    'task': 'msop_source_coordinate_transcription',
                    'question': ('Read the printed coordinate tables in this MSOP page. Return only JSON '
                                 '{"points":[{"name":"printed name","axes":["X","Y"],'
                                 '"values_mm":[0,0]}]}. Preserve printed XYZ when available; '
                                 'for XY-only tables keep two axes and two values. Never invent Z, '
                                 'rename points, transform coordinates, or infer CAD positions.'),
                    'context': page['text'], 'images': [str(image)], 'answer': answer,
                    'source_file': source['filename'], 'source_sha256': source['sha256'],
                    'image_sha256': sha((output / image).read_bytes()),
                    'label_scope': 'printed_coordinates_only_not_CAD_registration',
                    'geometry_ground_truth': False, 'xml_used_as_input': False,
                }
                row['example_sha256'] = sha(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
                splits['validation' if group == holdout else 'train'].append(row)
    train_groups = sorted({r['project_group'] for r in splits['train']})
    test_groups = sorted({r['project_group'] for r in splits['validation']})
    if not splits['train'] or not splits['validation'] or set(train_groups) & set(test_groups):
        raise ValueError('Require nonempty drawing-disjoint training/validation groups')
    train_sources = {r['source_sha256'] for r in splits['train']}
    test_sources = {r['source_sha256'] for r in splits['validation']}
    if train_sources & test_sources:
        raise ValueError('Same MSOP bytes appear under different train/validation project labels')
    hashes = {}
    for split, rows in splits.items():
        data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
        (output / f'{split}.jsonl').write_bytes(data)
        hashes[split] = sha(data)
    manifest = {
        'schema': 'oracle-cloud-qwen-source-training-v1',
        'training_scope': 'MSOP_image_and_text_coordinate_reading',
        'is_FAI_geometry_training': False, 'is_XML_generation_training': False,
        'input_policy': 'MSOP page image and native text only; existing XML files are excluded',
        'ultimate_workflow': ['2D_3D_MSOP', 'FAI_registration', 'station_setup_optics',
                              'virtual_shot_and_pixel_world_correspondence', 'measurement_program_XML'],
        'XML_role': 'final_output_from_virtual_shots_and_measurement_plan; never inference evidence',
        'train_groups': train_groups, 'validation_groups': test_groups,
        'split_unit': 'drawing_number_all_revisions', 'excluded_groups': sorted(excluded),
        'samples': {k: len(v) for k, v in splits.items()},
        'point_count': {k: sum(len(json.loads(r['answer'])['points']) for r in v) for k, v in splits.items()},
        'sha256': hashes, 'excluded_pages': skipped,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, default=ROOT / 'talker/state/viewer')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--holdout', default='613-16076')
    parser.add_argument('--exclude', action='append', default=['817-07781'])
    args = parser.parse_args()
    print(json.dumps(prepare(ViewerStore(args.store), args.output, args.holdout, set(args.exclude)),
                     ensure_ascii=False, indent=2))
