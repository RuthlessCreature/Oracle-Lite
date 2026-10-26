"""Read-only audit of MSOP point/program/alignment identities across projects."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oracle_viewer.candidate_mining import extract_feature_labels, extract_named_points
from oracle_viewer.msop_coordinates import _coordinate_rows
from oracle_viewer.store import ViewerStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', action='append', help='Optional project ID; defaults to all projects')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    store = ViewerStore(ROOT / 'talker/state/viewer')
    reports = []
    for summary in store.listing():
        if args.project and summary['id'] not in args.project:
            continue
        project = store.get(summary['id'])
        source = store.active_msop(project)
        if not source:
            continue
        pages = source['method_pages']
        named = extract_named_points(pages)
        rows = []
        for item in store.items(project):
            detail = store.detail(project['id'], item['number'])
            points = _coordinate_rows(detail['methods'], [],
                                      extract_feature_labels(detail['methods'], item.get('features', [])), pages, named)
            rows.append(dict(fai=item['number'], point_count=len(points),
                             conflicts=[p['name'] for p in points if p.get('coordinate_conflict')],
                             scopes=dict(Counter(str((p.get('program'), p.get('coordinate_system'))) for p in points)),
                             points=points))
        report = dict(project=project['id'], drawing=project['drawing_no'], msop_sha256=source['sha256'],
                      named_points=len(named), multi_scope_names=[n for n, r in named.items() if r.get('scope_ambiguous')],
                      conflicting_names=[n for n, r in named.items() if r.get('coordinate_conflict')], fai=rows)
        reports.append(report)
        print(project['drawing_no'], 'named:', len(named), 'multiple scopes:', len(report['multi_scope_names']),
              'conflicting:', len(report['conflicting_names']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
