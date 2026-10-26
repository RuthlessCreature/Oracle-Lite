"""Same-runtime adapter comparison; exact checks do not certify designs."""
from __future__ import annotations

import argparse
import json
import time
import re
from pathlib import Path


class NoRetrieval:
    def curated_context(self, *args, **kwargs):
        return None

    def context(self, *args, **kwargs):
        return ''


def equivalent_mapping(actual, expected) -> bool:
    """Accept equivalent tolerance notation, keeping all fields and values."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            equivalent_mapping(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            equivalent_mapping(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, str) and isinstance(actual, str):
        def normalize(value):
            match = re.fullmatch(r'\s*(?:±|\+/-)\s*([\d.]+)\s*(?:mm)?\s*', value)
            if match:
                return (float(match.group(1)), -float(match.group(1)))
            match = re.fullmatch(r'\s*([+][\d.]+)\s*/\s*(-[\d.]+)\s*(?:mm)?\s*', value)
            return tuple(float(part) for part in match.groups()) if match else value
        return normalize(actual) == normalize(expected)
    return type(actual) is type(expected) and actual == expected


def main():
    from talker.model import TalkerModel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--retention', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--generation-deadline-seconds', type=float, default=120)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()][:args.count]
    for row in rows:
        row['evaluation_split'] = 'heldout_fictional'
    if args.retention:
        retention = [json.loads(line) for line in args.retention.read_text().splitlines() if line.strip()]
        for row in retention[:3] + [next(row for row in retention if row.get('images'))]:
            row['evaluation_split'] = 'retention_seen_in_training'
            if 'expected_mapping' in row:
                row['expected'] = row['expected_mapping']
            elif row['answer'].startswith('{'):
                try:
                    row['expected'] = json.loads(row['answer'])
                except ValueError:
                    pass
            rows.append(row)
    rows.append({'evaluation_split': 'real_fai23_regression', 'question':
                 '针对MSOP的FAI23，出具初步光学视觉方案：明确名义值与测量坐标，说明相机镜头光源的几何关系，给出带数量的候选BOM、采样计算及验证计划。没有型号资料时允许按规格选型，标明设计假设。请控制在1200字以内。'})
    model = TalkerModel(args.output_dir)
    model.load()
    original_retriever = model.retriever
    report = {'model': model.info(), 'note': 'Exact mapping scores and saved generations are preliminary checks; engineering proposals need semantic review and sample validation.', 'results': []}
    if args.resume and args.report.exists():
        report = json.loads(args.report.read_text())
    original_generate_once = model._generate_once
    generation_started = 0
    timed_out = False

    def bounded_generate(*a, **kw):
        nonlocal timed_out
        result = original_generate_once(*a, **kw)
        if time.monotonic() - generation_started > args.generation_deadline_seconds:
            timed_out = True
            return result[0], False, result[2]
        return result

    model._generate_once = bounded_generate
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        if index < len(report['results']):
            continue
        real = row['evaluation_split'] == 'real_fai23_regression'
        model.retriever = original_retriever if real else NoRetrieval()
        attachment = [] if real else [{'filename': row['source_path'], 'parsed_text': row['context'], 'images': row.get('images', [])}]
        started = time.monotonic()
        generation_started = started
        timed_out = False
        answer = model.answer([{'role': 'user', 'content': row['question'], 'attachments': attachment}])
        expected = row.get('expected')
        parsed = None
        if expected is not None:
            try:
                parsed = json.loads(answer)
            except ValueError:
                pass
        entry = {'index': index, 'split': row['evaluation_split'], 'family': row.get('fixture_family'),
                 'question': row['question'], 'context': row.get('context'), 'images': row.get('images', []),
                 'answer': answer, 'seconds': round(time.monotonic()-started, 2),
                 'expected': expected, 'mapping_exact': not timed_out and parsed == expected if expected is not None else None,
                 'mapping_equivalent': not timed_out and equivalent_mapping(parsed, expected) if expected is not None else None,
                 'generation_deadline_exceeded': timed_out,
                 'engineering_semantic_pass': False if timed_out else None}
        report['results'].append(entry)
        report['exact_scored'] = sum(result['mapping_exact'] is not None for result in report['results'])
        report['exact_passed'] = sum(result['mapping_exact'] is True for result in report['results'])
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({'case': index, 'split': entry['split'], 'family': entry['family'],
                          'mapping_exact': entry['mapping_exact'], 'seconds': entry['seconds']}, ensure_ascii=False), flush=True)
    print(json.dumps({'report': str(args.report), 'exact_scored': report['exact_scored'],
                      'exact_passed': report['exact_passed']}), flush=True)


if __name__ == '__main__':
    main()
