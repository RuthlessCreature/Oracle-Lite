"""Reproducible unseen, fictional document-reading checks for a real Talker model."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def build_cases(seed: int = 20260930, count: int = 16) -> list[dict]:
    if count < 1:
        raise ValueError('count must be positive')
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        fields = {}
        for column in range(8):
            key = f'K{rng.randrange(10000, 99999)}_{column}'
            value = (
                f'{rng.randrange(1, 999)}.{rng.randrange(1000):03d} kg' if column % 4 == 0 else
                f'{rng.randrange(1, 99)}.{rng.randrange(100):02d}e-3 A' if column % 4 == 1 else
                f'{rng.randrange(1, 9999):06d}' if column % 4 == 2 else
                f'ZX-{rng.randrange(100000, 999999)}'
            )
            fields[key] = value
        rows = list(fields.items())
        rng.shuffle(rows)
        layout = index % 4
        if layout == 0:
            context = '\n'.join(f'{key}    {value}' for key, value in rows)
        elif layout == 1:
            context = '\n'.join(f'{key} = {value}' for key, value in rows)
        elif layout == 2:
            context = json.dumps(dict(rows), ensure_ascii=False, indent=2)
        else:
            context = '| Field | Value |\n| --- | --- |\n' + '\n'.join(f'| {key} | {value} |' for key, value in rows)
        question = (
            '仅根据附件，用一个 JSON 对象返回全部原始字段名到字符串值的映射。值须逐字保留单位、前导零和科学计数法，不增删字段。只输出 JSON。'
            if index % 2 == 0 else
            'Return only a JSON object mapping every original field name to its exact string value from the attachment. Preserve units, leading zeros and scientific notation. Do not add or omit fields.'
        )
        cases.append({'id': f'unseen-{index+1}', 'layout': layout, 'question': question,
                      'context': context, 'expected_fields': fields})
    return cases


def parse_answer(answer: str, *, allow_trailing=False) -> dict | None:
    value = answer.strip()
    if value.startswith('```') and value.endswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        parsed = json.JSONDecoder().raw_decode(value)[0] if allow_trailing else json.loads(value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _mapping_leaves(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _mapping_leaves(child, prefix + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _mapping_leaves(child, prefix + (str(index),))
    else:
        yield prefix, value


def score_json_mapping(answer: str, expected: dict) -> dict:
    """Separate exact schema compliance from recoverable JSON/value accuracy."""
    strict = parse_answer(answer)
    extracted = parse_answer(answer, allow_trailing=True)
    expected_leaves = dict(_mapping_leaves(expected))
    actual_leaves = dict(_mapping_leaves(extracted)) if extracted is not None else {}
    matched = sum(actual_leaves.get(path, object()) == value
                  for path, value in expected_leaves.items())
    stripped = answer.strip()
    json_only = (
        strict == expected and stripped.startswith("{") and stripped.endswith("}")
    )
    return {
        "json_only": json_only,
        "mapping_exact": extracted == expected,
        "fields_matched": matched,
        "expected_fields": len(expected_leaves),
        "expected_values_present": sum(
            isinstance(value, str) and value in answer
            for value in expected_leaves.values()
        ),
        "parsed_with_trailing_text": strict is None and extracted is not None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--count', type=int, default=16)
    args = parser.parse_args()
    cases = build_cases(args.seed, args.count)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / 'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in cases))
    from talker.model import TalkerModel
    model = TalkerModel(args.output_dir)
    model.load()
    # These facts are fictional and never enter the training corpus. Isolate
    # attached-document reading from reviewed QA and production retrieval.
    model.retriever.context = lambda *a, **k: ''
    model.retriever.curated_context = lambda *a, **k: None
    model.retriever.visual_context = lambda *a, **k: []
    results = []
    with (args.report_dir / 'answers.jsonl').open('w') as stream:
        for case in cases:
            start = time.monotonic()
            answer = model.answer([{'role': 'user', 'content': case['question'], 'attachments': [{
                'filename': 'Fictional unseen document', 'parsed_text': case['context'], 'images': []}]}])
            parsed = parse_answer(answer)
            extracted = parse_answer(answer, allow_trailing=True)
            expected = case['expected_fields']
            matched = sum(extracted.get(key) == value for key, value in expected.items()) if extracted is not None else 0
            result = {**case, 'answer': answer, 'parsed': parsed, 'extracted_mapping': extracted,
                      'exact_match': parsed == expected,
                      'fields_matched': matched, 'seconds': round(time.monotonic()-start, 2)}
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False)+'\n'); stream.flush()
            print(json.dumps({'id': case['id'], 'exact_match': result['exact_match'], 'fields_matched': matched}, ensure_ascii=False), flush=True)
    report = {'seed': args.seed, 'cases': len(cases), 'cases_passed': sum(r['exact_match'] for r in results),
              'fields': sum(len(c['expected_fields']) for c in cases),
              'fields_matched': sum(r['fields_matched'] for r in results), 'model': model.info(),
              'note': 'Exact checks on unseen fictional field mappings; does not measure open-domain knowledge, visual layout, or all hallucination modes.'}
    (args.report_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
