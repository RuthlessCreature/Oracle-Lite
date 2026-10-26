"""Deterministic fictional SFT tasks teaching evidence use, never corpus facts."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_fixtures(seed=271828, count=384):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        source = f'fictional-method-{i+1}.txt'
        kind = i % 10
        name = f'K{rng.randrange(10000,99999)}'
        value = f'{rng.randrange(1,99)}.{rng.randrange(1000):03d} mm'
        code = f'ZX-{rng.randrange(100000,999999)}'
        second = f'{rng.randrange(101,199)}.{rng.randrange(1000):03d} mm'
        if kind == 0:
            context = f'Field    Value\n{name}    {value}\nBatch    {code}'
            question = f'附件中 {name} 与 Batch 的值分别是什么？保留原始字段名和数值。'
            answer = f'{name}: {value}\nBatch: {code}'
        elif kind == 1:
            note = rng.randrange(10,99)
            context = f'Note {note}: Follow the storage procedure.\nRecord {rng.randrange(100,999)}\nWidth = {value}\nBatch = {code}'
            question = 'What is Width? Do not treat note numbers or record IDs as measurements.'
            answer = f'Width: {value}'
        elif kind == 2:
            context = f'Model = {code}\nLength = {value}\nNo calibration date is specified in this excerpt.'
            question = f'{code} 下次校准的具体年月日是什么？'
            answer = '所提供的附件片段没有给出下次校准日期，无法据此确定具体年月日。'
        elif kind == 3:
            context = f'Step 1: {name} = {value}\nStep 2: {name} = {second}'
            question = f'{name} 的值是什么？是否有唯一的答案？'
            answer = f'Step 1: {name} = {value}\nStep 2: {name} = {second}\n不同步骤的值不同；请指定步骤，不能把其中一个当成唯一答案。'
        elif kind == 4:
            first = rng.randrange(10, 70)
            last = first + rng.randrange(3, 9)
            note = rng.randrange(1, 10)
            other = last + rng.randrange(1, 10)
            context = f'Section: Feature definitions\nFeatures {first} through {last}: pull test.\nSection: Numbered notes\nNote {note}: Pull test sampling instructions.\nFeature {other}: torque test.'
            question = 'Which feature IDs are pull tests? Keep feature IDs distinct from note numbers and other test categories.'
            answer = f'Features {first} through {last} are pull tests. Note {note} is a note number, not a feature ID. Feature {other} is a torque test.'
        elif kind == 5:
            context = f'Instrument {code} cannot perform task {name}. No reason is specified.'
            question = f'为什么 {code} 不能执行 {name}？'
            answer = f'附件仅说明 {code} 不能执行 {name}，未说明具体原因，不能据此推断原因。'
        elif kind == 6:
            context = f'Document A: Limit = {value}\nDocument B: Limit = {value}\nDocument C: Limit = {second}\nNo source precedence is specified.'
            question = 'Can we choose a unique Limit because two documents agree? Report the evidence.'
            answer = f'Documents A and B state {value}; document C states {second}. No precedence is specified, so majority agreement does not establish a unique correct value.'
        elif kind == 7:
            context = f'Section: General standards\nFollow STANDARD-{code}.\nSection: Default values\n{name} = {value}\nSection: Notes\nNote 12 describes packaging.'
            question = f'附件的默认值 {name} 是多少，写在哪个部分？请简短回答，不扩展其他部分。'
            answer = f'{name}: {value}，见附件 Default values 部分。'
        elif kind == 8:
            a, chinese_a, b, chinese_b = rng.choice([
                ('Program', '程序', 'Probe', '探针'),
                ('Mode', '模式', 'Device', '设备'),
                ('Color', '颜色', 'Batch', '批次'),
                ('Policy', '政策', 'Owner', '负责人'),
            ])
            changed = f'VAL-{rng.randrange(100000,999999)}'
            context = f'[Excerpt A]\n{a}    {code}\n{b}    {name}\n[Excerpt B]\n{a}    {changed}\n{b}    {name}'
            question = f'{chinese_a}（{a}）和{chinese_b}（{b}）分别是什么？请保留不同片段的值，不要因为没有唯一值就说字段未给出。'
            answer = f'{a}: Excerpt A 为 {code}；Excerpt B 为 {changed}。\n{b}: 两个片段均为 {name}。\n{a} 已给出多个值；需要明确适用片段，不能称为缺失。'
        else:
            f1, f2 = rng.choice([('Length', 'Width'), ('Size', 'Mass'), ('Count', 'Quantity'), ('Code', 'Batch')])
            v1, v2, v3, v4 = value, second, f'{rng.randrange(200,299)}.125 mm', f'{rng.randrange(300,399)}.625 mm'
            if f1 == 'Count':
                v1, v2, v3, v4 = [f'{rng.randrange(1,9999):06d}' for _ in range(4)]
            elif f1 == 'Code':
                v1, v2, v3, v4 = [f'ZX-{rng.randrange(100000,999999)}' for _ in range(4)]
            elif f1 == 'Size':
                v2, v4 = v2.replace('mm','kg'), v4.replace('mm','kg')
            context = f'Record A:\n{f1}    {v1}\n{f2}    {v2}\nRecord B:\n{f1}    {v3}\n{f2}    {v4}'
            question = f'What are the {f1} and {f2} values for Record A and Record B? Include every relevant entry, rather than only the first matching field.'
            answer = f'Record A: {f1} = {v1}; {f2} = {v2}.\nRecord B: {f1} = {v3}; {f2} = {v4}.'
        if i % 3 == 0:
            if kind == 0:
                mapping = {name: value, 'Batch': code}
            elif kind == 1:
                mapping = {'Width': value}
            elif kind == 2:
                mapping = {'calibration_date': None}
            elif kind == 3:
                mapping = {'Step 1': {name: value}, 'Step 2': {name: second}}
            elif kind == 4:
                # Feature numbers are identifiers in this task, so retain them
                # as JSON strings rather than implying arithmetic quantities.
                mapping = {'features': [str(feature) for feature in range(first, last+1)]}
            elif kind == 5:
                mapping = {'reason': None}
            elif kind == 6:
                mapping = {'Document A': value, 'Document B': value, 'Document C': second, 'unique': False}
            elif kind == 7:
                mapping = {name: value, 'section': 'Default values'}
            elif kind == 8:
                mapping = {'Excerpt A': {a: code, b: name}, 'Excerpt B': {a: changed, b: name}}
            else:
                mapping = {'Record A': {f1: v1, f2: v2}, 'Record B': {f1: v3, f2: v4}}
            question += (' Return only a strict JSON object using these keys: '
                         + ', '.join(mapping) + '. Preserve nesting when excerpts, records or steps have separate fields. '
                         'Use null for missing facts. Include no Markdown, citations, or text outside JSON.')
            answer = json.dumps(mapping, ensure_ascii=False)
        else:
            answer += f'\n[Source 1: {source}]'
        rows.append({'mode':'instruction','text':answer,'answer':answer,'question':question,
                     'context':context,'source_path':source,'images':[],
                     'fixture_kind':kind,'fictional':True})
    rng.shuffle(rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=271828)
    p.add_argument('--count',type=int,default=384)
    args=p.parse_args()
    if args.count < 1: p.error('--count must be positive')
    rows=build_fixtures(args.seed,args.count)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps({'records':len(rows),'seed':args.seed,'fictional':True,'output':str(args.output)}))


if __name__ == '__main__': main()
