"""Fictional source images with deterministic, independently known SFT labels."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_cases(directory: Path, *, seed: int = 314159, count: int = 64) -> list[dict]:
    from PIL import Image, ImageDraw, ImageFont
    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    small = ImageFont.truetype('DejaVuSans.ttf', 14)
    cases = []
    for index in range(count):
        image = Image.new('RGB', (720, 420), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((20, 12), 'FICTIONAL DOCUMENT — NOT PRODUCTION EVIDENCE', font=small, fill='black')
        family = index % 4
        expected = {}
        if family == 0:
            draw.text((20, 45), 'ALL FRAME DIMENSIONS — millimeters', font=font, fill='black')
            labels = ['INNER LENGTH', 'OUTER LENGTH', 'INNER LENGTH', 'OUTER WIDTH']
            rng.shuffle(labels)
            for row, label in enumerate(labels):
                value = f'{rng.randrange(20, 180)}.{rng.randrange(100):02d}'
                tolerance = f'0.{rng.randrange(1, 10):02d}'
                y = 95 + row * 65
                feature = rng.randrange(1, 90)
                draw.text((20, y), f'Feature {feature}', font=small, fill='black')
                draw.text((155, y), f'{value} ±{tolerance}', font=font, fill='black')
                draw.text((155, y+25), label, font=font, fill='black')
                draw.text((540, y+10), f'Row {row+1}', font=small, fill='black')
                expected[f'Row {row+1}'] = {'label': label, 'value': value, 'tolerance': '±'+tolerance}
            question = 'List ALL inner and outer frame dimensions shown, including repeated labels. Use a JSON object keyed by Row number, with label, value and tolerance as exact strings for each row. Feature IDs are not measurements. Output only JSON.'
        elif family == 1:
            draw.text((20, 45), 'Probe catalog', font=font, fill='black')
            for x, title in [(20, 'Probe File'), (265, 'Holder'), (485, 'Components')]:
                draw.text((x, 85), title, font=font, fill='black')
            probes = []
            for row in range(4):
                probe = f'PX_{rng.randrange(1000, 9999)}_{rng.choice("ABCD")}'
                probes.append(probe)
                for x, text in [(20, probe), (265, f'Holder-{rng.randrange(10,99)}'),
                                (485, f'Adapter-{rng.randrange(10,99)}')]:
                    draw.text((x, 130+row*55), text, font=font, fill='black')
            expected = {'Probe File': probes}
            question = 'List every Probe File in this catalog. Return only JSON with key "Probe File" and the ordered list of exact strings. Other columns are not probe files.'
        elif family == 2:
            draw.text((20, 45), 'Independent source sections', font=font, fill='black')
            common = f'PX_{rng.randrange(1000,9999)}'
            for row, scope in enumerate(['Section A', 'Section B']):
                program = f'MODE_{rng.randrange(100,999)}'
                draw.rectangle((15, 95+row*130, 690, 210+row*130), outline='black')
                draw.text((25, 102+row*130), scope, font=font, fill='black')
                draw.text((30, 140+row*130), f'Program     {program}', font=font, fill='black')
                draw.text((360, 140+row*130), f'Probe     {common}', font=font, fill='black')
                expected[scope] = {'Program': program, 'Probe': common}
            question = 'What are Program and Probe in every source section? Values in different sections are not missing. Return only JSON keyed by section with both original fields and string values.'
        else:
            draw.text((20, 45), 'Default limits', font=font, fill='black')
            for row, label in enumerate(['X.X', 'X.XX', 'X.XXX', 'ANGLES']):
                value = f'±0.{rng.randrange(1,99):02d}' + ('°' if label == 'ANGLES' else ' mm')
                draw.text((25, 110+row*50), label, font=font, fill='black')
                draw.text((230, 110+row*50), value, font=font, fill='black')
                expected[label] = value
            draw.text((25, 345), f'Note {rng.randrange(1,99)}: see standard DOC-{rng.randrange(100,999)}', font=small, fill='black')
            question = 'Return only JSON mapping all four default limit labels to their exact visible string values. Preserve signs and units. Do not substitute nearby note or standard numbers.'
        target = directory / f'fictional-{seed}-{index:04d}.png'
        image.save(target)
        answer = json.dumps(expected, ensure_ascii=False)
        cases.append({'mode': 'instruction', 'question': question,
                      'context': 'Read the attached fictional source image. Its visible fields are the evidence; no additional values are supplied in text.',
                      'answer': answer, 'text': answer, 'source_path': target.name,
                      'images': [str(target.resolve())], 'expected_mapping': expected,
                      'fixture_family': family, 'fictional': True})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=314159)
    parser.add_argument('--count', type=int, default=64)
    args = parser.parse_args()
    cases = build_cases(args.output.parent / (args.output.stem+'-images'), seed=args.seed, count=args.count)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in cases))
    print(json.dumps({'records':len(cases),'seed':args.seed,'output':str(args.output),'fictional':True}))


if __name__ == '__main__':
    main()
