"""Conservative, source-preserving MSOP index. No inferred geometric anchors."""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
import pymupdf

TITLE = re.compile(r'MSOP\s*:\s*([^\n]+)', re.I)
FAI = re.compile(r'\bFAI\s*(\d+)(?:\s*(?:to\s*(?:FAI\s*)?|[-–~]\s*FAI\s*)(\d+)|[-–~](\d+)(?![\d.]))?(?![A-Za-z0-9_])', re.I)
FEATURE = re.compile(r'\b(?:CIR|LIN|PLN|PNT|DATUM[A-D])_[A-Z0-9_]+\b')
DIMENSION_PATTERNS = [
    (re.compile(r'\b(?:2D\s+)?distance\s+(?:from\s+)?point\s+to\s+(?:point|line)\s+'
                r'parallel\s+to\s+([XYZ])\s+axis\b', re.I),
     lambda m: ('Δ' + m.group(1).upper(), 'DISTANCE')),
    (re.compile(r'\b(flatness|straightness|circularity|roundness|cylindricity|'
                r'profile\s+of\s+(?:a\s+)?(?:line|surface)|line\s+profile|surface\s+profile|parallelism|perpendicularity|'
                r'angularity|position|concentricity|symmetry|circular\s+runout|total\s+runout)\b', re.I), None),
    (re.compile(r'\b([XYZ])\s+value\b', re.I), lambda m: (m.group(1).upper(), 'COORDINATE')),
    (re.compile(r'\bdiameter\b', re.I), lambda m: ('⌀', 'DIAMETER')),
]
GD_T_SYMBOLS = {
    'flatness': ('⏥', 'FLATNESS'), 'straightness': ('⏤', 'STRAIGHTNESS'),
    'circularity': ('○', 'CIRCULARITY'), 'roundness': ('○', 'CIRCULARITY'),
    'cylindricity': ('⌭', 'CYLINDRICITY'), 'profile of line': ('⌒', 'PROFILE'),
    'profile of a line': ('⌒', 'PROFILE'), 'profile of surface': ('⌓', 'PROFILE'),
    'profile of a surface': ('⌓', 'PROFILE'), 'line profile': ('⌒', 'PROFILE'), 'surface profile': ('⌓', 'PROFILE'), 'parallelism': ('∥', 'PARALLELISM'),
    'perpendicularity': ('⊥', 'PERPENDICULARITY'), 'angularity': ('∠', 'ANGULARITY'),
    'position': ('⌖', 'POSITION'), 'concentricity': ('◎', 'CONCENTRICITY'),
    'symmetry': ('≡', 'SYMMETRY'), 'circular runout': ('↗', 'CIRCULAR RUNOUT'),
    'total runout': ('⌰', 'TOTAL RUNOUT'),
}


def measurement_indicators(text: str) -> list[dict]:
    """Extract explicit MSOP dimension/geometry symbols without inferring controls."""
    found = {}
    for pattern, mapper in DIMENSION_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0).strip().rstrip('.:;')
            if mapper:
                symbol, label = mapper(match)
            else:
                key = re.sub(r'\s+', ' ', raw).lower()
                symbol, label = GD_T_SYMBOLS.get(key, ('', ''))
            if symbol:
                found[(symbol, label)] = {'symbol': symbol, 'label': label, 'source': raw}
    return list(found.values())


def field(text: str, label: str) -> str | None:
    suffix = r'(?:[ \t]+TIP\d+)?' if label == 'Probe' else ''
    if label in ('Measured System', 'Dimension System'):
        # Coordinate systems can be composite in MSOP (for example,
        # ``ALI_ABC_A& ALI_ABD``). Keep the complete expression; truncating at
        # whitespace silently assigns every following point to the first frame.
        frame = r'ALI_[A-Z0-9_]+(?:\s*&\s*ALI_[A-Z0-9_]+)*'
        m = re.search(re.escape(label) + r'[ \t]+(' + frame + r')\b', text, re.I)
        return re.sub(r'\s*&\s*', '&', m.group(1).strip()) if m else None
    m = re.search(re.escape(label) + r'[ \t]+([A-Z0-9_&.-]+' + suffix + ')', text)
    return m.group(1) if m else None


def parse_page(text: str, number: int) -> dict:
    m = TITLE.search(text)
    heading = m.group(1).strip() if m else ''
    ids = []
    for match in FAI.finditer(heading):
        lo, hi = int(match.group(1)), int(match.group(2) or match.group(3) or match.group(1))
        if 0 < lo <= hi <= 10000 and hi - lo <= 500:
            ids.extend(range(lo, hi + 1))
    note_ids = set()
    for match in FAI.finditer(text.replace(heading, '', 1)):
        lo, hi = int(match.group(1)), int(match.group(2) or match.group(3) or match.group(1))
        if 0 < lo <= hi <= 10000 and hi - lo <= 500:
            note_ids.update(range(lo, hi + 1))
    specification = re.search(r'(?:\(SPC\s+([^)]*)\))?\s+([-+]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)/([+-]?\d+(?:\.\d+)?)', heading)
    ref = re.search(r'^(ALI_[A-Z0-9_]+|DATUM\s+[A-D])\b', heading)
    settings = [line.strip() for line in text.splitlines() if re.search(r'Settings:|Density:|Filter:|Outlier:', line, re.I)]
    return {
        'page': number, 'heading': heading, 'text': text,
        'fai_ids': sorted(set(ids)), 'note_fai_ids': sorted(note_ids),
        'reference': ref.group(1) if ref else None,
        'measured_system': field(text, 'Measured System'),
        'dimension_system': field(text, 'Dimension System'),
        'program': field(text, 'Program'), 'probe': field(text, 'Probe'),
        'features': sorted(set(FEATURE.findall(text))), 'settings': settings,
        'measurement_indicators': measurement_indicators(text),
        'specification': ({'spc': specification.group(1), 'nominal': specification.group(2),
                           'upper': specification.group(3), 'lower': specification.group(4),
                           'unit': None} if specification else None),
        'warnings': (['标题 FAI 范围与正文 FAI 范围不一致，请核对原页。']
                     if ids and note_ids and set(ids) != note_ids else []),
    }


def read_msop(path: Path) -> list[dict]:
    with pymupdf.open(path) as pdf:
        return [parse_page(page.get_text(sort=True), i + 1) for i, page in enumerate(pdf)]


def inventory(pages: list[dict], supplemental: dict | None = None) -> list[dict]:
    groups = defaultdict(list)
    for p in pages:
        for n in p['fai_ids']:
            groups[n].append(p)
    items = []
    for n, group in sorted(groups.items()):
        # Preserve duplicate pages/methods: a FAI is not necessarily one scalar.
        warning = sorted({w for p in group for w in p['warnings']})
        note_pages = [p['page'] for p in pages if n in p['note_fai_ids'] and p['warnings']]
        if note_pages:
            warning.append('其他页正文也提到此编号，存在标题/正文关联冲突：' + ', '.join(map(str, note_pages)))
        item = {'number': n, 'pages': [p['page'] for p in group],
                      'headings': [p['heading'] for p in group],
                      'features': sorted({f for p in group for f in p['features']}),
                      'probes': sorted({p['probe'] for p in group if p['probe']}),
                      'programs': sorted({p['program'] for p in group if p['program']}),
                      'specifications': [p['specification'] for p in group if p['specification']],
                      'warnings': warning, 'binding': None}
        extra = (supplemental or {}).get(str(n), {})
        item['measurement_metrics'] = list(extra.get('measurement_metrics', []))
        item['measurement_indicators'] = list({(i['symbol'], i['label']): i for i in [
            *(indicator for p in group for indicator in measurement_indicators(p.get('text', ''))),
            *extra.get('measurement_indicators', []),
        ]}.values())
        item['measurement_features'] = list(extra.get('measurement_features', []))
        item['visual_evidence'] = extra.get('visual_evidence')
        item['warnings'] = sorted(set(item['warnings']) | set(extra.get('warnings', [])))
        if item['measurement_features']:
            item['features'] = sorted(set(item['features']) | set(item['measurement_features']))
        items.append(item)
    return items


def dependencies(pages: list[dict], selected: list[dict]) -> list[dict]:
    """Follow explicit alignment/datum references; preserve program scope."""
    programs = {p['program'] for p in selected if p['program']}
    result = {p['page']: p for p in selected}
    def refs(text):
        return set(re.findall(r'\bALI_[A-Z0-9_]+\b', text)) | {
            'DATUM ' + d for d in re.findall(r'\bDATUM([ABCD])_[A-Z0-9_]+\b', text)}
    wanted = set().union(*(refs(p['text']) for p in selected))
    for _ in range(len(pages)):
        added = False
        for p in pages:
            if p['page'] in result or not p['reference'] or p['reference'] not in wanted:
                continue
            if programs and p['program'] and p['program'] not in programs:
                continue
            result[p['page']] = p
            wanted |= refs(p['text'])
            added = True
        if not added:
            break
    return sorted(result.values(), key=lambda p: p['page'])


def context(pages: list[dict], selected: list[dict], filename: str, max_chars: int = 24000) -> str:
    related = dependencies(pages, selected)
    priority = selected + [p for p in related if p['page'] not in {s['page'] for s in selected}]
    blocks, size = [], 0
    for p in priority:
        # Methods plus unit evidence; large location tables remain available in the UI.
        text = re.split(r'Location Definitions', p['text'], flags=re.I)[0].strip()
        if re.search(r'X,\s*Y,\s*Z\s*\(mm\)', p['text'], re.I):
            text += '\nCoordinate unit label: X,Y,Z(mm)'
        block = f'[Source {len(blocks)+1}: {filename}, page {p["page"]}]\n{text}'
        if size + len(block) > max_chars:
            blocks.append('[Context incomplete: additional dependency pages omitted; inspect original MSOP.]')
            break
        blocks.append(block)
        size += len(block) + 2
    return '\n\n'.join(blocks)
