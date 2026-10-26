"""Read literal feature-coordinate tables, preserving missing axes and scope."""
import re

_HEADER = re.compile(r'\bFeatures?[ \t]+(X[ \t]+Y(?:[ \t]+Z)?)(?=[ \t]*(?:\n|$|Features?\b))', re.I | re.M)
_NAME = re.compile(r'[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+', re.I)
_NUMBER = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')
_COUNT = re.compile(r'\bMeasure\s+(\d+)\s+(?:(?:vector|surface)\s+)?points?\b', re.I)
_FEATURE_COUNT = re.compile(r'\bMeasure\s+(\d+)\s+(?:(?:top|bottom|upper|lower|vector|vision|surface)\s+)*(scans?|lines?|circles?|planes?)\b', re.I)
_RANGE_HEADER = re.compile(r'\bFeatures?[ \t]+Start[ \t]+point[ \t]+End[ \t]+point[ \t]+Direction[ \t]*(?=\n|$)', re.I | re.M)


def feature_coordinate_rows(page):
    text = page.get('text', '')
    headers = list(_HEADER.finditer(text))
    frames = {re.sub(r'\s*&\s*', '&', frame.upper()) for frame in re.findall(
        r'\bOutput\s+system\s*:?\s*(ALI_[A-Z0-9_]+(?:\s*&\s*ALI_[A-Z0-9_]+)*)\b', text, re.I)}
    output_frame = next(iter(frames)).upper() if len(frames) == 1 else None
    result = []
    for i, header in enumerate(headers):
        axes = header.group(1).upper().split()
        end = headers[i+1].start() if i+1 < len(headers) else len(text)
        for line in text[header.end():end].splitlines():
            cells = line.split()
            width = len(axes)+1
            if not cells or len(cells) % width:
                continue
            groups = [cells[j:j+width] for j in range(0, len(cells), width)]
            if not all(_NAME.fullmatch(g[0]) and all(_NUMBER.fullmatch(v) for v in g[1:]) for g in groups):
                continue
            for group in groups:
                result.append(dict(name=group[0].upper(), axes=axes,
                    values=group[1:], source_page=page.get('page'), original_row=line.strip(),
                    coordinate_system=output_frame,
                    coordinate_frame_source='explicit_output_system_table' if output_frame else None))
    return result


def table_point_manifest(page):
    """Propose names only when one generic Measure N matches the whole table.

    This is source extraction, not authority to assign any CAD position. XY
    tables stay XY, and a local model must confirm the proposed manifest.
    """
    matches = list(_COUNT.finditer(page.get('text', '')))
    if len(matches) != 1:
        return None
    match = matches[0]
    tail = page['text'][match.end():].splitlines()[0].strip()
    if tail and not tail.startswith('.') and not re.match(r'and\s+construct\b', tail, re.I):
        return None
    rows = feature_coordinate_rows(page)
    by_name = {}
    for row in rows:
        old = by_name.get(row['name'])
        if old and (old['axes'] != row['axes'] or old['values'] != row['values']):
            return None
        by_name[row['name']] = row
    count = int(match.group(1))
    if not count or len(by_name) != count:
        return None
    return dict(source_page=page.get('page'), declared_count=count,
                source_point_names=list(by_name), instruction=match.group(0),
                evidence_kind='explicit_feature_coordinate_table')


def feature_range_rows(page):
    """Keep line/scan limits as axis ranges, never manufacture an XYZ point."""
    text=page.get('text','');headers=list(_RANGE_HEADER.finditer(text));rows=[]
    record=re.compile(r'\b('+_NAME.pattern+r')\s+('+_NUMBER.pattern+r')\s+('
                      +_NUMBER.pattern+r')\s+([XYZ])-axis\b',re.I)
    for i,header in enumerate(headers):
        end=headers[i+1].start() if i+1<len(headers) else len(text)
        for match in record.finditer(text[header.end():end]):
            rows.append(dict(name=match.group(1).upper(),start=match.group(2),
                end=match.group(3),direction_axis=match.group(4).upper(),
                source_page=page.get('page'),original_row=match.group(0)))
    return rows


def table_measurement_manifest(page):
    points=table_point_manifest(page)
    if points:return points
    matches=list(_FEATURE_COUNT.finditer(page.get('text','')))
    if len(matches)!=1:return None
    rows=feature_range_rows(page);by_name={}
    for row in rows:
        old=by_name.get(row['name'])
        if old and any(old[key]!=row[key] for key in ('start','end','direction_axis')):return None
        by_name[row['name']]=row
    match=matches[0];count=int(match.group(1))
    if not count or len(by_name)!=count:return None
    return dict(source_page=page.get('page'),declared_count=count,
        source_point_names=list(by_name),instruction=match.group(0),
        feature_kind=match.group(2).lower().rstrip('s'),
        feature_ranges=list(by_name.values()),evidence_kind='explicit_feature_range_table')
