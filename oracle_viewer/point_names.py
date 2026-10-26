"""Expand literal MSOP identifier-list shorthand; never resolve a CAD entity."""
import re
from .annotations import MAX_ANNOTATION_POINTS


def expand_point_clause(clause,*,limit=MAX_ANNOTATION_POINTS,plane_view_suffix=True):
    if plane_view_suffix:
        clause=re.split(r'\bas\s+(?:the\s+)?plane\s+view\s+shows?\b',clause,maxsplit=1,flags=re.I)[0]
    result=[];prefix='';tag=''
    for token in re.split(r',|&|\band\b',clause,flags=re.I):
        token=re.sub(r'\s*([_/~–—-])\s*',r'\1',token.strip()).upper().strip(';:')
        if not token:continue
        if not re.fullmatch(r'[A-Z0-9_/~–—-]+',token):return None
        slash=token.split('/');head=slash[0]
        endpoints=re.split(r'~|–|—|-',head)
        if len(endpoints)>2:return None
        start=endpoints[0]
        if '_' not in start and prefix:start=prefix+start
        match=re.fullmatch(r'(.*_)?([A-Z]*)(\d+)',start)
        current_prefix=(match.group(1) or '') if match else ''
        current_tag=match.group(2) if match else ''
        if len(endpoints)==2:
            if not match:return None
            stop=endpoints[1]
            if '_' not in stop:stop=current_prefix+(stop if re.match(r'[A-Z]',stop) else current_tag+stop)
            end=re.fullmatch(re.escape(current_prefix+current_tag)+r'(\d+)',stop)
            if not end:return None
            lo,hi=int(match.group(3)),int(end.group(1))
            if hi<lo or hi-lo>=limit:return None
            names=[current_prefix+current_tag+str(i).zfill(len(match.group(3))) for i in range(lo,hi+1)]
        else:names=[start]
        for suffix in slash[1:]:
            if '_' in suffix:names.append(suffix)
            elif match and re.fullmatch(r'[A-Z]*\d+',suffix):
                names.append(current_prefix+(suffix if re.match(r'[A-Z]',suffix) else current_tag+suffix))
            else:return None
        if any(not re.fullmatch(r'[A-Z][A-Z0-9_]*',name) for name in names):return None
        result.extend(names)
        if len(result)>limit:return None
        if match:prefix=current_prefix;tag=current_tag
    return result or None
