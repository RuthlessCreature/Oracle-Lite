"""Audit completed PPT against immutable source plan, not an expected slide template."""
import argparse,json,zipfile,hashlib
from pathlib import Path
from xml.etree import ElementTree as ET
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}

def audit(folder):
    folder=Path(folder);src=json.loads((folder/'input.json').read_text());manifest=json.loads((folder/'deck-manifest.json').read_text());ppt=folder/'output/station-workflow.pptx';issues=[]
    with zipfile.ZipFile(ppt) as z:
        slides={int(p.rsplit('slide',1)[1][:-4]):ET.fromstring(z.read(p)) for p in z.namelist() if p.startswith('ppt/slides/slide') and p.endswith('.xml')}
        text={n:'\n'.join(e.text or '' for e in el.findall('.//a:t',NS)) for n,el in slides.items()}
        alltext='\n'.join(text.values());allnotes='\n'.join(e.text or '' for p in z.namelist() if p.startswith('ppt/notesSlides/notesSlide') and p.endswith('.xml') for e in ET.fromstring(z.read(p)).findall('.//a:t',NS))
        if len(slides)!=manifest['slideCount']:issues.append('slide count mismatch')
        expected={str(t['number']) for t in src['tasks']}
        if set(manifest['faiIndex'])!=expected:issues.append('FAI index incomplete')
        for task in src['tasks']:
            pages=manifest['faiIndex'].get(str(task['number']),[]);section='\n'.join(text.get(n,'') for n in pages)
            if f'FAI {task["number"]}' not in section:issues.append(f'FAI {task["number"]} absent from its own chapter')
            for spec in task.get('specifications',[]):
                for key in ('nominal','upper','lower'):
                    if str(spec[key]) not in section:issues.append(f'FAI {task["number"]} missing {key}')
            for method in task['methods']:
                # Native speaker notes retain the full source method JSON.
                escaped=json.dumps(method,ensure_ascii=False,separators=(',',':'))
                if escaped not in allnotes:issues.append(f'FAI {task["number"]} source method P{method["page"]} missing from notes')
        for st in src['record']['result']['stations']:
            for u in st['setups']:
                if u['id'] not in manifest['setupIndex']:issues.append(f'setup {u["id"]} omitted')
                if 'setup_'+u['id'] not in src['images']:issues.append(f'setup {u["id"]} missing actual render')
        for slide in manifest['tableSlides']:
            if not slides[slide].findall('.//a:tbl',NS):issues.append(f'noneditable table slide {slide}')
        if len(manifest['nodeIndex'])!=6:issues.append('missing workflow node')
        for keyword in ('安装','计量','待验证','唯一结果','MSOP','CAD'):
            if keyword not in alltext:issues.append('missing qualification '+keyword)
    receipt=dict(file=str(ppt.resolve()),sha256=hashlib.sha256(ppt.read_bytes()).hexdigest(),slides=len(slides),sourceFAI=len(src['tasks']),indexedFAI=len(manifest['faiIndex']),workflowNodes=len(manifest['nodeIndex']),setups=len(manifest['setupIndex']),nativeTableSlides=len(manifest['tableSlides']),sourcePoints=manifest['sourcePointCount'],cadAnchors=manifest['cadAnchorCount'],localModel=src['record']['model_identity'],explanationTruncated=src['model_attempt']['truncated'],issues=issues)
    (folder/'.build/content-audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');r=audit(p.parse_args().folder);print(json.dumps(r,ensure_ascii=False));raise SystemExit(bool(r['issues']))
