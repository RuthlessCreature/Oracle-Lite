"""Fetch model-specific manufacturer attachments from already saved source pages.

No guessed download URLs, no certificate bypass, no specification inference
from model numbers. Existing verified files are retained.
"""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin,urlparse,unquote
import requests
from oracle_viewer.store import now


def main():
    root=Path('oracle_viewer/catalog');cat=json.loads((root/'hardware.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    known={a['url'] for a in manifest if a.get('status')=='downloaded'};attempts=[]
    for item in cat['items']:
        candidates={item['datasheet_url']} if item.get('datasheet_url') else set()
        for a in manifest:
            if a['item_id']!=item['id'] or a.get('kind')!='html' or a.get('status')!='downloaded':continue
            text=(root/'assets'/a['file']).read_text(errors='replace')
            for link in re.findall(r'(?:href|src)=[\"\']([^\"\']+)[\"\']',text,re.I):
                if item['id'].lower() in unquote(link).lower() and re.search(r'\.(?:pdf|step|stp|dxf|dwg|zip)(?:[?#]|$)',link,re.I):candidates.add(urljoin(a['url'],link))
        for url in sorted(candidates-known):
            path=urlparse(url);kind=Path(unquote(path.path)).suffix[1:].lower()
            if kind not in ('pdf','step','stp','dxf','dwg','zip'):continue
            row=dict(item_id=item['id'],url=url,kind=kind,retrieved_at=now())
            try:
                response=requests.get(url,timeout=(8,20));response.raise_for_status();blob=response.content
                if len(blob)>40*1024**2:raise ValueError('Attachment exceeds 40MiB')
                if kind=='pdf' and not blob.startswith(b'%PDF'):raise ValueError('Server did not return a PDF')
                if kind in ('step','stp') and b'ISO-10303-21' not in blob[:1000]:raise ValueError('Server did not return a STEP')
                filename=item['id']+'-'+hashlib.sha256(url.encode()).hexdigest()[:8]+'.'+kind
                (root/'assets'/filename).write_bytes(blob)
                row.update(file=filename,status='downloaded',bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),tls_verified=True)
                manifest.append(row);known.add(url)
            except Exception as e:row.update(status='unavailable',error=str(e)[:300])
            attempts.append(row);print(item['id'],row['status'],flush=True)
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    out=Path('reports/engineering-checks/20261012');out.mkdir(parents=True,exist_ok=True)
    (out/'hardware-attachment-retrieval.json').write_text(json.dumps(attempts,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
