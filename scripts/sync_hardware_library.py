"""Refresh public manufacturer snapshots, preserving verified files on failure.
Parameter-only public tables are valid sources; missing/gated PDFs stay unavailable.
"""
from pathlib import Path
import json,hashlib,datetime,concurrent.futures
import requests
root=Path(__file__).resolve().parents[1]/'oracle_viewer/catalog'
data=json.loads((root/'hardware.json').read_text());assets=root/'assets';assets.mkdir(exist_ok=True)
previous=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else []
def fetch(pair):
 item,kind=pair;url=item['source_url' if kind=='html' else 'datasheet_url'];name=item['id']+'.'+kind
 try:
  session=requests.Session();session.trust_env=False
  r=session.get(url,timeout=30,headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status();body=r.content
  if kind=='pdf' and not body.startswith(b'%PDF'):raise ValueError('Not a PDF; original not replaced')
  if kind=='html' and not any(t in body[:10000].lower() for t in [b'<html',b'<div',b'<table']):raise ValueError('Not HTML or public parameter fragment')
  (assets/name).write_bytes(body)
  return dict(item_id=item['id'],kind=kind,file=name,url=url,sha256=hashlib.sha256(body).hexdigest(),bytes=len(body),status='downloaded',retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),tls_verified=True)
 except Exception as e:
  old=next((m for m in previous if m['item_id']==item['id'] and m['kind']==kind and m['status']=='downloaded'),None)
  if old and (assets/old['file']).exists() and hashlib.sha256((assets/old['file']).read_bytes()).hexdigest()==old['sha256']:return old|dict(last_refresh_error=str(e),last_refresh_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
  return dict(item_id=item['id'],kind=kind,url=url,status='failed',error=str(e))
jobs=[(i,k) for i in data['items'] for k,key in [('html','source_url'),('pdf','datasheet_url')] if i.get(key)]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,jobs))
keys={(r['item_id'],r['kind']) for r in rows};rows=[m for m in previous if (m['item_id'],m['kind']) not in keys]+rows
(root/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print(json.dumps(dict(downloaded=sum(r['status']=='downloaded' for r in rows),failed=sum(r['status']=='failed' for r in rows),retained_with_refresh_error=sum('last_refresh_error' in r for r in rows)),indent=2))
