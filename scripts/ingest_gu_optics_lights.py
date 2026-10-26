#!/usr/bin/env python3
"""Download product-linked machine-vision light STEP models from GU Optics' public catalog."""
from __future__ import annotations
import concurrent.futures,hashlib,json,os,pathlib,re,requests,shutil,tempfile
from datetime import datetime,timezone
from threading import Lock
ROOT=pathlib.Path(__file__).resolve().parents[1];CAT=ROOT/'oracle_viewer/catalog';ASSETS=CAT/'assets'
INDEX=pathlib.Path('/tmp/gu-light-library.json');PROGRESS=pathlib.Path('/tmp/gu-optics-step-download-progress.json')
BASE='https://www.gu-optics.com';VENDOR='GU Optics Shanghai product listing (reseller; OEM identity unverified)'
LOCK=Lock()
def parse_price(s):
 try:return float(str(s).replace(',','').replace('￥','').strip())
 except:return None
def fetch(row):
 path=row.get('step_path')
 if not path:return {'model':row.get('model'),'status':'missing_step_path'}
 url=path if path.startswith('http') else BASE+path
 try:
  with requests.get(url,headers={'User-Agent':'Mozilla/5.0 Oracle-Lite AOI catalog ingestion','Referer':row.get('source_detail',BASE)},stream=True,timeout=(20,120)) as r:
   r.raise_for_status();size=int(r.headers.get('Content-Length') or 0)
   h=hashlib.sha256();tmp=ASSETS/f'.gu-tmp-{os.getpid()}-{hashlib.sha1(row["model"].encode()).hexdigest()[:10]}'
   total=0;prefix=b''
   with tmp.open('wb') as f:
    for chunk in r.iter_content(1024*1024):
     if not chunk:continue
     if len(prefix)<256:prefix=(prefix+chunk)[:256]
     h.update(chunk);f.write(chunk);total+=len(chunk)
   if b'ISO-10303-21' not in prefix:
    sample=prefix[:60];tmp.unlink(missing_ok=True);return {'model':row.get('model'),'status':'invalid_step_header','url':url,'sample':sample.decode('ascii','replace')}
   digest=h.hexdigest();name=f'GU-CAD-{digest[:20]}.step';dest=ASSETS/name
   if dest.exists():tmp.unlink(missing_ok=True)
   else:os.replace(tmp,dest)
   return {'model':row.get('model'),'status':'downloaded','file':name,'url':url,'sha256':digest,'bytes':total,'expected_bytes':size or None,'retrieved_at':datetime.now(timezone.utc).isoformat()}
 except Exception as e:return {'model':row.get('model'),'status':'error','url':url,'error':repr(e)}
def main():
 index=json.loads(INDEX.read_text());rows=index['rows'];ASSETS.mkdir(parents=True,exist_ok=True)
 old=json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {};todo=[r for r in rows if r.get('model') not in old or old[r['model']].get('status')!='downloaded']
 print(f'GU Optics products={len(rows)} cached={len(rows)-len(todo)} pending={len(todo)} workers=6',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
  futures={ex.submit(fetch,r):r for r in todo}
  for n,f in enumerate(concurrent.futures.as_completed(futures),1):
   result=f.result();old[result['model']]=result
   if n%20==0 or n==len(todo):
    tmp=PROGRESS.with_suffix('.tmp');tmp.write_text(json.dumps(old,ensure_ascii=False,indent=2));os.replace(tmp,PROGRESS)
    counts={k:sum(x.get('status')==k for x in old.values()) for k in sorted({x.get('status') for x in old.values()})}
    print(f'{n}/{len(todo)} latest={result["model"]}:{result["status"]} counts={counts}',flush=True)
 PROGRESS.write_text(json.dumps(old,ensure_ascii=False,indent=2))
 hw=json.loads((CAT/'hardware.json').read_text());manifest=json.loads((CAT/'manifest.json').read_text());items={i['id']:i for i in hw['items']};assets={(x.get('item_id'),x.get('file')):x for x in manifest if x.get('file')};added=0
 for row in rows:
  result=old.get(row.get('model'),{})
  if result.get('status')!='downloaded':continue
  sku=row['model'];item_id='GU-'+re.sub('[^A-Za-z0-9-]+','-',sku).strip('-').upper();kind='controller' if row['category_kind']=='controller' else 'light'
  specs={}
  for sp in row.get('specs') or []:
   if isinstance(sp,dict) and sp.get('title'):specs[str(sp['title'])]=sp.get('attr_value')
  price=parse_price(row.get('price'));delivery='；'.join(str(v) for v in (row.get('availability'),row.get('warehouse')) if v)
  item={
   'id':item_id,'kind':kind,'vendor':VENDOR,'source_url':row['source_detail'],'datasheet_url':BASE+row['pdf_path'] if row.get('pdf_path') else None,
   'spec':specs,'price':price,'price_currency':'CNY','price_source':'GU Optics public product detail page; recheck before purchase',
   'availability':delivery or '网页未注明，请向供方核实','lifecycle_status':'listed_status_unverified',
   'cad_status':'supplier_product_linked_step_downloaded_geometry_unvalidated',
   'review_status':'SKU-specific STEP linked directly from public product detail; OEM identity and revision verification pending',
   'unknown_fields':['OEM/manufacturer identity','current stock and lead time','installation dimensions and revision verification'],
   'field_source':{k:row['source_detail'] for k in specs},'source_model':sku,'source_title':row.get('name'),
   'source_category':row['category_kind'],'source_supplier_code':row.get('supplier_model'),'source_model_id':row.get('source_id'),
   'source_vendor_status':'GU Optics Shanghai public catalog listing/reseller; OEM/manufacturer not verified',
   'cad_files':[result['file']],
   'cad_scope':'STEP downloaded from the exact SKU detail page. This is a source product-linked model; FreeCAD B-Rep and measured envelope checks are pending.',
   'source_pages':[row['source_detail']]
  }
  if item_id not in items:items[item_id]=item;added+=1
  else:items[item_id].update({k:v for k,v in item.items() if k not in ('spec',)})
  asset={'item_id':item_id,'kind':'step','file':result['file'],'url':result['url'],'sha256':result['sha256'],'bytes':result['bytes'],'status':'downloaded','retrieved_at':result['retrieved_at'],'manufacturer_file_name':pathlib.Path(row['step_path']).name,'cad_scope':'product-linked official GU Optics catalog STEP; OEM geometry origin unverified','source_page':row['source_detail'],'source_page_title':row.get('name'),'source_model_label':sku,'source_category':row['category_kind'],'source_supplier_code':row.get('supplier_model')}
  assets[(item_id,result['file'])]=asset
 hw['items']=sorted(items.values(),key=lambda x:(x['kind'],x.get('vendor',''),x['id']));hw['revision']='2026-10-14.21'
 (CAT/'hardware.json').write_text(json.dumps(hw,ensure_ascii=False,indent=2)+'\n');(CAT/'manifest.json').write_text(json.dumps(sorted(assets.values(),key=lambda x:(x.get('item_id',''),x['file'],x.get('kind',''))),ensure_ascii=False,indent=2)+'\n')
 counts={k:sum(v.get('status')==k for v in old.values()) for k in sorted({v.get('status') for v in old.values()})}
 print(json.dumps({'revision':hw['revision'],'download_status':counts,'catalog_records_added':added,'catalog_total':len(hw['items']),'GU_listings':len(rows),'GU_successful_model_SKUs':sum(old.get(r.get('model'),{}).get('status')=='downloaded' for r in rows),'unique_GU_STEP_assets':len({old[r['model']]['file'] for r in rows if old.get(r.get('model'),{}).get('status')=='downloaded'}),'total_GU_STEP_bytes':sum(old[r['model']]['bytes'] for r in rows if old.get(r.get('model'),{}).get('status')=='downloaded')},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
