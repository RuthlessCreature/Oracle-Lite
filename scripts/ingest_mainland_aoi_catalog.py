#!/usr/bin/env python3
"""Idempotently ingest Shuangyi mainland industrial-vision CAD archives into Oracle Viewer.
Archives and crawl metadata are local staging inputs; only validated CAD files enter the catalog.
"""
from __future__ import annotations
import ctypes as C, hashlib, json, os, pathlib, re, shutil, sys
from datetime import datetime, timezone

ROOT=pathlib.Path(__file__).resolve().parents[1]
CAT=ROOT/'oracle_viewer/catalog'; ASSETS=CAT/'assets'
INDEX=pathlib.Path('/tmp/shuangyi-index.json'); DOWNLOADS=pathlib.Path('/tmp/shuangyi-download-progress.json')
CAD_EXT={'.step','.stp','.x_t','.x_b','.igs','.iges','.sat'}
TYPE_RULES=[
 ('camera',r'工业相机|模拟工业相机|智能相机|面阵相机|线阵相机|相机|camera|CCD|CMOS'),
 ('lens',r'远心镜头|定焦镜头|变焦镜头|镜头|lens|增倍镜'),
 ('light',r'光源|背光|同轴光|条形光|环形光|面光|穹顶光|低角度光|点光|光纤光'),
 ('controller',r'光源控制器|视觉控制器|视觉工控机|工控机|采集卡|IO模块|数字控制器|控制器'),
 ('sensor',r'3D相机|3D传感器|激光位移|位移传感器|测量传感器|读码器|扫码器'),
 ('mechanical_accessory',r'对位调整平台|精密平台|丝印机对位平台|镜头支架|相机支架|调节架|支架|滑台|模组'),
]
LIB=C.CDLL('libarchive.so.13')
LIB.archive_read_new.restype=C.c_void_p
for n,a,r in [('archive_read_support_format_all',[C.c_void_p],C.c_int),('archive_read_support_filter_all',[C.c_void_p],C.c_int),('archive_read_open_filename',[C.c_void_p,C.c_char_p,C.c_size_t],C.c_int),('archive_read_next_header',[C.c_void_p,C.POINTER(C.c_void_p)],C.c_int),('archive_entry_pathname',[C.c_void_p],C.c_char_p),('archive_entry_filetype',[C.c_void_p],C.c_uint),('archive_read_data',[C.c_void_p,C.c_void_p,C.c_size_t],C.c_ssize_t),('archive_read_data_skip',[C.c_void_p],C.c_int),('archive_read_free',[C.c_void_p],C.c_int)]:
 f=getattr(LIB,n); f.argtypes=a; f.restype=r

def extract(src,out):
 a=LIB.archive_read_new(); LIB.archive_read_support_format_all(a); LIB.archive_read_support_filter_all(a)
 if LIB.archive_read_open_filename(a,os.fsencode(src),65536)!=0: LIB.archive_read_free(a); return []
 entry=C.c_void_p(); buf=C.create_string_buffer(1024*1024); found=[]; seq=0
 while LIB.archive_read_next_header(a,C.byref(entry))==0:
  raw=LIB.archive_entry_pathname(entry); name=raw.decode('utf8','replace') if raw else ''
  if pathlib.PurePosixPath(name).suffix.lower() not in CAD_EXT or LIB.archive_entry_filetype(entry)!=32768:
   LIB.archive_read_data_skip(a); continue
  base=pathlib.PurePosixPath(name).name
  if not base or '..' in base: LIB.archive_read_data_skip(a); continue
  dst=out/f'{seq:04d}_{base}'; seq+=1; size=0
  with dst.open('wb') as f:
   while True:
    n=LIB.archive_read_data(a,buf,len(buf))
    if n<=0: break
    f.write(buf.raw[:n]); size+=n
  if size>=128:
   with dst.open('rb') as f: head=f.read(128)
   ext=dst.suffix.lower()
   valid=(b'ISO-10303-21' in head or ext in ('.igs','.iges') and (b'S' in head or size>1024) or ext in ('.x_t','.x_b','.sat'))
   if valid: found.append((dst,name,size))
   else: dst.unlink(missing_ok=True)
 LIB.archive_read_free(a); return found

def clean_label(s):
 s=re.sub(r'(?i)\s*[_\- ]*3D\s*(图纸|图|模型|drawing|model)\s*$','',s)
 s=re.sub(r'\s+',' ',s).strip(' _-')
 return s or 'unnamed'
def classify(text):
 for kind,pattern in TYPE_RULES:
  if re.search(pattern,text,re.I): return kind
 return None
def make_id(model,kind): return 'SHY-'+kind.upper()+'-'+hashlib.sha1((kind+'|'+model.casefold()).encode()).hexdigest()[:12]

def main():
 pages=json.loads(INDEX.read_text()); dl=json.loads(DOWNLOADS.read_text()); urls={d['url']:(p,d) for p in pages for d in p.get('docs',[])}
 hw=json.loads((CAT/'hardware.json').read_text()); manifest=json.loads((CAT/'manifest.json').read_text())
 old_items={x['id']:x for x in hw['items']}; old_manifest={(x['item_id'],x['file']):x for x in manifest if x.get('file')}
 temp=pathlib.Path('/tmp/shuangyi-extract'); temp.mkdir(exist_ok=True); assets={}; rows={}; page_products=0; failed=[]; classified=0
 for url,(page,doc) in urls.items():
  meta=dl.get(url,{})
  if meta.get('status')!='downloaded': failed.append((url,meta.get('error','missing download'))); continue
  title=page.get('title',''); label=doc.get('label',''); model=clean_label(label)
  kind=classify(title+' '+label)
  if not kind: continue
  classified+=1; item_id=make_id(model,kind)
  discontinued='停产' in title
  row=rows.setdefault(item_id,dict(id=item_id,kind=kind,vendor='Shenzhen Shuangyi product listing',source_url=page.get('page'),datasheet_url=None,spec={},price=None,availability='已停产；不可按在售器件计数' if discontinued else '中国大陆产品页可查；现货/交期需向供应商核实',lifecycle_status='discontinued' if discontinued else 'listed_status_unverified',cad_status='source_cad_pending_validation',review_status='source_model_unverified',unknown_fields=['verified_maker','technical_specifications','price','current_stock','installation_envelope'],field_source={},source_title=title,source_model_label=label,source_model=model,source_vendor_status='listing/distributor; manufacturer identity not uniformly verified',cad_files=[],cad_scope='CAD linked from the product listing with this model label; geometry and dimensions require supplier drawing verification before clearance or fit approval.',source_pages=[]))
  if page.get('page') not in row['source_pages']: row['source_pages'].append(page.get('page'))
  extracted=[]
  try: extracted=extract(meta['path'],temp)
  except Exception as e: failed.append((url,'extract '+repr(e)))
  for file,original,size in extracted:
   digest=hashlib.sha256(file.read_bytes()).hexdigest(); ext=file.suffix.lower(); final=f'SHY-CAD-{digest[:20]}{ext}'; dest=ASSETS/final
   if not dest.exists(): shutil.copyfile(file,dest)
   else: file.unlink(missing_ok=True)
   asset=dict(item_id=item_id,kind=ext.lstrip('.'),file=final,url=url,sha256=digest,bytes=dest.stat().st_size,status='downloaded',retrieved_at=datetime.now(timezone.utc).isoformat(),manufacturer_file_name=original,cad_scope='source_link_model_label; dimensional/model-specific verification pending',source_page=page.get('page'),source_page_title=title,source_model_label=label)
   assets[(item_id,final)]=asset
   if final not in row['cad_files']: row['cad_files'].append(final)
  for file in temp.iterdir(): file.unlink(missing_ok=True)
 # Update deterministic source records; never overwrite existing curated IDs.
 for item_id,row in rows.items():
  if not row['cad_files']:
   row['cad_status']='source_archive_has_no_supported_3d_cad'
  else:
   row['cad_status']='source_3d_cad_downloaded_unverified'
  if item_id not in old_items: old_items[item_id]=row
  else:
   # Refresh only provenance and CAD availability for records generated by this importer.
   if old_items[item_id].get('vendor')=='Shenzhen Shuangyi product listing':
    old_items[item_id].update({k:row[k] for k in ('source_url','source_title','source_model_label','source_model','source_pages','cad_files','cad_status','cad_scope','availability','lifecycle_status')})
 for k,v in assets.items(): old_manifest[k]=v
 hw['items']=sorted(old_items.values(),key=lambda i:(i['kind'],i['vendor'],i['id']))
 hw['revision']='2026-10-14.10'
 (CAT/'hardware.json').write_text(json.dumps(hw,ensure_ascii=False,indent=2)+'\n')
 (CAT/'manifest.json').write_text(json.dumps(sorted(old_manifest.values(),key=lambda x:(x['item_id'],x['file'],x.get('kind',''))),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(source_pages=len(pages),source_cad_links=len(urls),classified_aoi_links=classified,unique_classified_models=len(rows),models_with_cad=sum(bool(r['cad_files']) for r in rows.values()),unique_new_assets=len({x['file'] for x in assets.values()}),new_manifest_rows=len(assets),catalog_items=len(hw['items']),extraction_or_download_errors=failed[:12],error_count=len(failed)),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
