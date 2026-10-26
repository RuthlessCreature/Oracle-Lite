#!/usr/bin/env python3
"""Download every CAD drawing exposed by the public HIKROBOT distributor catalog.

Original ZIP packages and direct CAD files are retained under source_archives;
CAD members are copied byte-for-byte to assets and tied to the model label shown
on the source page. This does not claim topology validation or exact commercial
variant equivalence beyond the source label.
"""
from __future__ import annotations
import datetime as dt, hashlib, json, os, re, unicodedata, urllib.parse, zipfile
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
import requests
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1]; CAT=ROOT/'oracle_viewer/catalog'; ASSETS=CAT/'assets'; ARCHIVES=CAT/'source_archives/hikrobot/mvisionpro/all-cad'
DOCS='https://mvisionpro.com/hikrobot/documents/'
CAD_EXTS={'.step','.stp','.iges','.igs','.dxf'}
KIND={'Area scan cameras':'camera','3D cameras & profilers':'surface_sensor','Lenses':'lens','Vision controllers':'controller','Smart cameras & vision sensors':'camera','Code readers':'camera','Line scan cameras':'camera','Frame grabbers':'controller'}
def sha(b): return hashlib.sha256(b).hexdigest()
def safe(s): return re.sub(r'[^A-Za-z0-9._-]+','-',unicodedata.normalize('NFKD',s)).strip('-._') or 'unnamed'
def write_json(p,v):
 with NamedTemporaryFile('w',encoding='utf8',dir=p.parent,delete=False) as f: json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');q=Path(f.name)
 os.replace(q,p)
def model_label(raw, fallback):
 s=raw.strip().replace('_','-')
 # Keep SKU and explicit version suffix; discard any trailing translated description.
 m=re.match(r'^((?:MVL?|MV|SA|ID|CL|GE)[-_A-Za-z0-9./]+?(?:\s+V\d+(?:\.\d+)*)?)$',s,re.I)
 return m.group(1).replace('/','-') if m else fallback.strip()
def main():
 ASSETS.mkdir(parents=True,exist_ok=True);ARCHIVES.mkdir(parents=True,exist_ok=True)
 sess=requests.Session();sess.headers['User-Agent']='Mozilla/5.0 Oracle-AOI-CAD-catalog/1.0'
 resp=sess.get(DOCS,timeout=60);resp.raise_for_status(); soup=BeautifulSoup(resp.text,'html.parser')
 hp,mp=CAT/'hardware.json',CAT/'manifest.json';hw=json.loads(hp.read_text());manifest=json.loads(mp.read_text());items={i['id']:i for i in hw['items']};linked={(x.get('item_id'),x.get('file')) for x in manifest}
 now=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds'); summary=[];errors=[];added=0;cad_links=0
 for sec in soup.select('section.cat-section'):
  title=sec.select_one('h2').get_text(' ',strip=True); kind=KIND.get(title,'camera')
  for art in sec.select('article'):
   for link in art.select('a[data-document-type="drawing"][data-document-action="download"]'):
    name=link.get('download','').strip();url=urllib.parse.urljoin(DOCS,link.get('href',''));model=link.get('data-document-model','').strip()
    if not name or not model or not url: continue
    ext=Path(name).suffix.lower()
    if ext not in CAD_EXTS and ext!='.zip': continue
    cad_links+=1; source_page=urllib.parse.urljoin(DOCS,(art.select_one('h2 a[href],h3 a[href],h4 a[href]') or {}).get('href',''))
    try:
     local=ARCHIVES/(safe(Path(name).stem)+ext)
     # Reuse the earlier profiler batch when the original package filename is
     # already present. Do not fetch a second byte-identical ZIP just because
     # the all-category ingest uses its own archive subdirectory.
     prior=ARCHIVES.parent/(safe(Path(name).stem)+ext)
     if local.exists(): blob=local.read_bytes()
     elif prior.exists(): local=prior;blob=local.read_bytes()
     else:
      r=sess.get(url,timeout=180);r.raise_for_status();blob=r.content
      if ext=='.zip' and not blob.startswith(b'PK'): raise ValueError('download is not ZIP')
      local.write_bytes(blob)
     archive_hash=sha(blob);archive_rel=str(local.relative_to(CAT)); entries=[]
     if ext=='.zip':
      with zipfile.ZipFile(local) as z:
       for m in z.infolist():
        p=PurePosixPath(m.filename)
        if m.is_dir() or p.suffix.lower() not in CAD_EXTS: continue
        if p.is_absolute() or '..' in p.parts: errors.append({'model':model,'member':m.filename,'error':'unsafe member path'});continue
        label=model_label(p.stem,model)
        # Reject another obvious SKU from an unrelated bundle; keep ambiguous labels under page SKU.
        if re.match(r'^(?:MV|MVL|SA|ID|CL|GE)[-_]',p.stem,re.I) and label!=model and not label.startswith(model+' '):
         # Versioned drawings of the same base SKU remain separate variants.
         if not (model.startswith('MV-') and label.startswith(model+' ')): label=model
        entries.append((label,p.name,z.read(m),m.filename))
     else: entries=[(model,Path(name).name,blob,Path(name).name)]
     saved=0
     for label,member_name,data,member_path in entries:
      item_id=label
      fn=f'HIK-{safe(label)}-{sha(data)[:10]}{Path(member_name).suffix.lower()}';dst=ASSETS/fn
      if not dst.exists(): dst.write_bytes(data)
      it=items.get(item_id)
      if it is None:
       it={'id':item_id,'kind':kind,'vendor':'HIKROBOT','source_url':source_page,'datasheet_url':None,'spec':{},'price':None,'availability':'Supplier CAD documentation found; Mainland stock and lead time require verification','cad_status':'exact_source_model_CAD_downloaded_geometry_pending','cad_files':[],'cad_model_ref':label,'catalog_section':title,'cad_scope':'CAD member downloaded from public HIKROBOT distributor drawing package. Source label is retained; B-Rep, dimensions and order-code equivalence require audit.','review_status':'Source CAD downloaded; geometry and variant correspondence pending audit','unknown_fields':['published envelope','mounting datum','cable clearance','price','availability and revision status'],'layout_adapter_ready':False,'cad_brep_validation':'pending','field_source':{'manufacturer_model':source_page}}
       items[item_id]=it
      if fn not in it.setdefault('cad_files',[]): it['cad_files'].append(fn)
      if (item_id,fn) not in linked:
       manifest.append({'item_id':item_id,'kind':Path(member_name).suffix.lower().lstrip('.'),'file':fn,'url':url,'source_page':source_page,'document_library_url':DOCS,'direct_package_url':url,'sha256':sha(data),'bytes':len(data),'status':'downloaded','retrieved_at':now,'tls_verified':True,'source_archive_file':archive_rel,'source_archive_sha256':archive_hash,'source_archive_bytes':len(blob),'source_archive_member':member_path,'manufacturer_file_name':member_name,'source_model_label':label,'catalog_section':title,'cad_scope':'Byte-preserved source CAD; model label copied from supplier page or filename. No geometry repair or variant substitution.'});linked.add((item_id,fn));added+=1
      saved+=1
     summary.append({'model':model,'section':title,'source':archive_rel,'cad_files':saved,'archive_bytes':len(blob)})
    except Exception as e: errors.append({'model':model,'url':url,'error':f'{type(e).__name__}: {e}'})
 if added:
  # Keep the current catalog date/revision sequence. Older versions of this
  # importer hard-coded 2026-10-14 and could roll back a newer catalog stamp.
  current=hw.get('revision','2026-10-21.8')
  try:
   date,seq=current.rsplit('.',1);n=int(seq)+1
  except (ValueError,TypeError):
   date=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat();n=1
  today=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
  if date < today: date=today;n=1
  hw['revision']=f'{date}.{n}';hw['items']=sorted(items.values(),key=lambda x:(x.get('kind',''),x.get('vendor',''),x['id']));manifest.sort(key=lambda x:(x.get('item_id',''),x['file'],x.get('kind','')));write_json(hp,hw);write_json(mp,manifest)
 print(json.dumps({'revision':hw.get('revision'),'drawing_packages_processed':len(summary),'drawing_links_found':cad_links,'new_cad_assets':added,'packages':summary,'errors':errors},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
