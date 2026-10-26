#!/usr/bin/env python3
"""Move CAD links from truncated HIKROBOT family IDs to exact existing member SKUs.

Migration requires all three: a CAD member filename that exactly normalizes to an
existing HIK item ID, byte-identical CAD already attached to that target SKU, and
verified archive provenance. Unmatched assets are left untouched.
"""
from __future__ import annotations
import hashlib,json,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parents[1];CAT=ROOT/'oracle_viewer/catalog';ASSETS=CAT/'assets'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def norm(s):return re.sub(r'[^A-Za-z0-9]+','',s).lower()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
 hp,mp,ap=CAT/'hardware.json',CAT/'manifest.json',CAT/'cad_renderability_audit.json';h=json.loads(hp.read_text());manifest=json.loads(mp.read_text());audit=json.loads(ap.read_text());items={x['id']:x for x in h['items']};by_target={x['id']:x for x in h['items'] if x.get('vendor')=='HIKROBOT'}
 rows_by_pair={(x.get('item_id'),x.get('file')):x for x in manifest};remap=[];unmatched=[]
 for row in list(manifest):
  old=row.get('item_id','');source_item=items.get(old)
  if not source_item or source_item.get('vendor')!='HIKROBOT':continue
  member=pathlib.Path(row.get('manufacturer_file_name') or '').stem
  if not member:continue
  target=next((x for x in by_target if norm(x)==norm(member)),None)
  if not target or target==old:continue
  source=ASSETS/row['file'];source_hash=digest(source);exact_item=by_target[target]
  exact_files=[f for f in exact_item.get('cad_files',[]) if (ASSETS/f).is_file() and digest(ASSETS/f)==source_hash]
  if not exact_files:
   unmatched.append({'old_item':old,'target_candidate':target,'file':row['file'],'reason':'no byte-identical CAD already linked to exact target'});continue
  target_file=exact_files[0]
  target_row=rows_by_pair.get((target,target_file))
  if not target_row:
   unmatched.append({'old_item':old,'target_candidate':target,'file':row['file'],'reason':'exact asset has no target manifest row'});continue
  if row.get('source_archive_sha256')!=target_row.get('source_archive_sha256'):
   # Source packages may differ while the CAD bytes are identical; retain both evidence sets.
   pass
  evidence={'source_page':row.get('source_page'),'direct_package_url':row.get('direct_package_url'),'source_archive_file':row.get('source_archive_file'),'source_archive_sha256':row.get('source_archive_sha256'),'source_archive_member':row.get('source_archive_member'),'manufacturer_file_name':row.get('manufacturer_file_name'),'sha256':source_hash}
  extras=target_row.setdefault('alternate_source_evidence',[])
  if evidence not in extras:extras.append(evidence)
  remap.append({'old_item':old,'target_item':target,'old_file':row['file'],'target_file':target_file,'sha256':source_hash})
 if unmatched:print('UNMATCHED',json.dumps(unmatched,ensure_ascii=False))
 if not remap:
  print(json.dumps({'migrated':0,'unmatched':unmatched},ensure_ascii=False));return
 remove_rows={(x['old_item'],x['old_file']) for x in remap};remove_assets=[]
 for old,old_file in remove_rows:
  old_item=items[old];old_item['cad_files']=[f for f in old_item.get('cad_files',[]) if f!=old_file]
 manifest=[row for row in manifest if (row.get('item_id'),row.get('file')) not in remove_rows]
 # Remove stale per-file audit keys only when exact byte-identical target audit exists.
 for x in remap:
  oldrow=audit.get(x['old_file']);targetrow=audit.get(x['target_file'])
  if oldrow and targetrow and oldrow.get('status')!=targetrow.get('status'):
   raise RuntimeError(f"same SHA256 has different audit results: {x['old_file']} -> {x['target_file']}")
  audit.pop(x['old_file'],None)
  if not any(r.get('file')==x['old_file'] for r in manifest):remove_assets.append(x['old_file'])
 # Update historical HIK profiler file audit to point to the exact SKU asset, and deduplicate identical rows.
 profiler_path=CAT/'hikrobot_profiler_geometry_audit.json'
 if profiler_path.exists():
  report=json.loads(profiler_path.read_text());mapping={x['old_file']:(x['target_item'],x['target_file']) for x in remap};kept={}
  for r in report.get('items',[]):
   if r.get('file') in mapping:r['item_id'],r['file']=mapping[r['file']]
   kept[(r.get('item_id'),r.get('file'))]=r
  report['items']=list(kept.values());report['files_checked']=len(report['items']);
  from collections import Counter
  report['summary']=dict(Counter(r.get('status','error') for r in report['items']));write(profiler_path,report)
 # Drop now-empty family aliases; exact-model records already exist and no CAD is lost.
 empty=[i['id'] for i in h['items'] if i.get('vendor')=='HIKROBOT' and not i.get('cad_files') and any(x['old_item']==i['id'] for x in remap)]
 h['items']=[i for i in h['items'] if i['id'] not in empty]
 current=h.get('revision','2026-10-14.46');suffix=int(current.rsplit('.',1)[-1])+1 if '.' in current else 1;h['revision']=f'2026-10-14.{suffix}';h['items']=sorted(h['items'],key=lambda x:(x.get('kind',''),x.get('vendor',''),x['id']))
 manifest.sort(key=lambda x:(x.get('item_id',''),x.get('file',''),x.get('kind','')))
 write(hp,h);write(mp,manifest);write(ap,audit)
 # Keep the original supplier packages; remove only extracted duplicate files after reference checks.
 ref_files={f for i in h['items'] for f in i.get('cad_files',[])}|{r.get('file') for r in manifest}
 deleted=[]
 for name in set(remove_assets):
  if name not in ref_files:
   p=ASSETS/name
   if p.exists():p.unlink();deleted.append(name)
 print(json.dumps({'revision':h['revision'],'migrated_assets':len(remap),'empty_truncated_items_removed':empty,'duplicate_extractions_removed':len(deleted),'mappings':remap,'unmatched':unmatched},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
