#!/usr/bin/env python3
"""Create STEP derivatives for catalog products whose only usable CAD is IGES."""
from __future__ import annotations
import concurrent.futures,hashlib,json,os,pathlib,shutil,tempfile
from datetime import datetime,timezone
ROOT=pathlib.Path(__file__).resolve().parents[1];CAT=ROOT/'oracle_viewer/catalog';ASSETS=CAT/'assets'
METRICS=pathlib.Path('/tmp/shuangyi-cad-metrics.json');CHECKPOINT=pathlib.Path('/tmp/shuangyi-iges-step-progress.json')

def convert(src_name):
 from OCP.IGESControl import IGESControl_Reader
 from OCP.STEPControl import STEPControl_Writer,STEPControl_AsIs
 from OCP.Bnd import Bnd_Box
 from OCP.BRepBndLib import BRepBndLib
 from OCP.IFSelect import IFSelect_ReturnStatus
 src=ASSETS/src_name; digest=hashlib.sha256(src.read_bytes()).hexdigest(); out_name=f'SHY-CAD-CONV-{digest[:20]}.step'
 temp=pathlib.Path('/tmp')/(out_name+'.tmp')
 try:
  reader=IGESControl_Reader(); st=reader.ReadFile(str(src))
  if st!=IFSelect_ReturnStatus.IFSelect_RetDone: return dict(source=src_name,error='IGES read failed')
  if reader.TransferRoots()<1:return dict(source=src_name,error='IGES transfer failed')
  shape=reader.OneShape(); box=Bnd_Box();BRepBndLib.Add_s(shape,box);v=box.Get();dims=[round(v[i+3]-v[i],4) for i in range(3)]
  writer=STEPControl_Writer();st=writer.Transfer(shape,STEPControl_AsIs)
  if st!=IFSelect_ReturnStatus.IFSelect_RetDone:return dict(source=src_name,error='STEP transfer failed')
  st=writer.Write(str(temp))
  if st!=IFSelect_ReturnStatus.IFSelect_RetDone:return dict(source=src_name,error='STEP write failed')
  if temp.stat().st_size<128:return dict(source=src_name,error='invalid STEP output')
  h=hashlib.sha256()
  with temp.open('rb') as f:
   head=f.read(128);h.update(head)
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  if b'ISO-10303-21' not in head:return dict(source=src_name,error='invalid STEP output')
  return dict(source=src_name,file=out_name,sha256=h.hexdigest(),bytes=temp.stat().st_size,envelope_mm=dims,temp=str(temp))
 except Exception as e:return dict(source=src_name,error=repr(e))

def main():
 hw=json.loads((CAT/'hardware.json').read_text());manifest=json.loads((CAT/'manifest.json').read_text());by={}
 for row in manifest:
  if row.get('item_id','').startswith('SHY-') and row.get('status') in ('downloaded','converted'):
   by.setdefault(row['item_id'],[]).append(row)
 need={}
 for item_id,rows in by.items():
  if any(a.get('kind','').lower() in ('step','stp') for a in rows):continue
  for a in rows:
   if a.get('kind','').lower() in ('igs','iges'):need.setdefault(a['file'],[]).append((item_id,a))
 done=json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {};todo=[f for f in sorted(need) if f not in done]
 print(f'IGES-only source models={len(need)} already converted={len(done)} remaining={len(todo)}',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  for n,result in enumerate(pool.map(convert,todo,chunksize=1),1):
   done[result['source']]=result
   if n%10==0 or n==len(todo):
    CHECKPOINT.write_text(json.dumps(done,indent=2)); print(f'progress {len(done)}/{len(need)} errors={sum("error" in x for x in done.values())}',flush=True)
 # Append derived assets, keyed by item and file, preserve every original source row.
 rows={(a['item_id'],a['file']):a for a in manifest if a.get('file')}
 items={x['id']:x for x in hw['items']}
 for source,result in done.items():
  if 'error' in result:continue
  src=ASSETS/source; dest=ASSETS/result['file']
  if not dest.exists():shutil.move(result['temp'],dest)
  else:pathlib.Path(result['temp']).unlink(missing_ok=True)
  for item_id,orig in need[source]:
   if (item_id,result['file']) in rows:continue
   rows[(item_id,result['file'])]=dict(item_id=item_id,kind='step',file=result['file'],url=orig['url'],sha256=result['sha256'],bytes=dest.stat().st_size,status='converted',retrieved_at=datetime.now(timezone.utc).isoformat(),manufacturer_file_name=orig.get('manufacturer_file_name'),source_page=orig.get('source_page'),source_page_title=orig.get('source_page_title'),source_model_label=orig.get('source_model_label'),derived_from=source,derivation='OpenCascade IGES-to-STEP transfer; preserve source IGES alongside this layout/display derivative; dimensions and model fidelity require vendor confirmation.',derived_envelope_mm=result['envelope_mm'])
   if result['file'] not in items[item_id].setdefault('cad_files',[]):items[item_id]['cad_files'].append(result['file'])
   items[item_id]['cad_status']='source_iges_downloaded_and_step_derivative_available_unverified'
 for item in hw['items']:
  if item['id'].startswith('SHY-') and item.get('cad_envelope_file') and item['cad_envelope_file'].lower().endswith(('.igs','.iges')):
   if item.get('cad_envelope_status')=='extracted_from_source_3d_model_unverified':item['cad_envelope_status']='extracted_from_source_iges_unverified'
 hw['revision']='2026-10-14.13'
 (CAT/'hardware.json').write_text(json.dumps(hw,ensure_ascii=False,indent=2)+'\n')
 (CAT/'manifest.json').write_text(json.dumps(sorted(rows.values(),key=lambda x:(x['item_id'],x['file'],x.get('kind',''))),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(igs_only_models=len(need),converted=sum('error' not in x for x in done.values()),conversion_errors=[x for x in done.values() if 'error' in x][:10],new_catalog_items=len(hw['items']),derived_assets=sum(x.get('status')=='converted' and x.get('file','').startswith('SHY-CAD-CONV-') for x in rows.values())),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
