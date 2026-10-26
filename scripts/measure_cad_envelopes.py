#!/usr/bin/env python3
"""Read CAD envelopes for layout reference; never treat as certified installation dimensions."""
from __future__ import annotations
import concurrent.futures,json,pathlib,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];CAT=ROOT/'oracle_viewer/catalog';ASSETS=CAT/'assets'
OUTPUT=pathlib.Path('/tmp/shuangyi-cad-metrics.json')

def measure(name):
 p=ASSETS/name; started=time.time()
 try:
  from OCP.Bnd import Bnd_Box
  from OCP.BRepBndLib import BRepBndLib
  from OCP.IGESControl import IGESControl_Reader
  from OCP.STEPControl import STEPControl_Reader
  reader=IGESControl_Reader() if p.suffix.lower() in ('.igs','.iges') else STEPControl_Reader()
  status=reader.ReadFile(str(p))
  if 'RetDone' not in str(status):return dict(file=name,status='read_failed',seconds=round(time.time()-started,3))
  transferred=reader.TransferRoots()
  if not transferred:return dict(file=name,status='transfer_failed',seconds=round(time.time()-started,3))
  shape=reader.OneShape(); box=Bnd_Box();BRepBndLib.Add_s(shape,box);vals=box.Get()
  dims=[round(vals[i+3]-vals[i],4) for i in range(3)]
  if not all(0<d<100000 for d in dims):return dict(file=name,status='invalid_bounds',bounds=dims,seconds=round(time.time()-started,3))
  return dict(file=name,status='bounded',envelope_mm=dims,seconds=round(time.time()-started,3))
 except Exception as e:return dict(file=name,status='error',error=repr(e),seconds=round(time.time()-started,3))

def main():
 manifest=json.loads((CAT/'manifest.json').read_text())
 names=sorted({x['file'] for x in manifest if x.get('item_id','').startswith('SHY-') and x.get('kind','').lower() in ('step','stp','igs','iges')})
 old=json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {};todo=[n for n in names if n not in old]
 print(f'files={len(names)} completed={len(old)} remaining={len(todo)}',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  for i,row in enumerate(pool.map(measure,todo,chunksize=1),1):
   old[row['file']]=row
   if i%25==0 or i==len(todo):
    OUTPUT.write_text(json.dumps(old,indent=2)); counts={s:sum(x['status']==s for x in old.values()) for s in set(x['status'] for x in old.values())}
    print(f'progress {len(old)}/{len(names)} current={row["file"]} {row["status"]} seconds={row["seconds"]} states={counts}',flush=True)
 OUTPUT.write_text(json.dumps(old,indent=2))
if __name__=='__main__':main()
