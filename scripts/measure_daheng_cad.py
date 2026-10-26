#!/usr/bin/env python3
"""Audit STEP BRep import and CAD-coordinate envelopes for official Daheng assets."""
from __future__ import annotations
import concurrent.futures,json,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1]; CAT=ROOT/'oracle_viewer/catalog'; ASSETS=CAT/'assets'; OUT=CAT/'daheng_geometry_audit.json'
def measure(filename):
 p=ASSETS/filename; started=time.time()
 try:
  from OCP.Bnd import Bnd_Box
  from OCP.BRepBndLib import BRepBndLib
  from OCP.STEPControl import STEPControl_Reader
  reader=STEPControl_Reader(); status=reader.ReadFile(str(p))
  if 'RetDone' not in str(status): return {'file':filename,'status':'read_failed','seconds':round(time.time()-started,2)}
  transferred=reader.TransferRoots()
  if not transferred:return {'file':filename,'status':'transfer_failed','seconds':round(time.time()-started,2)}
  shape=reader.OneShape(); box=Bnd_Box(); BRepBndLib.Add_s(shape,box); vals=box.Get()
  dims=[round(vals[i+3]-vals[i],4) for i in range(3)]
  if not all(0<d<100000 for d in dims):return {'file':filename,'status':'invalid_bounds','envelope_mm':dims,'seconds':round(time.time()-started,2)}
  return {'file':filename,'status':'bounded','envelope_mm':dims,'seconds':round(time.time()-started,2)}
 except Exception as e:return {'file':filename,'status':'error','error':repr(e),'seconds':round(time.time()-started,2)}
def main():
 hw=json.loads((CAT/'hardware.json').read_text()); names=sorted({f for x in hw['items'] if x.get('vendor','').startswith('DAHENG') for f in x.get('cad_files',[])})
 prior=json.loads(OUT.read_text()) if OUT.exists() else {}; todo=[f for f in names if f not in prior]
 print(f'Daheng STEP assets={len(names)}, already audited={len(prior)}, remaining={len(todo)}',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  for i,row in enumerate(pool.map(measure,todo,chunksize=1),1):
   prior[row['file']]=row
   if i%10==0 or i==len(todo):
    OUT.write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n')
    from collections import Counter
    print(f'progress {len(prior)}/{len(names)} latest={row["file"]}:{row["status"]} counts={dict(Counter(x["status"] for x in prior.values()))}',flush=True)
 OUT.write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
