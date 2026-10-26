#!/usr/bin/env python3
"""Check STEP files from official Daheng drawings with OpenCascade BRepCheck_Analyzer."""
from __future__ import annotations
import concurrent.futures,json,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1];CAT=ROOT/'oracle_viewer/catalog';ASSETS=CAT/'assets';OUT=CAT/'daheng_brep_validation.json'
def validate(filename):
 started=time.time()
 try:
  from OCP.BRepCheck import BRepCheck_Analyzer
  from OCP.STEPControl import STEPControl_Reader
  reader=STEPControl_Reader();st=reader.ReadFile(str(ASSETS/filename))
  if 'RetDone' not in str(st):return {'file':filename,'status':'read_failed'}
  n=reader.TransferRoots()
  if not n:return {'file':filename,'status':'transfer_failed'}
  valid=bool(BRepCheck_Analyzer(reader.OneShape()).IsValid())
  return {'file':filename,'status':'valid' if valid else 'invalid','seconds':round(time.time()-started,2)}
 except Exception as e:return {'file':filename,'status':'error','error':repr(e),'seconds':round(time.time()-started,2)}
def main():
 h=json.loads((CAT/'hardware.json').read_text());names=sorted({f for x in h['items'] if x.get('vendor','').startswith('DAHENG') for f in x.get('cad_files',[])})
 results={};print('validating',len(names),'STEP solids with OpenCascade',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  for i,r in enumerate(pool.map(validate,names,chunksize=1),1):
   results[r['file']]=r
   if i%20==0 or i==len(names):
    OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    from collections import Counter
    print(i,dict(Counter(x['status'] for x in results.values())),flush=True)
 OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
