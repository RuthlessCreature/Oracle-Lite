#!/usr/bin/env python3
"""Time-boxed FreeCAD B-Rep audit for STEP/STP/IGES assets.

A pathological source file must not stall the catalog-wide audit. Each file is
validated in its own FreeCAD process; timeout is recorded as unresolved, never
as a geometry failure.
"""
from __future__ import annotations
import concurrent.futures,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];CAT=ROOT/'oracle_viewer/catalog';ASSETS=CAT/'assets';OUT=CAT/'cad_renderability_audit.json';WORKER=ROOT/'scripts/cad_renderability_worker.py'
SUPPORTED={'.step','.stp','.igs','.iges'};TIMEOUT_SECONDS=35;MAX_WORKERS=4

def validate(name):
 started=time.time();path=ASSETS/name
 try:
  p=subprocess.run([sys.executable,str(WORKER),str(path)],capture_output=True,text=True,timeout=TIMEOUT_SECONDS)
  result=next((line[len('ORACLE_RESULT '):] for line in reversed(p.stdout.splitlines()) if line.startswith('ORACLE_RESULT ')),None)
  if result:return json.loads(result)
  return {'file':name,'status':'error','seconds':round(time.time()-started,2),'error':(p.stderr or p.stdout or f'worker exit {p.returncode}')[-1200:]}
 except subprocess.TimeoutExpired:
  return {'file':name,'status':'timed_out','timeout_seconds':TIMEOUT_SECONDS,'seconds':round(time.time()-started,2),'error':'FreeCAD import or B-Rep check exceeded per-file time limit; geometry remains unresolved'}
 except Exception as exc:return {'file':name,'status':'error','seconds':round(time.time()-started,2),'error':repr(exc)}

def save(rows):OUT.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
def main():
 names=sorted(p.name for p in ASSETS.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
 rows=json.loads(OUT.read_text()) if OUT.exists() else {};todo=[n for n in names if n not in rows]
 print(f'FreeCAD renderability: {len(names)} solid CAD assets; {len(rows)} cached; {len(todo)} pending; {MAX_WORKERS} workers x {TIMEOUT_SECONDS}s/file cap',flush=True)
 done=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
  futures={pool.submit(validate,name):name for name in todo}
  for future in concurrent.futures.as_completed(futures):
   name=futures[future]
   try: rows[name]=future.result()
   except Exception as exc: rows[name]={'file':name,'status':'error','error':repr(exc)}
   done+=1
   if done%5==0 or done==len(todo):
    save(rows)
    from collections import Counter
    print(f'progress {len(rows)}/{len(names)} latest={name} {rows[name]["status"]} states={dict(Counter(x["status"] for x in rows.values()))}',flush=True)
 save(rows)
if __name__=='__main__':main()
