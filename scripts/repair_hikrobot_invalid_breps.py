#!/usr/bin/env python3
"""Try metric-preserving ShapeFix on exact-label HIKROBOT STEP sources.

Only attach a derivative after OpenCascade reports unchanged bbox/volume/area,
FreeCAD reports a valid round-trip B-Rep, and the Viewer geometry worker builds a mesh.
Original vendor assets and archives remain untouched.
"""
from __future__ import annotations
import datetime as dt,hashlib,json,pathlib,re,shutil,subprocess,sys,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1];CAT=ROOT/'oracle_viewer/catalog';ASSETS=CAT/'assets';REPAIR=ROOT/'scripts/repair_step_brep_for_preview.py';VALIDATE=ROOT/'scripts/cad_renderability_worker.py';MESH=ROOT/'oracle_viewer/geometry_worker.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def safe(s):return re.sub(r'[^A-Za-z0-9._-]+','-',s).strip('-._')
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 hp,mp,ap=CAT/'hardware.json',CAT/'manifest.json',CAT/'cad_renderability_audit.json';hw,manifest,audit=read(hp),read(mp),read(ap);items={i['id']:i for i in hw['items']};midx={(r.get('item_id'),r.get('file')):r for r in manifest};candidates=[]
 for item in hw['items']:
  if item.get('vendor')!='HIKROBOT' or item.get('cad_brep_validation')!='invalid':continue
  for name in item.get('cad_files',[]):
   row=audit.get(name,{})
   if pathlib.Path(name).suffix.lower() in {'.step','.stp'} and row.get('status')=='invalid_brep':
    candidates.append((item['id'],name,sha(ASSETS/name),row))
 groups={}
 for rec in candidates:groups.setdefault(rec[2],[]).append(rec)
 results=[];errors=[];added=[];time_now=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
 with tempfile.TemporaryDirectory(prefix='hikrobot-shapefix-') as td:
  for source_hash,records in groups.items():
   item_id,source_name,_,source_audit=records[0];source=ASSETS/source_name;temp=pathlib.Path(td)/(source_hash[:16]+'-fixed.step')
   try:
    p=subprocess.run([sys.executable,str(REPAIR),str(source),str(temp)],capture_output=True,text=True,timeout=90)
    if p.returncode or not temp.is_file():raise RuntimeError((p.stdout+'\n'+p.stderr)[-1200:])
    check=subprocess.run([sys.executable,str(VALIDATE),str(temp)],capture_output=True,text=True,timeout=45)
    line=next((x[len('ORACLE_RESULT '):] for x in check.stdout.splitlines() if x.startswith('ORACLE_RESULT ')),None)
    validated=json.loads(line) if line else {'status':'error','error':(check.stderr or check.stdout)[-800:]}
    if validated.get('status')!='renderable':raise RuntimeError(f'FreeCAD round-trip rejected derivative: {validated}')
    mesh_out=pathlib.Path(td)/(source_hash[:16]+'-mesh.json')
    mesh=subprocess.run([sys.executable,str(MESH),str(temp),str(mesh_out)],capture_output=True,text=True,timeout=90)
    line=next((x[len('ORACLE_RESULT='):] for x in mesh.stdout.splitlines() if x.startswith('ORACLE_RESULT=')),None)
    if mesh.returncode or not line:raise RuntimeError('Viewer geometry worker rejected derivative: '+(mesh.stderr or mesh.stdout)[-800:])
    mesh_meta=json.loads(line);fixed_hash=sha(temp)
    for target_id,origin_name,_,origin_audit in records:
     out_name=f'HIK-{safe(target_id)}-{source_hash[:10]}-shape-fixed.step';out=ASSETS/out_name
     if not out.exists():shutil.copyfile(temp,out)
     item=items[target_id];srcrow=midx[(target_id,origin_name)]
     if out_name not in item.setdefault('cad_files',[]):item['cad_files'].append(out_name)
     repair={'derived_file':out_name,'source_file':origin_name,'source_sha256':source_hash,'derived_sha256':fixed_hash,'method':'OpenCascade ShapeFix_Shape; precision 1e-5 mm; max tolerance 1e-2 mm','source_geometry_unchanged':True,'validation':'FreeCAD Part.read + Shape.isValid and Viewer geometry_worker mesh passed','face_count':validated.get('face_count'),'triangle_count':mesh_meta.get('triangle_count'),'bbox_mm':validated.get('bbox_mm'),'review_status':'metric-preserving full-shape derivative; source CAD and archive remain attached'}
     item.setdefault('cad_repairs',[]).append(repair)
     item['cad_scope']=(item.get('cad_scope','')+' Full-shape ShapeFix derivative is separately attached; source B-Rep metrics were preserved and original remains unchanged.').strip()
     row={'item_id':target_id,'kind':'step','file':out_name,'sha256':fixed_hash,'bytes':out.stat().st_size,'status':'converted','retrieved_at':time_now,'manufacturer_file_name':out_name,'source_model_label':target_id,'source_page':srcrow.get('source_page'),'source_archive_file':srcrow.get('source_archive_file'),'source_archive_sha256':srcrow.get('source_archive_sha256'),'derived_from_file':origin_name,'derived_from_sha256':source_hash,'brep_validation':'valid','body_count':validated.get('body_count'),'face_count':validated.get('face_count'),'bbox_mm':validated.get('bbox_mm'),'geometry_check':'OCP ShapeFix_Shape; metric-preserving; FreeCAD Part.read + Shape.isValid; Viewer geometry_worker mesh','cad_scope':'Full-shape derivative from the exact model item source STEP; unchanged source bounding box, volume and surface area; original source preserved.'}
     if not any(x.get('item_id')==target_id and x.get('file')==out_name for x in manifest):manifest.append(row)
     audit[out_name]={'file':out_name,'status':'renderable_derived_shape_fixed','brep_valid':True,'body_count':validated.get('body_count'),'face_count':validated.get('face_count'),'bbox_mm':validated.get('bbox_mm'),'geometry_check':row['geometry_check'],'seconds':0}
     added.append({'item_id':target_id,'source':origin_name,'file':out_name,'faces':validated.get('face_count'),'triangles':mesh_meta.get('triangle_count')})
    results.append({'source_sha256':source_hash,'source_files':[r[1] for r in records],'source_status':'invalid_brep','repair':'passed metric preservation, FreeCAD B-Rep round-trip, and Viewer mesh'})
   except Exception as exc:errors.append({'source_sha256':source_hash,'source_files':[r[1] for r in records],'error':repr(exc)[:1500]})
 if added:
  hw['items']=sorted(hw['items'],key=lambda x:(x.get('kind',''),x.get('vendor',''),x['id']));manifest.sort(key=lambda x:(x.get('item_id',''),x.get('file',''),x.get('kind','')));write(hp,hw);write(mp,manifest);write(ap,audit)
 print(json.dumps({'candidate_files':len(candidates),'unique_source_geometries':len(groups),'repaired_geometries':len(results),'registered_derivatives':len(added),'errors':errors,'results':results,'derivatives':added},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
