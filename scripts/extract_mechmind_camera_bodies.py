#!/usr/bin/env python3
"""Derive camera-only STEP compounds from official camera+FOV STEP assets.

The cutoffs are per source model and documented here because camera dimensions
vary. FOV solids are separated only when their individual CAD AABB exceeds that
source-specific observed cutoff. Original files remain intact and linked.
"""
from __future__ import annotations
import hashlib,json,pathlib
from datetime import datetime,timezone
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRep import BRep_Builder
from OCP.STEPControl import STEPControl_AsIs,STEPControl_Reader,STEPControl_Writer
from OCP.TopAbs import TopAbs_ShapeEnum as T
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound
ROOT=pathlib.Path(__file__).resolve().parents[1];CAT=ROOT/'oracle_viewer/catalog';ASSETS=CAT/'assets'
FILES={
 'Mech-Eye_Deep(V3)_FOV.STEP':1000,
 'Mech-Eye_Laser_L(V3S)_FOV.STEP':1000,
 'Mech-Eye_Laser_L_Enhanced-12MP(V3S)_FOV.STEP':1000,
 'Mech-Eye_Nano(V3)_FOV.STEP':300,
 'Mech-Eye_Pro_M_Enhanced(V3)_FOV.STEP':1000,
 'Mech-Eye_Pro_S_Enhanced(V3)_FOV.STEP':500,
}
def main():
 h=json.loads((CAT/'hardware.json').read_text());m=json.loads((CAT/'manifest.json').read_text());items={i['id']:i for i in h['items']};manifest={(a.get('item_id'),a.get('file')):a for a in m if a.get('file')};out=[]
 for original,cutoff in FILES.items():
  src=next((ASSETS/f for f in ASSETS.glob('MM-CAD-*') if manifest.get((None,f.name)) is None and False),None)
  # Resolve original catalog asset by the recorded manufacturer filename.
  row=next((a for a in m if a.get('manufacturer_file_name')==original),None)
  if not row:raise FileNotFoundError('No linked source model '+original)
  src=ASSETS/row['file'];reader=STEPControl_Reader();status=reader.ReadFile(str(src))
  if 'RetDone' not in str(status) or not reader.TransferRoots():raise RuntimeError('Could not read '+original)
  root_shape=reader.OneShape();ex=TopExp_Explorer(root_shape,T.TopAbs_SOLID);keep=[];drop=[]
  while ex.More():
   solid=ex.Current();b=Bnd_Box();BRepBndLib.Add_s(solid,b);v=b.Get();dims=[v[i+3]-v[i] for i in range(3)]
   if max(dims)>cutoff:drop.append([round(x,2) for x in dims])
   else:keep.append(solid)
   ex.Next()
  if not keep or not drop:raise RuntimeError(f'FOV/body split ambiguous for {original}: keep={len(keep)} drop={len(drop)}')
  if any(not BRepCheck_Analyzer(s).IsValid() for s in keep):raise RuntimeError('Retained camera component fails BRep validation: '+original)
  compound=TopoDS_Compound();builder=BRep_Builder();builder.MakeCompound(compound)
  for solid in keep:builder.Add(compound,solid)
  writer=STEPControl_Writer();writer.Transfer(compound,STEPControl_AsIs)
  temp=ASSETS/'.mechmind-camera-body.tmp.step';write_status=writer.Write(str(temp))
  if 'RetDone' not in str(write_status):temp.unlink(missing_ok=True);raise RuntimeError('STEP export failed '+original)
  data=temp.read_bytes();digest=hashlib.sha256(data).hexdigest();name=f'MM-CAMERA-CAD-{digest[:20]}.step';dest=ASSETS/name
  if dest.exists():temp.unlink(missing_ok=True)
  else:temp.replace(dest)
  item_id=row['item_id'];item=items[item_id]
  item['cad_files']=[name,row['file']];item['cad_body_model_file']=name;item['cad_preview_only_files']=[row['file']]
  item['cad_body_extraction']={'method':'retained source STEP solids with per-solid AABB maximum dimension <= documented cutoff; FOV solids exceed cutoff','cutoff_mm':cutoff,'retained_solids':len(keep),'removed_fov_solids':len(drop),'removed_solid_aabbs_mm':drop,'review_status':'automatically separated source FOV geometry; verify against mechanical drawing before collision/installation approval'}
  item['cad_status']='manufacturer_official_step_with_derived_camera_body_and_source_fov'
  item['cad_scope']='Derived camera-only compound uses original source STEP solids; large solids above the model-specific cutoff are retained separately in the original camera+FOV source. Treat derived body as layout candidate only until checked against official mounting drawing.'
  new={'item_id':item_id,'kind':'step','file':name,'url':row.get('url'),'sha256':digest,'bytes':dest.stat().st_size,'status':'converted','retrieved_at':datetime.now(timezone.utc).isoformat(),'manufacturer_file_name':original+' [camera-only derived]','derivation':f'OpenCascade STEP solid extraction; retained {len(keep)} source solids, separated {len(drop)} FOV solids exceeding {cutoff} mm max AABB dimension. Original full camera+FOV STEP remains available.','source_page':row.get('source_page'),'source_page_title':row.get('source_page_title'),'source_model_label':row.get('source_model_label'),'cad_scope':'Derived from exact official STEP source solids; FOV-only solids removed by documented geometric threshold. Not a vendor-issued camera-only model.'}
  manifest[(item_id,name)]=new;out.append({'source':original,'derived':name,'kept_solids':len(keep),'separated_fov_solids':len(drop),'cutoff_mm':cutoff,'derived_bytes':dest.stat().st_size})
 h['revision']='2026-10-14.20';(CAT/'hardware.json').write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n');(CAT/'manifest.json').write_text(json.dumps(sorted(manifest.values(),key=lambda a:(a.get('item_id',''),a['file'],a.get('kind',''))),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'revision':h['revision'],'models':out},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
