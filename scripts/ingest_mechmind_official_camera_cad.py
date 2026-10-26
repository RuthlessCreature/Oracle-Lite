#!/usr/bin/env python3
"""Add official Mech-Mind camera-plus-FOV STEP models to the local AOI catalog.

These files intentionally remain distinct from mechanical-only camera models:
source naming and imported solids show that FOV volumes are included.
"""
from __future__ import annotations
import hashlib,json,pathlib,shutil
from datetime import datetime,timezone
ROOT=pathlib.Path(__file__).resolve().parents[1];CAT=ROOT/'oracle_viewer/catalog';ASSETS=CAT/'assets';SRC=pathlib.Path('/tmp/mech-dh')
PAGE='https://downloads.mech-mind.com.cn/?tab=tab-camera-materials'
MODELS={
 'Mech-Eye_Deep(V3)_FOV.STEP':('MECH-EYE-DEEP-V3-FOV','Mech-Eye Deep V3','Deep','includes FOV geometry volumes; not clean camera-only CAD'),
 'Mech-Eye_Laser_L(V3)_FOV.STEP':('MECH-EYE-LASER-L-V3-FOV','Mech-Eye Laser L V3','Laser L','includes FOV geometry volumes; STEP has no closed solids in initial OCCT inspection'),
 'Mech-Eye_Laser_L(V3S)_FOV.STEP':('MECH-EYE-LASER-L-V3S-FOV','Mech-Eye Laser L V3S','Laser L','includes FOV geometry volumes; not clean camera-only CAD'),
 'Mech-Eye_Laser_L_Enhanced-12MP(V3S)_FOV.STEP':('MECH-EYE-LASER-L-ENHANCED-12MP-V3S-FOV','Mech-Eye Laser L Enhanced 12MP V3S','Laser L Enhanced','includes FOV geometry volumes; not clean camera-only CAD'),
 'Mech-Eye_Nano(V3)_FOV.STEP':('MECH-EYE-NANO-V3-FOV','Mech-Eye Nano V3','Nano','includes FOV geometry volumes; not clean camera-only CAD'),
 'Mech-Eye_Pro_M_Enhanced(V3)_FOV.STEP':('MECH-EYE-PRO-M-ENHANCED-V3-FOV','Mech-Eye Pro M Enhanced V3','Pro M Enhanced','includes FOV geometry volumes; not clean camera-only CAD'),
 'Mech-Eye_Pro_S_Enhanced(V3)_FOV.STEP':('MECH-EYE-PRO-S-ENHANCED-V3-FOV','Mech-Eye Pro S Enhanced V3','Pro S Enhanced','includes FOV geometry volumes; not clean camera-only CAD'),
}
h=json.loads((CAT/'hardware.json').read_text());manifest=json.loads((CAT/'manifest.json').read_text());items={i['id']:i for i in h['items']};rows={(a.get('item_id'),a.get('file')):a for a in manifest if a.get('file')};added=0;records=0
for source,(item_id,model,series,scope) in MODELS.items():
 p=SRC/source
 if not p.exists():raise FileNotFoundError(p)
 data=p.read_bytes()
 if b'ISO-10303-21' not in data[:256]:raise ValueError('Invalid STEP header: '+source)
 sha=hashlib.sha256(data).hexdigest();name=f'MM-CAD-{sha[:20]}.step';dest=ASSETS/name
 if not dest.exists():shutil.copyfile(p,dest);added+=1
 item=items.get(item_id)
 if not item:
  item={
   'id':item_id,'kind':'surface_sensor','vendor':'Mech-Mind (manufacturer official CAD)',
   'source_url':PAGE,'datasheet_url':None,'spec':{},'price':None,
   'availability':'Official download-center file; exact order-code lifecycle and Mainland stock unverified',
   'lifecycle_status':'official_CAD_listed_commercial_status_unverified',
   'cad_status':'manufacturer_official_step_fov_geometry_physical_only_separation_pending',
   'review_status':'official_model_downloaded; includes FOV solids; mechanical envelope must not use full source bbox',
   'unknown_fields':['exact order code and hardware revision','current production and availability','mechanical-only body envelope','FOV/body separation map','price'],
   'field_source':{},'source_model':model,'source_title':series+' 3D model with FOV',
   'source_vendor_status':'manufacturer download center; old V3/V3S geometry version; commercial order code mapping unverified',
   'cad_files':[name],'cad_scope':'Official manufacturer STEP model named as camera plus FOV. It contains very large optical FOV solids. Preserve source for inspection; do not use its full bounding box for device collision/clearance or bind FAI until camera body and FOV bodies are separated and reviewed.',
   'cad_preview_only_files':[name],'cad_brep_validation':'unchecked','source_archive':None,
  }
  items[item_id]=item;records+=1
 row={'item_id':item_id,'kind':'step','file':name,'url':PAGE,'sha256':sha,'bytes':dest.stat().st_size,'status':'downloaded','retrieved_at':datetime.now(timezone.utc).isoformat(),'manufacturer_file_name':source,'cad_scope':scope,'source_page':PAGE,'source_page_title':series+' official 3D model with FOV','source_model_label':model,'cad_geometry_note':'model is camera plus optical FOV geometry; physical camera and FOV solids not separated'}
 rows[(item_id,name)]=row
h['items']=sorted(items.values(),key=lambda x:(x['kind'],x.get('vendor',''),x['id']));h['revision']='2026-10-14.19'
(CAT/'hardware.json').write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n');(CAT/'manifest.json').write_text(json.dumps(sorted(rows.values(),key=lambda x:(x.get('item_id',''),x['file'],x.get('kind',''))),ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'revision':h['revision'],'models_added':records,'source_STEP_assets_copied':added,'new_models':len(MODELS),'bytes_added':sum((ASSETS/f'MM-CAD-{hashlib.sha256((SRC/n).read_bytes()).hexdigest()[:20]}.step').stat().st_size for n in MODELS)},indent=2))
