#!/usr/bin/env python3
"""Import exact AOI hardware SKUs from HIKROBOT's public product-list API.
Only currently listed/available items are imported. Missing engineering fields remain unknown.
"""
from __future__ import annotations
import argparse, datetime as dt, json, re, sys
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
CAT=ROOT/'oracle_viewer/catalog'
API='https://www.hikrobotics.com/en/Api/Foreground/Vision/'
SITE='https://www.hikrobotics.com/en/machinevision'
CATEGORIES=[
 # parent id, child id, catalog kind, subtype, source-friendly label
 (78,145,'camera','area_scan','Area Scan Camera / All Series'),
 (146,155,'camera','line_scan','Line Scan Camera / CL Series'),
 (40,49,'lens','fa','FA Lens'),
 (40,180,'lens','m12','M12 Lens'),
 (150,182,'light','light_source','HIK Light Source'),
 (150,181,'controller','light_controller','HIK Light Controller'),
 (150,184,'mechanical_accessory','light_accessory','HIK Light Accessories'),
 (40,195,'mechanical_accessory','lens_accessory','Lens Accessories'),
 (39,185,'controller','vision_controller','VC2000 Vision Controller'),
 (39,176,'controller','vision_controller','VC3000 Vision Controller'),
 (39,140,'controller','vision_controller','VC5000 Vision Controller'),
 (39,193,'controller','vision_controller','VT2000 Vision Controller'),
 (39,227,'controller','vision_controller','VT3000 Vision Controller'),
 (143,149,'controller','frame_grabber','Frame Grabber'),
 (212,214,'surface_sensor','3d_profile_sensor','DP2000 3D Laser Profile Sensor'),
 (212,213,'surface_sensor','3d_profile_sensor','DP3000 3D Laser Profile Sensor'),
 (212,215,'surface_sensor','3d_profile_sensor','DP4000 3D Laser Profile Sensor'),
 (217,220,'surface_sensor','stereo_3d_sensor','RoboScan Stereo 3D Camera'),
 (217,221,'surface_sensor','stereo_3d_sensor','RoboScan Ultra Stereo 3D Camera'),
]

def get_session():
 s=requests.Session();s.headers.update({'User-Agent':'Mozilla/5.0 (compatible; OracleViewerCatalog/1.0)','Accept':'application/json','Referer':SITE+'/visionproduct/'});return s

def fetch_category(s,root,child):
 url=API+'VisionProductContent';params={'firstModuleId':root,'secondaryModuleId':child,'page':1,'size':500,'showEol':'true'}
 r=s.get(url,params=params,timeout=30);r.raise_for_status();body=r.json()
 if body.get('status')!='success':raise RuntimeError(f'API error for {root}/{child}: {body}')
 data=body['data'];page=data['VisionProductContent'];rows=[]
 for page_no in range(1,int(page.get('pages') or 0)+1):
  if page_no==1:current=data
  else:
   params['page']=page_no;r=s.get(url,params=params,timeout=30);r.raise_for_status();body=r.json()
   if body.get('status')!='success':raise RuntimeError(f'API error for {root}/{child}/{page_no}: {body}')
   current=body['data']
  products=current['VisionProductContent']['records'];configs=current.get('VisionProductConfig') or []
  display=current.get('VisionConfigDisplay') or []
  labels=[x.get('name') or f'field_{i+1}' for i,x in enumerate(display)]
  for idx,row in enumerate(products):
   raw=configs[idx] if idx<len(configs) and isinstance(configs[idx],list) else []
   # Preserve the positional API table exactly; display schema is manufacturer-provided.
   row=dict(row);row['_config_labels']=labels;row['_config_values']=raw;row['_display_schema']=display
   rows.append(row)
 return rows

def clean_model(x):
 return re.sub(r'\s+',' ',str(x or '').strip())

def resolution(raw):
 m=re.search(r'(\d+)\s*[×xX]\s*(\d+)',str(raw or ''))
 return (int(m.group(1)),int(m.group(2))) if m else None

def parse_number(raw):
 m=re.search(r'[-+]?\d+(?:\.\d+)?',str(raw or '').replace(',',''))
 return float(m.group()) if m else None

def item_from(row,kind,subtype,category):
 model=clean_model(row.get('productModel'))
 if not model:return None
 source=f"{SITE}/productdetail/?id={row.get('id')}"
 config={clean_model(k):v for k,v in zip(row.get('_config_labels',[]),row.get('_config_values',[])) if k}
 product_name=clean_model(row.get('productName'))
 offering=clean_model(row.get('offeringName'))
 status=clean_model(row.get('productTrends')) or 'status_unpublished'
 spec={'manufacturer_category':category,'manufacturer_product_name':product_name,'manufacturer_offering_name':offering,
       'manufacturer_config':config,'catalog_record_id':row.get('id'),'material_code':row.get('materialCode'),
       'lifecycle_status':status}
 unknown=[]
 if kind=='camera':
  wh=resolution(config.get('Resolution'))
  if wh:
   spec.update(width_px=wh[0],height_px=wh[1],megapixels=round(wh[0]*wh[1]/1_000_000,3))
  else: unknown.append('resolution_px')
  fps=parse_number(config.get('Max. frame rate'))
  if fps is not None:spec['fps']=fps
  for key,sourcekey in [('sensor_model','Sensor model'),('interface','Data interface'),('color','Mono/color')]:
   if config.get(sourcekey):spec[key]=config[sourcekey]
  spec['scan_type']=subtype
  # Pixel pitch and mount are populated only when explicit in manufacturer's name/configuration.
  intro=re.sub(r'<[^>]+>',' ',row.get('productIntroduction') or '')
  intro=re.sub(r'\s+',' ',intro)
  pitch=re.search(r'(?:pixel size|pixel pitch)[^\d]{0,30}(\d+(?:\.\d+)?)\s*(?:µm|um)',intro,re.I)
  if pitch:spec['pixel_um']=float(pitch.group(1))
  mount=re.search(r'\b(C|F|M42|M58)(?:-mount|\s+mount)\b',product_name,re.I)
  if mount:spec['mount']=mount.group(1).upper()
  unknown += [k for k in ['mount','pixel_um','body_mm','connector_clearance'] if k not in spec]
 elif kind=='lens':
  for key,label in [('focal_length_mm','Focal length'),('mount','Mount')]:
   v=config.get(label)
   if key=='focal_length_mm':
    n=parse_number(v)
    if n is not None:spec[key]=n
   elif v:spec[key]=v.replace('-mount','').replace('-Mount','')
  fnum=config.get('F-number')
  if fnum:spec['f_number_text']=fnum
  image=config.get('Image size')
  if image:
   m=re.search(r'[ØΦ]\s*(\d+(?:\.\d+)?)\s*mm',image,re.I)
   if m:spec['image_circle_mm']=float(m.group(1))
   spec['image_size_text']=image
  unknown += [k for k in ['magnification','working_distance_mm','dof_mm','length_mm','front_diameter_mm'] if k not in spec]
 elif kind=='light':
  spec['lighting_family']=offering
  # Geometry and electrical data are retained verbatim below; don't infer physical size from SKU strings.
  for k in ['Color','Power','Voltage','Current','Connector type','LED rows']:
   if config.get(k):spec[k.lower().replace(' ','_')]=config[k]
  unknown += ['active_mm','body_mm','voltage_v_exact','current_a_exact','mounting_envelope']
 elif kind=='controller':
  spec['controller_type']=subtype
  unknown += ['body_mm_model_specific','power_and_io_details','CAD']
 elif kind=='surface_sensor':
  spec['sensor_type']=subtype
  unknown += ['body_mm_model_specific','working_distance_and_measurement_range_for_all_axes','mounting_envelope']
 elif kind=='mechanical_accessory':
  spec['accessory_type']=subtype
  unknown += ['body_mm_model_specific','mounting_envelope']
 url=f"{SITE}/visionproduct/?id={row.get('secondaryModuleId')}&pageSize=500&typeId={row.get('firstModuleId')}"
 return {'id':model,'kind':kind,'vendor':'HIKROBOT','source_url':source,'datasheet_url':None,'spec':spec,
  'price':None,'availability':'Available' if status=='Available' else 'Manufacturer status: '+status,
  'lifecycle_status':status,'cad_status':'no_model_specific_CAD_in_catalog',
  'review_status':'Official HIKROBOT catalog SKU and listing fields imported; missing layout/installation fields remain unverified',
  'field_source':{k:source for k in spec},'unknown_fields':sorted(set(unknown+['price','China_stock_lead_time','model_specific_CAD'])),
  'source_model':model,'source_product_name':product_name,'manufacturer_record_id':row.get('id'),
  'manufacturer_material_code':row.get('materialCode'),'manufacturer_offering':offering,
  'catalog_page':url,'catalog_imported_at':'2026-10-21',
  'product_image_url':('https://www.hikrobotics.com'+row['previewUrl']) if row.get('previewUrl') else None,
  'front_view_image_url':('https://www.hikrobotics.com'+row['frontViewUrl']) if row.get('frontViewUrl') else None,
  'back_view_image_url':('https://www.hikrobotics.com'+row['backViewUrl']) if row.get('backViewUrl') else None,'catalog_priority':'common_aoi_baseline_25mp_plus' if kind=='camera' and spec.get('megapixels',0)>=25 else ('catalog_only_incomplete_for_layout' if kind in ('camera','lens','light') else 'catalog_only_incomplete_for_layout')}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');ap.add_argument('--include-lifecycle',action='store_true');args=ap.parse_args()
 s=get_session(); collected=[];stats={}
 for root,child,kind,subtype,label in CATEGORIES:
  rows=fetch_category(s,root,child)
  # Deduplicate by authoritative model code. Prefer Available over non-current status.
  best={}
  for r in rows:
   model=clean_model(r.get('productModel'))
   if not model:continue
   if not args.include_lifecycle and r.get('productTrends')!='Available':continue
   if model not in best or r.get('productTrends')=='Available':best[model]=r
  items=[item_from(r,kind,subtype,label) for r in best.values()]
  items=[x for x in items if x]
  collected.extend(items);stats[label]={'listed':len(rows),'current_unique_models':len(items)}
 # API may list a camera under more than one series; keep one item per order code, prefer more specific category.
 byid={}
 for item in collected:
  prev=byid.get(item['id'])
  if prev is None or prev.get('spec',{}).get('scan_type')=='area_scan' and item.get('spec',{}).get('scan_type')=='line_scan':byid[item['id']]=item
 collected=list(byid.values())
 hw=json.loads((CAT/'hardware.json').read_text());old={x['id']:x for x in hw['items']};added=[];enriched=[]
 for item in collected:
  if item['id'] not in old:added.append(item)
  elif old[item['id']].get('vendor')=='HIKROBOT':
   # The official catalog can enrich a previously generic family record only with listed provenance,
   # source model and missing model fields. Preserve curated geometry and STEP evidence.
   current=old[item['id']];current.setdefault('catalog_sources',[])
   ref={'url':item['source_url'],'record_id':item['manufacturer_record_id'],'status':item['lifecycle_status'],'product_name':item['source_product_name'],'config':item['spec'].get('manufacturer_config')}
   if ref not in current['catalog_sources']:current['catalog_sources'].append(ref)
   for k,v in item['spec'].items():
    if k not in current.get('spec',{}) and v is not None:current['spec'][k]=v
   for k in ['source_model','manufacturer_record_id','manufacturer_material_code','manufacturer_offering','catalog_page','catalog_imported_at','catalog_priority','product_image_url','front_view_image_url','back_view_image_url']:
    current.setdefault(k,item[k])
   current['field_source']={**current.get('field_source',{}),**{k:v for k,v in item['field_source'].items() if k not in current.get('field_source',{})}}
   enriched.append(item['id'])
 if not args.dry_run:
  for x in added:old[x['id']]=x
  hw['items']=sorted(old.values(),key=lambda x:(x['kind'],x.get('vendor',''),x['id']))
  hw['revision']='2026-10-21.7'
  hw.setdefault('notes',[]).append(f"2026-10-21: Imported {len(added)} new exact HIKROBOT machine-vision SKUs from the manufacturer's public catalog API; availability is retained per SKU and unreported layout parameters remain unknown.")
  (CAT/'hardware.json').write_text(json.dumps(hw,ensure_ascii=False,indent=2)+'\n')
  (CAT/'batches'/'hikrobot-public-api-2026-10-21.json').write_text(json.dumps({'batch':'hikrobot-public-api-2026-10-21','source':'https://www.hikrobotics.com/en/Api/Foreground/Vision/VisionProductContent','imported_at':'2026-10-21','revision':'2026-10-21.7','target_incremental_2000_note':'This batch is additive against the 2000-device expansion target; check master receipt before updating cumulative totals.','categories':stats,'added_count':len(added),'added_ids':[x['id'] for x in added],'enriched_existing_ids':sorted(set(enriched)),'policy':'Exact manufacturer order code only; Available items imported; delisting/unknown lifecycles excluded; original configuration fields preserved verbatim; no SKU expansion or inferred mechanical geometry.'},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'dry_run':args.dry_run,'categories':stats,'current_unique_official_models':len(collected),'would_add':len(added),'would_enrich':len(set(enriched)),'samples':[x['id'] for x in added[:20]]},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
