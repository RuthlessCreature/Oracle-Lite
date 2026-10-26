import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from expand_hardware_library import base,snapshot,ROOT
rows=[('MV-CS050-10GM','camera','https://www.hikrobotics.com/cn2/source/vision/document/2023/12/6/MV-CS050-10GM(NPOE)_20230508.pdf',dict(width_px=2448,height_px=2048,pixel_um=3.45,mount='C',shutter='global',color='mono',interface='GigE',fps=24.2,body_mm=[29,29,42],power_w=2.2,supply_voltage_v=[9,24],poe=False)),('MV-DP3020-01H','surface_sensor','https://www.hikrobotics.com/cn2/source/vision/document/2023/7/15/MV-DP3020-01H_20230329.pdf',dict(points_per_profile=3200,reference_distance_mm=21,z_range_mm=6,x_range_mm=[11.5,13.2,14],z_resolution_um=[.5,1],repeatability_um=.15,repeatability_conditions='Standard gauge block on optical bench; mean over 4096 samples in restricted range; not installed measurement accuracy',linearity_pct_mr=.01,profile_rate_hz=1300,roi_max_rate_hz=19000,wavelength_nm=405,laser_class='3R',interface='GigE',supply_voltage_v=[12,24],power_w=13,body_mm=[139,111,59.5],sdk='3DMVS Windows 7/10 per 2023 sheet'))]
cat=json.loads((ROOT/'hardware.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text());known={i['id']:i for i in cat['items']}
for id,kind,url,spec in rows:
 i=base(id,kind,'HIKROBOT',url,spec);i['datasheet_url']=url;i['review_status']='manufacturer_pdf_fields_verified_via_web_reader';i['layout_adapter_ready']=kind=='camera';i['unknown_fields']+=['local_original_PDF_blocked_by_vendor_WAF'];known[id]=i
 # Explicitly a curated extract of verified facts, not a downloaded original PDF.
 content=json.dumps({'source_url':url,'extraction':'manufacturer PDF read via web tool 2026-10-10; direct download returned WAF HTML','spec':spec},ensure_ascii=False,indent=2).encode();manifest=[m for m in manifest if m['item_id']!=id]+[snapshot(id,'txt',url,content),dict(item_id=id,kind='pdf',url=url,status='failed',error='Vendor WAF returned HTML, original PDF not stored')]
cat['items']=list(known.values());(ROOT/'hardware.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2));(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('models',len(known))
