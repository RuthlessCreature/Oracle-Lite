"""Snapshot public manufacturer parameter tables. Missing specifications stay missing.
OPT public HTTPS currently has an expired certificate: only its unauthenticated
public tables use a recorded TLS exception. No login/download restriction bypass.
"""
import concurrent.futures,datetime,hashlib,json,re,subprocess
from pathlib import Path
import requests
import html
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ROOT=Path(__file__).resolve().parents[1]/'oracle_viewer/catalog'; ASSETS=ROOT/'assets'
OPT={'light':['17_'+str(x) for x in [720,819,833,2353,2355,2329,2285,809,805,2317,14638,14639,14643,14667,14692,14694,2743,13763,14812,3191,3184,3176,2433,2399,2459,2405,2389,2384,1815,14177,14191]],'lens':['75_'+str(x) for x in [36575,36576,36555,36617,36570,36621,36629,36579,54060,36624,54150]],'controller':['72_46439']}
IRAY=['A5501MG20E_Datasheet_2023-09-04','A5200MG000E_Datasheet_2023-09-04','AE5200MG010E_Datasheet_2025-10-16','AH5200MG000E_Datasheet_2025-11-09','A7500MG20E_Datasheet_2023-09-04','A7500CG20E_Datasheet_2023-09-04','A7500CU75E_Datasheet_2023-09-04','AE7500CG000E_Datasheet_2025-10-16','AE7500CG010E_Datasheet_2025-10-16']
def snapshot(id,kind,url,blob,tls=True):
 name=id+'.'+kind;(ASSETS/name).write_bytes(blob)
 return dict(item_id=id,kind=kind,file=name,url=url,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),status='downloaded',retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),tls_verified=tls)
def base(id,kind,vendor,url,spec):
 return dict(id=id,kind=kind,vendor=vendor,source_url=url,datasheet_url=None,spec=spec,price=None,cad_status='not_downloaded',field_source={k:'source_url' for k in spec},review_status='manufacturer_parameter_table',unknown_fields=['price','availability','vendor_CAD'],layout_adapter_ready=False)
def number(v):
 m=re.search(r'[-+]?\d+(?:\.\d+)?',v or '');return float(m.group()) if m else None
def opt(pair):
 kind,key=pair;s=requests.Session();s.trust_env=False;url='https://en.optmv.com/content/performance_details'+key+'.html'
 r=s.get(url,timeout=25,verify=False);r.raise_for_status();r.encoding='utf-8';params={html.unescape(re.sub('<[^>]+>','',a)).strip():html.unescape(re.sub('<[A-Za-z/][^>]*>','',b)).strip() for a,b in re.findall(r'<tr><th>(.*?)</th><td>(.*?)</td></tr>',r.text,re.S)}
 id=params.get('Model');
 if not id:raise ValueError('No manufacturer model '+key)
 spec={'manufacturer_parameters':params}
 if kind=='light':
  spec['type']='backlight' if 'FL' in id else 'coaxial' if 'CO' in id or 'LSSC' in id else 'dome' if 'RID' in id else 'bar' if 'LI' in id else 'ring'
  for source,dest in [('Input Voltage','voltage_v'),('Power','power_w'),('Weight(KG)','weight_kg'),('Emitting Surface (Diameter)','emitting_diameter_mm')]:
   v=number(params.get(source));
   if v is not None:spec[dest]=v
  if params.get('LED Color'):spec['color']=params['LED Color']
  if params.get('Recommended Controller'):spec['recommended_controller']=params['Recommended Controller']
  missing=['body_dimensions','light_distance','emission_geometry']
 elif kind=='lens':
  for source,dest in [('Working Distance(mm)','working_distance_mm'),('Magnification（X）','magnification'),('Image Circle Ø(mm)','image_circle_mm'),('Depth Of Field(mm)','dof_mm'),('MTF30(lp/mm)','mtf30_lp_mm'),('Telecentricity','telecentricity_max_deg')]:
   v=number(params.get(source));
   if v is not None:spec[dest]=v
  if params.get('Mount'):spec['mount']=params['Mount']
  spec['telecentric']='manufacturer_telecentric_series';missing=['front_diameter_mm','length_mm','DOF_test_conditions']
 else:
  # Manufacturer's 5A/4CH is an aggregate rating; never 5A per channel.
  spec.update(channels=4,supply_voltage_v=[22,25],total_current_a=5,interface='RS232/Ethernet',mode='continuous/strobe');missing=['per_channel_continuous_current','output_voltage_limits','connector','body_dimensions']
 item=base(id,kind,'OPT',url,spec);item['product_url']='https://en.optmv.com/content/details'+key+'.html';item['unknown_fields']+=missing+['PDF_requires_vendor_registration'];item['transport_note']='Public HTTPS table downloaded with certificate-expiry exception; tls_verified=false in manifest'
 return item,[snapshot(id,'html',url,r.content,False)]
def iray(key):
 id=key.split('_Datasheet')[0];url='https://file.irayple.com/dataSheet/en/'+key+'.pdf';s=requests.Session();s.trust_env=False;r=s.get(url,timeout=35);r.raise_for_status()
 if not r.content.startswith(b'%PDF'):raise ValueError('Invalid PDF '+id)
 asset=snapshot(id,'pdf',url,r.content);text=subprocess.check_output(['pdftotext','-layout',str(ASSETS/asset['file']),'-']).decode();spec={}
 def value(label):
  m=re.search(r'^\s*'+re.escape(label)+r'\s{2,}(.+)$',text,re.M);return m.group(1).strip() if m else ''
 res=re.findall(r'(\d+)\s*[×x]\s*(\d+)',value('Resolution'));dim=re.findall(r'[\d.]+',value('Product Dimensions').split('(')[0]);pixel=number(value('Pixel Size'))
 if res:spec.update(width_px=int(res[0][0]),height_px=int(res[0][1]))
 if pixel:spec['pixel_um']=pixel
 if len(dim)==3:spec['body_mm']=list(map(float,dim))
 if value('Lens Mount').startswith('C'):spec['mount']='C'
 if value('Shutter'):spec['shutter']=value('Shutter').lower()
 if value('Mono/Color'):spec['color']=value('Mono/Color').lower()
 if value('Frame Rate'):spec['fps']=number(value('Frame Rate'))
 if value('Sensor'):spec['sensor']=value('Sensor')
 spec['interface']=value('Port');spec['body_excludes']='lens mount and rear connector';spec['power_supply_description']=value('Power Supply');spec['power_description']=value('Power Consumption')
 item=base(id,'camera','HuaRay / iRAYPLE (Dahua machine vision)',url,spec);item['datasheet_url']=url;item['field_source']={k:'datasheet_url' for k in spec};item['layout_adapter_ready']=all(k in spec for k in ['width_px','height_px','pixel_um','body_mm','mount']);item['unknown_fields']+=['connector_clearance'];return item,[asset,snapshot(id,'txt',url,text.encode())]
def main():
 cat=json.loads((ROOT/'hardware.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text());out=[];fail=[]
 jobs=[(opt,p) for k,vs in OPT.items() for p in [(k,v) for v in vs]]+[(iray,x) for x in IRAY]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  pending=[pool.submit(fn,arg) for fn,arg in jobs]
  for f in concurrent.futures.as_completed(pending):
   try:item,assets=f.result();out.append(item);manifest=[m for m in manifest if not (m['item_id']==item['id'] and m['kind'] in [a['kind'] for a in assets])]+assets
   except Exception as e:fail.append(str(e))
 existing={i['id']:i for i in cat['items']};existing.update({i['id']:i for i in sorted(out,key=lambda i:(i['kind'],i['id']))});cat['items']=list(existing.values());cat['revision']='2026-10-10.2';cat['notes']+=['Public parameter tables are retained when PDF downloads require vendor registration. Partial entries cannot silently enter geometric planning.']
 (ROOT/'hardware.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2));(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps({'added':len(out),'total':len(cat['items']),'failed':fail,'kinds':{k:sum(i['kind']==k for i in cat['items']) for k in ['camera','lens','light','controller']}},ensure_ascii=False))
if __name__=='__main__':main()
