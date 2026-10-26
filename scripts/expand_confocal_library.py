import requests,re,json,html
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'scripts'))
from expand_hardware_library import base,snapshot,ROOT
s=requests.Session();s.trust_env=False;url='https://en.hypersen.com/chromatic-confocal-sensor/';r=s.get(url,timeout=25);r.raise_for_status();r.encoding='utf-8'
items=[];assets=[]
for tr in re.findall(r'<tr[^>]*>(.*?)</tr>',r.text,re.S):
 cells=[html.unescape(re.sub('<[^>]+>','',v)).strip() for v in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>',tr,re.S)]
 if len(cells)<6 or not cells[0].startswith('HPS-CFL'):continue
 numeric=lambda v:float(re.search(r'[\d.]+',v).group())
 spec=dict(working_distance_mm=numeric(cells[1]),measuring_range_half_mm=numeric(cells[2]),nonlinear_error_um=numeric(cells[3]),angle_half_deg=numeric(cells[4]),spot_um=numeric(cells[5]),manufacturer_parameters=dict(zip(['model','working_distance','range','nonlinear_error','angle','spot'],cells)),rating_conditions='Refer to manufacturer table footnotes; controller-dependent, standard version, no installed-system accuracy claim')
 i=base(cells[0],'confocal_sensor','Hypersen',url,spec);i['unknown_fields']+=['head_body_dimensions','compatible_controller_model','controller_channels','installed_accuracy'];items.append(i);assets.append(snapshot(i['id'],'html',url,r.content))
cat=json.loads((ROOT/'hardware.json').read_text());m=json.loads((ROOT/'manifest.json').read_text());known={i['id']:i for i in cat['items']};known.update({i['id']:i for i in items});cat['items']=list(known.values());m=[a for a in m if a['item_id'] not in [i['id'] for i in items]]+assets;(ROOT/'hardware.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2));(ROOT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));print('Hypersen models',len(items))
