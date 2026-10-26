#!/usr/bin/env python3
"""Ingest official DAHENG IMAGING STEP models from its technical drawings library.

The source ZIPs are retained verbatim under catalog/source_archives/daheng; individual
CAD files are extracted with ZIP path/type checks, content-addressed, and linked to
model records derived from the manufacturer's own CAD filenames. Family names are
not promoted to exact commercial SKUs unless the source CAD itself provides one.
"""
from __future__ import annotations
import hashlib, json, pathlib, re, shutil, zipfile
from datetime import datetime, timezone

ROOT=pathlib.Path(__file__).resolve().parents[1]
CAT=ROOT/'oracle_viewer/catalog'; ASSETS=CAT/'assets'; ARCHIVES=CAT/'source_archives/daheng'
SOURCE='https://en.daheng-imaging.com/list-61-1.html'
INDEX=pathlib.Path('/tmp/daheng-techdrawings.json')
CAD_EXT={'.step','.stp','.x_t','.x_b','.igs','.iges','.sat'}

def code_from_filename(name:str)->str:
    stem=pathlib.PurePosixPath(name).stem
    stem=re.sub(r'(?i)_3d_(modeling|drawing)$','',stem)
    stem=re.sub(r'(?i)_user_[as]\d+(?:\.\d+\.\d+)?$','',stem)
    stem=re.sub(r'(?i)_s\d+\.\d+\.\d+$','',stem)
    stem=re.sub(r'(?i)_a\d+\.\d+\.\d+$','',stem)
    stem=re.sub(r'(?i)_user$','',stem)
    return re.sub(r'\s+',' ',stem.replace('_','-')).strip('- ')

def safe_id(kind, model):
    return 'DH-'+kind.upper()+'-'+hashlib.sha1((kind+'|'+model.casefold()).encode()).hexdigest()[:12]

def main():
    rows=json.loads(INDEX.read_text()); hw=json.loads((CAT/'hardware.json').read_text()); manifest=json.loads((CAT/'manifest.json').read_text())
    items={x['id']:x for x in hw['items']}; mrows={(x.get('item_id'),x.get('file')):x for x in manifest if x.get('file')}
    ASSETS.mkdir(parents=True,exist_ok=True); ARCHIVES.mkdir(parents=True,exist_ok=True)
    archive_copies=0; cad_links=0; failed=[]; entries=0; seen_sha={}
    for row in rows:
        if row.get('status')!='downloaded': failed.append((row.get('title'),row.get('error'))); continue
        source_zip=pathlib.Path(row['path']); archive_name=f"{row['sha256'][:16]}_{source_zip.name}"
        archived=ARCHIVES/archive_name
        if not archived.exists(): shutil.copyfile(source_zip,archived); archive_copies+=1
        group_is_lens='lens' in row.get('product_group','').lower()
        kind='lens' if group_is_lens else 'camera'
        try: zf=zipfile.ZipFile(source_zip)
        except Exception as e: failed.append((row['title'],repr(e))); continue
        for info in zf.infolist():
            path=pathlib.PurePosixPath(info.filename)
            if path.suffix.lower() not in CAD_EXT or info.is_dir(): continue
            if path.is_absolute() or '..' in path.parts or info.file_size<=128 or info.file_size>2_000_000_000:
                failed.append((row['title'],f'unsafe/invalid member {info.filename}')); continue
            try: data=zf.read(info)
            except Exception as e: failed.append((row['title'],f'{info.filename}: {e}')); continue
            head=data[:256]
            ext=path.suffix.lower()
            if ext in {'.step','.stp'} and b'ISO-10303-21' not in head: failed.append((row['title'],f'bad STEP header: {info.filename}')); continue
            digest=hashlib.sha256(data).hexdigest(); code=code_from_filename(path.name)
            # Exact byte-identical CAD files share one asset; unique actual CAD exports keep their own record.
            filename=seen_sha.get(digest)
            if not filename:
                filename=f'DH-CAD-{digest[:20]}{ext}'; dest=ASSETS/filename
                if not dest.exists(): dest.write_bytes(data)
                filename=filename; seen_sha[digest]=filename
            item_id=safe_id(kind,code); entries+=1
            if item_id not in items:
                items[item_id]={
                    'id':item_id,'kind':kind,'vendor':'DAHENG IMAGING (manufacturer official CAD)',
                    'source_url':SOURCE,'datasheet_url':None,'spec':{},'price':None,
                    'availability':'Manufacturer CAD is listed; Mainland stock and lead time require supplier confirmation',
                    'lifecycle_status':'manufacturer_documented_status_unverified',
                    'cad_status':'manufacturer_official_step_downloaded_sku_mapping_pending',
                    'review_status':'official_manufacturer_CAD_filename; commercial_variant_mapping_pending',
                    'unknown_fields':['commercial_order_code','technical_specifications','current_lifecycle_status','China_stock_and_lead_time','price'],
                    'field_source':{},'source_group':row.get('product_group'),'source_model':code,
                    'source_model_label':path.name,'source_title':row.get('title'),'cad_files':[],
                    'cad_scope':'Original STEP from DAHENG IMAGING official technical drawings ZIP; source CAD filename retained. Verify exact order code, enclosure revision, and interfaces before installation approval.',
                    'source_archive':f'source_archives/daheng/{archive_name}'
                }
            item=items[item_id]
            if filename not in item.setdefault('cad_files',[]): item['cad_files'].append(filename)
            asset={
                'item_id':item_id,'kind':ext.lstrip('.'),'file':filename,'url':row['url'],
                'sha256':digest,'bytes':len(data),'status':'downloaded','retrieved_at':datetime.now(timezone.utc).isoformat(),
                'manufacturer_file_name':info.filename,'cad_scope':'official manufacturer STEP; commercial model mapping pending',
                'source_page':SOURCE,'source_page_title':row.get('title'),'source_model_label':path.name,
                'source_archive':f'source_archives/daheng/{archive_name}'
            }
            mrows[(item_id,filename)]=asset; cad_links+=1
    hw['items']=sorted(items.values(),key=lambda x:(x['kind'],x.get('vendor',''),x['id']))
    try: rev=int(hw.get('revision','2026-10-14.14').split('.')[-1])+1
    except ValueError: rev=15
    hw['revision']=f'2026-10-14.{rev}'
    (CAT/'hardware.json').write_text(json.dumps(hw,ensure_ascii=False,indent=2)+'\n')
    (CAT/'manifest.json').write_text(json.dumps(sorted(mrows.values(),key=lambda x:(x.get('item_id',''),x['file'],x.get('kind',''))),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'catalog_revision':hw['revision'],'downloaded_source_archives':sum(r.get('status')=='downloaded' for r in rows),'retained_archive_copies':archive_copies,'CAD_exports_seen':entries,'catalog_links_added_or_refreshed':cad_links,'catalog_total_items':len(hw['items']),'new_DH_items':sum(x.get('vendor')=='DAHENG IMAGING (manufacturer official CAD)' for x in items.values()),'unique_DH_CAD_assets':len(seen_sha),'errors':failed[:20],'error_count':len(failed)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
