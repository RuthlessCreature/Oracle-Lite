"""Literal MSOP measurement-manifest supervision; no CAD labels or XML input."""
import copy
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oracle_viewer.store import ViewerStore
from oracle_viewer.local_matching import _declared_marker_manifest
from scripts.prepare_cloud_qwen_training import sha


def prepare(output):
    import fitz
    store=ViewerStore(ROOT/'talker/state/viewer')
    output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir(exist_ok=True)
    splits={'train':[],'validation':[]};excluded=[]
    for summary in store.listing():
        project=store.get(summary['id']);group=project.get('drawing_no')
        if group=='817-07781':continue
        source=store.active_msop(project)
        if not source:continue
        if sha(Path(source['path']).read_bytes())!=source['sha256']:raise ValueError('Changed source PDF')
        with fitz.open(source['path']) as pdf:
            for item in store.items(project):
                methods=store.detail(project['id'],item['number'])['methods']
                groups=_declared_marker_manifest(methods)
                if not groups:continue
                # Keep single-page manifests so no evidence image is silently omitted.
                pages={g['source_page'] for g in groups}
                if len(pages)!=1:
                    excluded.append({'project':group,'fai':item['number'],'reason':'multi_page_manifest_requires_multi_image_collator'});continue
                page=next(iter(pages));text=next(p['text'] for p in methods if p['page']==page)
                names=list(dict.fromkeys(n for g in groups for n in g['source_point_names']))
                image=Path('images')/f'{source["sha256"][:16]}-{page:04d}.png'
                if not (output/image).exists():pdf[page-1].get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(output/image)
                question=(f'Confirm the complete directly measured feature manifest for FAI {item["number"]}. '
                    'Read the original instruction and table. Expand explicit ranges. Do not include constructed '
                    'features or auxiliary retrieved coordinates. Output only JSON with decision=CONFIRM_MANIFEST, '
                    'fai, source_point_names (each exact printed name once), evidence, confidence (high/medium/low strings only). '
                    'Reject any candidate name absent from the printed directly measured list with decision=ABSTAIN and source_point_names=[]. This is source interpretation only, never CAD placement. Explicit source instruction groups: '+json.dumps(groups,ensure_ascii=False))
                answer={'decision':'CONFIRM_MANIFEST','fai':item['number'],'source_point_names':names,
                        'evidence':' | '.join(g['instruction'] for g in groups),'confidence':'high'}
                row=dict(id=f'{group}:manifest:{item["number"]}',project_group=group,page=page,
                    task='fai_direct_measurement_manifest',question=question,context=text,images=[str(image)],
                    answer=json.dumps(answer,ensure_ascii=False,separators=(',',':')),
                    source_file=source['filename'],source_sha256=source['sha256'],image_sha256=sha((output/image).read_bytes()),
                    geometry_ground_truth=False,xml_used_as_input=False,label_scope='literal_measurement_manifest_not_CAD_registration')
                split='validation' if group=='613-16076' else 'train'
                splits[split].append(row)
                # Paired rejection prevents learning unconditional confirmation of supplied candidates.
                corrupt=copy.deepcopy(groups)
                corrupt[0]['source_point_names'].append('UNPRINTED_POINT_999')
                rejected=copy.deepcopy(row)
                rejected['id']+=':invalid-candidate'
                rejected['question']=question.split('Explicit source instruction groups: ')[0]+'Explicit source instruction groups: '+json.dumps(corrupt,ensure_ascii=False)
                rejected['answer']=json.dumps(dict(decision='ABSTAIN',fai=item['number'],source_point_names=[],evidence='Candidate contains a point name absent from the original measurement instruction.',confidence='high'),separators=(',',':'))
                splits[split].append(rejected)
    hashes={}
    for split,rows in splits.items():
        if not rows:raise ValueError('Empty split')
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();(output/f'{split}.jsonl').write_bytes(data);hashes[split]=sha(data)
    if {r['source_sha256'] for r in splits['train']}&{r['source_sha256'] for r in splits['validation']}:raise ValueError('Source leakage')
    manifest=dict(schema='oracle-fai-manifest-training-v1',training_scope='FAI_direct_measurement_manifest',
        is_FAI_geometry_training=False,is_XML_generation_training=False,XML_role='final_output_only',
        train_groups=sorted({r['project_group'] for r in splits['train']}),validation_groups=['613-16076'],
        samples={k:len(v) for k,v in splits.items()},sha256=hashes,
        point_count={k:sum(len(json.loads(r['answer'])['source_point_names']) for r in v) for k,v in splits.items()},excluded_pages=excluded)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.output),ensure_ascii=False,indent=2))
