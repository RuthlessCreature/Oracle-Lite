"""Source-pinned local-model classification and strategy selection for every FAI.

Classification never grants a geometry placement or promotes existing marks to
verified ground truth. Successful and unresolved source items share one taxonomy.
"""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

FAMILIES={
 'coordinate':'坐标尺寸','distance':'距离尺寸','flatness':'平面度','straightness':'直线度',
 'parallelism':'平行度','perpendicularity':'垂直度','profile':'轮廓度','circle':'圆与孔',
 'mechanical_test':'力学试验位置','weld_height':'焊接高度与焊点','visual_inspection':'外观检查','other':'其他测量'}
EVIDENCE={
 'complete_xyz_single_frame':'完整 XYZ · 单坐标系','complete_xyz_multiple_frames':'完整 XYZ · 多坐标系',
 'partial_xyz':'部分 XYZ','named_features_no_xyz':'有点名或特征 · 无完整 XYZ',
 'drawing_pattern_no_xyz':'图纸阵列或重复实例 · 无 XYZ','explicit_absent_entity':'原文明确当前装配不含实体',
 'conflicting_sources':'原文定义冲突','unknown':'证据类型待确认'}
FAILURES={
 'none':'本轮无失败记录','source_manifest':'测量点或特征清单不完整','coordinate_scope':'坐标版本或归属不清',
 'frame_registration':'坐标配准未通过','part_identity':'实体身份对应未确认','repeated_pattern':'重复实例或阵列未确认',
 'visibility':'可见性或表面落点未通过','emission_contract':'模型结构化输出不完整',
 'insufficient_evidence':'当前证据不足','source_conflict':'原文定义冲突','unknown':'原因待确认'}
ROUTES={
 'rigid_full_group':'整组 XYZ 刚体配准','split_frame_then_join':'按原坐标系分组，再合并验证',
 'datum_graph_reconstruction':'重建基准与坐标系关系图',
 'feature_topology_graph':'先识别 CAD 零部件与拓扑，再定位测量特征',
 'drawing_pattern_to_part':'先匹配图纸重复实例，再展开整组点位',
 'assembly_applicability':'核对装配阶段与实体适用性',
 'evidence_repair':'重新抽取原文清单与定义关系'}

def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)

def strategy_path(store,pid):return store._dir(pid)/'automatic-learning'/'fai-strategies.json'

def strategy_state(store,pid):
    path=strategy_path(store,pid)
    if not path.is_file():return {'status':'not_started','items':[],'families':FAMILIES,'evidence_classes':EVIDENCE,'failure_classes':FAILURES,'routes':ROUTES}
    data=json.loads(path.read_text());project=store.get(pid);msop=store.active_msop(project) or {}
    data['source_stale']=(data.get('msop_sha256')!=msop.get('sha256') or data.get('cad_sha256')!=(project.get('model') or {}).get('source_sha256'))
    trials=path.parent/'strategy-route-trials.json'
    if trials.is_file():
        trial_data=json.loads(trials.read_text())
        data['route_trials']=[{'fai':row['fai'],'route':row['route'],
            'status':row.get('audit',{}).get('status'),'reason':row.get('audit',{}).get('reason'),
            'decision':row.get('audit',{}).get('decision')} for row in trial_data.get('results',[])]
    xyz_trials=path.parent/'xyz-route-trials.json'
    if xyz_trials.is_file():
        trial_data=json.loads(xyz_trials.read_text())
        if (trial_data.get('cad_sha256')==(project.get('model') or {}).get('source_sha256')
                and trial_data.get('msop_sha256')==msop.get('sha256')):
            data.setdefault('route_trials',[]).extend(trial_data.get('results',[]))
    return data


def archive_marking_state(store,pid,state,*,legacy_report=None):
    """Keep failure diagnostics when the next run clears its live counters."""
    if state.get('status') not in ('partial','completed','failed','blocked'):return
    project=store.get(pid);msop=store.active_msop(project) or {}
    receipt={'state':state,'msop_sha256':msop.get('sha256'),
        'cad_sha256':(project.get('model') or {}).get('source_sha256'),
        'legacy_report':legacy_report,'historical_source_hash_independently_recorded':legacy_report is None}
    content=json.dumps(receipt,ensure_ascii=False,sort_keys=True).encode()
    target=store._dir(pid)/'automatic-learning'/'marking-runs'/f'{hashlib.sha256(content).hexdigest()}.json'
    if not target.is_file():_write(target,receipt)


def collect_strategy_sources(store,pid):
    from .candidate_mining import extract_named_points,extract_feature_labels,resolve_dominant_source_page_conflicts
    from .msop_coordinates import _coordinate_rows
    from .automatic_candidates import _source_point_evidence,_required_marker_count
    from .local_matching import _coalesce_source_point_definitions,_declared_marker_manifest,_canonical_source_point_alias
    project=store.get(pid);msop=store.active_msop(project);named=extract_named_points(msop['method_pages'])
    outcomes={row['fai']:row for row in (project.get('local_model_marking') or {}).get('items',[])}
    histories=[]
    for path in (store._dir(pid)/'automatic-learning'/'marking-runs').glob('*.json'):
        saved=json.loads(path.read_text())
        if (saved.get('cad_sha256')==(project.get('model') or {}).get('source_sha256')
                and saved.get('msop_sha256')==msop.get('sha256')):
            histories.append(saved)
    for saved in sorted(histories,key=lambda row:row['state'].get('updated_at',''),reverse=True):
        for row in saved['state'].get('items',[]):
            if row.get('fai') not in outcomes:
                outcomes[row['fai']]={**row,'historical_diagnostic':True,
                    'historical_source_hash_independently_recorded':saved.get('historical_source_hash_independently_recorded',False)}
    rows=[]
    for item in store.items(project):
        detail=store.detail(pid,item['number']);methods=detail['methods']
        coordinates=_coordinate_rows(methods,[],extract_feature_labels(methods,item.get('features',[])),msop['method_pages'],named)
        coordinates,_=resolve_dominant_source_page_conflicts(coordinates)
        points=_coalesce_source_point_definitions(_source_point_evidence(coordinates))
        annotation=item.get('annotation') or {};failure=outcomes.get(item['number']) or {}
        rows.append({'fai':item['number'],'drawing_no':project.get('drawing_no'),'revision':project.get('revision'),
            'methods':[{key:page.get(key) for key in ('page','text','program','measured_system','dimension_system')} for page in methods],
            'source_points':points,'explicit_required_count':_required_marker_count(methods,named),
            'measurement_indicators':item.get('measurement_indicators',[]),
            'outcome':{'status':annotation.get('status','unmarked'),'origin':annotation.get('origin'),
                'confidence':annotation.get('confidence'),'anchor_count':len(annotation.get('anchors',[])),
                'is_verified_ground_truth':False,'failure_reason':failure.get('reason'),
                'validation_failures':failure.get('validation_failures',[]),
                'geometry_failure':(failure.get('geometry_diagnostic') or {}).get('reason')}})
        if failure.get('historical_diagnostic'):
            rows[-1]['outcome'].update(historical_diagnostic=True,
                historical_diagnostic_is_geometry_authority=False)
        manifest=_declared_marker_manifest(methods) or []
        declared=list(dict.fromkeys(name for group in manifest for name in group['source_point_names']))
        matches={_canonical_source_point_alias(point['name']):point for point in points}
        missing=[name for name in declared if _canonical_source_point_alias(name) not in matches
            or not (matches[_canonical_source_point_alias(name)].get('xyz')
                or matches[_canonical_source_point_alias(name)].get('coordinate_variants'))]
        if missing:
            rows[-1].update(direct_measurement_manifest=manifest,
                missing_xyz_for_declared_points=missing,
                coordinate_retrieval_warning='Retrieved similarly named rows cannot replace these explicitly measured names.')
    return project,msop,rows


def strategy_source_sha256(source,source_key):
    """Ignore retrieval order while retaining every coordinate and source fact."""
    def canonical(value,key=''):
        if isinstance(value,dict):return {k:canonical(v,k) for k,v in value.items() if k!='input_sha256'}
        if isinstance(value,list):
            values=[canonical(row) for row in value]
            if key in ('source_points','coordinate_variants','pages','source_pages'):
                return sorted(values,key=lambda row:json.dumps(row,sort_keys=True,ensure_ascii=False))
            return values  # XYZ vectors and original source instructions keep order.
        return value
    return hashlib.sha256(json.dumps({'source':canonical(source),**source_key},sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def _validate_batch(parsed,sources):
    if not isinstance(parsed,dict) or parsed.get('decision')!='CLASSIFY':return False,[]
    items=parsed.get('items');expected={row['fai']:row for row in sources}
    if not isinstance(items,list) or len(items)!=len(expected):return False,[]
    seen=set();normalized=[]
    for row in items:
        if not isinstance(row,dict) or type(row.get('fai')) is not int or row['fai'] not in expected or row['fai'] in seen:return False,[]
        seen.add(row['fai']);source=expected[row['fai']]
        if (row.get('measurement_family') not in FAMILIES or row.get('evidence_class') not in EVIDENCE
            or row.get('failure_class') not in FAILURES or row.get('next_route') not in ROUTES
            or not isinstance(row.get('reason'),str) or not row['reason'].strip()):return False,[]
        quotes=row.get('source_quotes')
        if not isinstance(quotes,list) or not quotes:return False,[]
        texts={page['page']:re.sub(r'\s+',' ',page.get('text') or '').strip().casefold() for page in source['methods']}
        for quote in quotes:
            if not isinstance(quote,dict) or type(quote.get('page')) is not int or not isinstance(quote.get('text'),str):return False,[]
            value=re.sub(r'\s+',' ',quote['text']).strip().casefold()
            if len(value)<4 or value not in texts.get(quote['page'],''):return False,[]
        if source.get('missing_xyz_for_declared_points') and row['evidence_class'].startswith('complete_xyz_'):return False,[]
        if row['evidence_class']=='explicit_absent_entity' and row['next_route']!='assembly_applicability':return False,[]
        normalized.append({**row,'outcome_at_classification':source['outcome'],
            'source_pages':[page['page'] for page in source['methods']]})
    return seen==set(expected),normalized

def _classify_project(store,pid,model,*,batch_size=5):
    from .local_matching import _object,_complete_visual_context_budget,_is_local_inference_model
    if not model.loaded or not _is_local_inference_model(model):raise RuntimeError('Classification requires a verified local model')
    project,msop,sources=collect_strategy_sources(store,pid)
    source_key={'msop_sha256':msop['sha256'],'cad_sha256':(project.get('model') or {}).get('source_sha256'),
        'classifier_revision':1,'model_alias':model.info().get('model_alias'),'inference_backend':model.info().get('inference_backend')}
    path=strategy_path(store,pid);old=strategy_state(store,pid)
    state={'status':'running','items':[],'done':0,'total':len(sources),'updated_at':datetime.now(timezone.utc).isoformat(),
        **source_key,'families':FAMILIES,'evidence_classes':EVIDENCE,'failure_classes':FAILURES,'routes':ROUTES,
        'model':model.info(),'classification_is_not_geometry_authorization':True,'errors':[]}
    _write(path,state)
    # Unresolved cases first. Existing marked cases remain evidence examples,
    # explicitly not assumed to be correct or metrologically validated.
    sources.sort(key=lambda row:(row['outcome']['status']=='auto_marked',row['fai']))
    cache={row.get('source_input_sha256'):row for row in old.get('items',[]) if not old.get('source_stale')}
    # Re-key older receipts only when their complete original input is retained.
    # This is a cache-format migration, never a new semantic classification.
    receipts={}
    for row in old.get('items',[]) if not old.get('source_stale') else []:
        if row.get('source_input_hash_kind')=='canonical-v1':continue
        reference=row.get('receipt')
        if not reference:continue
        receipt_path=Path(reference)
        if not receipt_path.is_file() or hashlib.sha256(receipt_path.read_bytes()).hexdigest()!=row.get('receipt_sha256'):continue
        if reference not in receipts:receipts[reference]=json.loads(receipt_path.read_text())
        saved=receipts[reference]
        if saved.get('source_key')!=source_key:continue
        source=next((source for source in saved.get('sources',[]) if source.get('fai')==row['fai']),None)
        if source is not None:cache[strategy_source_sha256(source,source_key)]=row
    batches=[];current=[]
    for source in sources:
        digest=strategy_source_sha256(source,source_key)
        source['input_sha256']=digest
        if digest in cache:
            state['items'].append({**cache[digest],'source_input_sha256':digest,'source_input_hash_kind':'canonical-v1'});state['done']+=1;continue
        trial=[*current,source]
        if current and (len(trial)>batch_size or len(json.dumps(trial,ensure_ascii=False))>43000):batches.append(current);current=[]
        current.append(source)
    if current:batches.append(current)
    for index,batch in enumerate(batches):
        state.update(current_fai=[row['fai'] for row in batch],batch=index+1,batch_total=len(batches))
        _write(path,state)
        question=('你是本地 Oracle MSOP 方法分类与路线选择模型。本轮不定位、不修改点位。'
            '对每个FAI按原始测量方法、坐标证据和本轮失败诊断分类，使用两个项目通用的类别。'
            '已标记项仅是已有结果样本，不能当作正确真值。未标记项应按原因选择不同的新路线，不要统一回到六视图猜像素。'
            '有完整XYZ优先利用完整点组；多坐标系保留各自基准关系；无XYZ的焊点应先识别零件和重复实例，不能在任意面上凑点。'
            '若missing_xyz_for_declared_points非空，原文直接测量点没有完整XYZ，不得用检索到的相似点名宣称complete_xyz。'
            '原文明确当前装配无此实体时选择装配适用性核对，不伪造几何位置，也不自行改成已标记。'
            'source_quotes仅引用本项methods原文，每项1到2条短摘录并标page，每条优先3至12个连续原文单词。'
            'PDF多列文字可能交错，请选短的连续标签或短语；不得用省略号、拼接分散片段或重排版面来伪造连续引用，保持原字母和数字。'
            '只输出 JSON：decision="CLASSIFY",items数组，每项含fai,measurement_family,evidence_class,failure_class,next_route,'
            'source_quotes:[{page,text}],reason（一句中文说明为何选择该路线）。每个输入FAI恰好一次。'
            f'\n可用测量类别：{json.dumps(FAMILIES,ensure_ascii=False)}'
            f'\n可用证据类别：{json.dumps(EVIDENCE,ensure_ascii=False)}'
            f'\n可用失败类别：{json.dumps(FAILURES,ensure_ascii=False)}'
            f'\n可用路线：{json.dumps(ROUTES,ensure_ascii=False)}')
        attempts=[];accepted={};remaining=list(batch);selected=None;corrections=[]
        for attempt in range(2):
            content=question+('仅重答本次仍未通过校验的FAI；其他FAI已独立保存。请核对类别代码与逐字原文引用，禁止修正原字母或数字。' if attempt else '')
            if corrections:content+='\n上次未通过的独立原文校验：'+json.dumps(corrections,ensure_ascii=False)
            evidence=json.dumps(remaining,ensure_ascii=False,separators=(',',':'))
            messages=[{'role':'user','content':content,'attachments':[{'filename':'Complete source methods and point definitions','parsed_text':evidence}]}]
            try:
                raw,truncated,tokens=model._generate_once(messages,max_context_chars=_complete_visual_context_budget(messages),max_new_tokens=4096,max_images=0,temperature=0.)
                parsed=_object(raw) or {};valid,items=_validate_batch(parsed,remaining)
                row_results=[];corrections=[]
                if not truncated and parsed.get('decision')=='CLASSIFY' and isinstance(parsed.get('items'),list):
                    reported=parsed['items']
                    for source in remaining:
                        matches=[row for row in reported if isinstance(row,dict) and type(row.get('fai')) is int and row['fai']==source['fai']]
                        row_valid,rows=_validate_batch({'decision':'CLASSIFY','items':matches},[source])
                        row_results.append({'fai':source['fai'],'valid':row_valid})
                        if row_valid:accepted[source['fai']]=rows[0]
                        elif len(matches)==1:
                            row=matches[0];texts={page['page']:re.sub(r'\s+',' ',page.get('text') or '').strip().casefold() for page in source['methods']}
                            invalid_quotes=[quote for quote in row.get('source_quotes',[]) if not isinstance(quote,dict)
                                or not isinstance(quote.get('text'),str) or re.sub(r'\s+',' ',quote['text']).strip().casefold() not in texts.get(quote.get('page'),'')]
                            corrections.append({'fai':source['fai'],'noncontiguous_or_invalid_quotes':invalid_quotes,
                                'missing_xyz_for_declared_points':source.get('missing_xyz_for_declared_points',[])})
                attempts.append({'raw_output':raw,'truncated':bool(truncated),'tokens':tokens,'valid':valid,
                    'independent_FAI_validation':row_results})
                remaining=[row for row in remaining if row['fai'] not in accepted]
                if not remaining:break
            except Exception as exc:attempts.append({'error':str(exc)[:1000]})
        receipt={'source_key':source_key,'sources':batch,'attempts':attempts,'model':model.info()}
        raw_receipt=json.dumps(receipt,ensure_ascii=False,indent=2).encode();digest=hashlib.sha256(raw_receipt).hexdigest()
        receipt_path=path.parent/'strategy-evidence'/f'{digest}.json';_write(receipt_path,receipt)
        selected=list(accepted.values())
        if selected:
            inputs={row['fai']:row['input_sha256'] for row in batch}
            state['items'].extend({**row,'source_input_sha256':inputs[row['fai']],'source_input_hash_kind':'canonical-v1',
                'receipt':str(receipt_path),'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest()} for row in selected)
        if remaining:state['errors'].append({'fai':[row['fai'] for row in remaining],'receipt':str(receipt_path),'reason':'local_classification_unconfirmed'})
        state['done']+=len(batch);state['updated_at']=datetime.now(timezone.utc).isoformat();_write(path,state)
    current=store.get(pid)
    if (store.active_msop(current) or {}).get('sha256')!=source_key['msop_sha256'] or (current.get('model') or {}).get('source_sha256')!=source_key['cad_sha256']:
        state['status']='stale'
    else:state['status']='partial' if state['errors'] else 'completed'
    state['current_fai']=[];_write(path,state)
    return state


def classify_project(store,pid,model,*,batch_size=5):
    import fcntl
    lock_path=strategy_path(store,pid).with_suffix('.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        return _classify_project(store,pid,model,batch_size=batch_size)
