"""Build geometric registration hypotheses, then let the local model decide."""
import hashlib
import inspect
import json
import math
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from .auto_registration import registration_algorithm_sha256

_REGISTRATION_ATLAS_MESH_CACHE = {}
_REGISTRATION_VISION_DISABLED_REASON = None


def _registration_visual_evidence(store, project_id, model, frame, hypotheses):
    """Create locally sourced image evidence for a model's transform choice.

    The model remains the sole authority: images only expose source MSOP pages
    and candidate projections. Text-only native backends retain their normal
    path, and missing/corrupt display assets never silently become a decision.
    """
    global _REGISTRATION_VISION_DISABLED_REASON
    from .local_matching import _method_page_images
    if _REGISTRATION_VISION_DISABLED_REASON:
        return [], {'status': 'local_vision_disabled_after_failure',
                    'reason': _REGISTRATION_VISION_DISABLED_REASON}
    identity = model.info() if hasattr(model, 'info') else {}
    if identity.get('image_inference') not in ('strata_native_vision', 'local_auxiliary_vision', 'configured_multimodal_backend'):
        return [], {'status': 'local_vision_unavailable'}
    if store is None:
        return [], {'status': 'project_assets_unavailable'}
    try:
        project = store.get(project_id)
        model_record=project.get('model') or {}
        mesh_path=Path(model_record.get('mesh_file') or '')
        cache_key=(str(mesh_path.resolve()),mesh_path.stat().st_mtime_ns,mesh_path.stat().st_size) \
            if mesh_path.is_file() else (project_id,model_record.get('source_sha256'))
        mesh = _REGISTRATION_ATLAS_MESH_CACHE.get(cache_key)
        if mesh is None:
            full_mesh=store.mesh(project_id)
            from .cad_view_atlas import registration_atlas_mesh
            mesh=registration_atlas_mesh(full_mesh,max_triangles=60000)
            _REGISTRATION_ATLAS_MESH_CACHE.clear()
            _REGISTRATION_ATLAS_MESH_CACHE[cache_key]=mesh
        pages = []
        for row in frame.get('source_points', []):
            for page in row.get('pages', []) or []:
                if type(page) is int and page > 0 and page not in pages:
                    pages.append(page)
        detail = {'methods': [{'page': page} for page in pages[:2]], 'dependencies': []}
        source_images = _method_page_images(store, project, detail, model)
        # Keep the actual XYZ definition pages, not neighboring pages inferred
        # from ordering. The renderer verifies the active MSOP digest.
        source_images = source_images[:2]
        evidence_dir = store._dir(project_id) / 'automatic-learning' / 'registration-evidence'
        evidence_dir.mkdir(parents=True, exist_ok=True)
        xyz = [row['xyz'] for row in frame.get('source_points', [])]
        from .cad_view_atlas import render_registration_hypothesis_atlas
        atlas_images = []
        for hypothesis in hypotheses:
            support_names=set(hypothesis.get('support_point_names') or [])
            hypothesis_xyz=([row['xyz'] for row in frame.get('source_points', [])
                            if row.get('name') in support_names] if support_names else xyz)
            cache_key = hashlib.sha256(json.dumps([
                project.get('model', {}).get('source_sha256'), frame.get('program'),
                frame.get('coordinate_system'), hypothesis_xyz, hypothesis['rotation'],
                hypothesis['translation_mm'], hypothesis['hypothesis_id'],
            ], sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
            target = evidence_dir / f"registration-{cache_key}.png"
            if not target.is_file():
                render_registration_hypothesis_atlas(
                    mesh, hypothesis_xyz, hypothesis['rotation'], hypothesis['translation_mm'],
                    target, hypothesis['hypothesis_id'], panel_size=320)
            atlas_images.append(str(target))
        images = [*source_images, *atlas_images]
        return images[:4], {
            'status': 'attached' if images else 'no_visual_assets',
            'source_pages': pages[:2], 'cad_overlay_count': len(atlas_images),
            'image_order': ['MSOP source page(s)', 'candidate CAD overlays'],
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        # The numeric evidence path remains available. Record the omission so
        # the receipt makes clear why this comparison was text-only.
        return [], {'status': 'visual_evidence_generation_failed'}


def _local_model_registration_decisions(store, project_id, model, receipt, checkpoint=None):
    """Select a registration hypothesis with the configured local model.

    The geometric worker only supplies alternatives. It never authorizes one.
    The model's choice is source-pinned into the receipt and is required by the
    downstream marker candidate builder.
    """
    from .local_matching import _is_local_inference_model, _local_model_runtime_ready, _object
    if not _local_model_runtime_ready(model):
        raise RuntimeError('本地模型未就绪，不能决定 MSOP→CAD 配准')
    if not _is_local_inference_model(model):
        raise RuntimeError('MSOP→CAD 配准只允许本机本地模型决定')
    identity = model.info()
    identity_key = {key: identity.get(key) for key in ('model_alias','inference_backend','base_model_path','adapter_dir')}
    identity_key['decision_policy_sha256']=hashlib.sha256(
        inspect.getsource(_local_model_registration_decisions).encode('utf-8')).hexdigest()
    saved = receipt.get('local_model_registration') or {}
    if (saved.get('model_identity') == identity_key and saved.get('status') == 'complete'
            and len(saved.get('frames', [])) == len(receipt.get('frames', []))):
        return receipt
    # Persist each model decision as it is made. Registration is commonly a
    # multi-minute local inference job; losing every earlier frame because a
    # later request was interrupted made retries needlessly start from zero.
    resumable = saved.get('frames', []) if saved.get('model_identity') == identity_key else []
    resumable_by_index = {row.get('frame_index'): row for row in resumable
                          if type(row.get('frame_index')) is int}
    decisions=[]
    total_frames=len(receipt.get('frames', []))
    update_progress=getattr(store,'update_local_model_marking',None)

    def save_frame_decision(frame_index, decision):
        decision = dict(decision, frame_index=frame_index)
        if len(decisions)==frame_index:
            decisions.append(decision)
        elif len(decisions)>frame_index:
            decisions[frame_index]=decision
        else:
            raise ValueError('Registration checkpoint cannot skip an earlier frame')
        receipt['local_model_registration']=dict(
            status='running',model_identity=identity_key,frames=list(decisions),
            authority='local_model_decision',source_to_cad_registered=False,metrology_accepted=False)
        if checkpoint:
            checkpoint(receipt)

    for frame_index, frame in enumerate(receipt.get('frames', [])):
        previous=resumable_by_index.get(frame_index)
        if (previous and previous.get('program')==frame.get('program')
                and previous.get('coordinate_system')==frame.get('coordinate_system')
                and previous.get('status') in ('selected','abstained')):
            # The receipt pins CAD, MSOP, registration algorithm, model, and
            # decision policy. Reuse only a terminal decision for this exact
            # frame index and scope; transient inference failures are not saved.
            decisions.append(previous)
            if update_progress:
                update_progress(project_id,progress=dict(stage='registration_model_selection',done=frame_index+1,
                    total=total_frames,current_fai=None,
                    message=f"复用本地模型已完成的配准决定 {frame_index+1}/{total_frames}"))
            continue
        fit=frame.get('triangle_refinement') or {}
        option_rows=[dict(hypothesis_id=f'H{index:02d}',option=option)
                     for index,option in enumerate([fit.get('best'),*(fit.get('alternatives') or [])])
                     if isinstance(option,dict)]
        for family_index,family in enumerate(frame.get('feature_family_hypotheses') or []):
            family_fit=family.get('triangle_refinement') or {}
            for option_index,option in enumerate([family_fit.get('best'),*(family_fit.get('alternatives') or [])]):
                if isinstance(option,dict):
                    option_rows.append(dict(hypothesis_id=f'G{family_index:02d}_{option_index:02d}',
                        option=option,support_feature_group=family.get('feature_group'),
                        support_point_names=family.get('support_point_names') or [],
                        support_point_count=family.get('support_point_count')))
        option_rows=[row for row in option_rows
                     if isinstance(row['option'].get('rotation'),list)
                     and isinstance(row['option'].get('translation_mm'),list)]
        if not option_rows:
            save_frame_decision(frame_index,dict(program=frame.get('program'),coordinate_system=frame.get('coordinate_system'),
                                                  status='abstained',reason='no_rigid_transform_hypotheses'))
            continue
        source_rows=[dict(name=row.get('name'),xyz=row.get('xyz'),pages=row.get('pages'))
                     for row in frame.get('source_points', [])]
        if not source_rows:
            save_frame_decision(frame_index,dict(program=frame.get('program'),coordinate_system=frame.get('coordinate_system'),
                                                  status='abstained',reason='empty_source_point_group'))
            continue
        hypotheses=[]
        def compact_vector(values):
            return [round(float(value),4) for value in values] if isinstance(values,list) else values
        for option_row in option_rows:
            option=option_row['option'];hypothesis_id=option_row['hypothesis_id']
            mapped=option.get('cad_points_mm') or []
            distances=option.get('residuals_mm') or []
            support_names=set(option_row.get('support_point_names') or [])
            mapping_rows=([row for row in source_rows if row.get('name') in support_names]
                          if support_names else source_rows)
            point_mappings=[]
            for point_index,row in enumerate(mapping_rows):
                if point_index<len(mapped) and point_index<len(distances):
                    point_mappings.append([row.get('name'),compact_vector(row.get('xyz')),
                                           compact_vector(mapped[point_index]),
                                           round(float(distances[point_index]),4)])
            hypothesis=dict(
                hypothesis_id=hypothesis_id,
                rotation=[compact_vector(row) for row in option['rotation']],
                translation_mm=compact_vector(option['translation_mm']),
                fit_p95_residual_mm=option.get('fit_p95_residual_mm'),
                fit_max_residual_mm=option.get('fit_max_residual_mm',option.get('max_residual_mm')),
                p95_residual_mm=option.get('p95_residual_mm'),
                max_residual_mm=option.get('max_residual_mm'),
                held_out_max_mm=option.get('held_out_max_mm'),
                inlier_ratio=option.get('inlier_ratio'),
                initialization=option.get('initialization','rigid_surface_fit'),
                all_source_point_mappings=point_mappings if not support_names else [],
                support_point_mappings=point_mappings,
                support_feature_group=option_row.get('support_feature_group'),
                support_point_names=option_row.get('support_point_names'),
                support_point_count=option_row.get('support_point_count'))
            hypotheses.append(hypothesis)
        incomplete=[hypothesis['hypothesis_id'] for hypothesis in hypotheses
                    if not hypothesis.get('support_feature_group')
                    and len(hypothesis['all_source_point_mappings'])!=len(source_rows)
                    or any(not isinstance(row,list) or len(row)!=4
                           or not isinstance(row[2],list) or isinstance(row[3],bool)
                           or not isinstance(row[3],(int,float))
                           or any(isinstance(value,bool) or not isinstance(value,(int,float))
                                  or not math.isfinite(value)
                                  for vector in (row[1],row[2]) if isinstance(vector,list) for value in vector)
                           or not isinstance(row[1],list) or len(row[1])!=3
                           or not isinstance(row[2],list) or len(row[2])!=3
                           or not math.isfinite(float(row[3]))
                           for row in hypothesis['support_point_mappings'])]
        if incomplete:
            save_frame_decision(frame_index,dict(program=frame.get('program'),coordinate_system=frame.get('coordinate_system'),
                                                  status='abstained',reason='incomplete_hypothesis_correspondences',
                                                  source_point_count=len(source_rows),incomplete_hypotheses=incomplete,
                                                  unresolved_source_conflict_count=len(frame.get('conflicts',[]))))
            continue
        # Compare every full-source-point hypothesis in a local-model tournament.
        # A deterministic seed prevents the residual-sorted H00 from always
        # receiving the first position, while preserving reproducibility.
        seed_key=f"{frame.get('program')}|{frame.get('coordinate_system')}|{receipt.get('cad_sha256')}|{receipt.get('msop_sha256')}"
        tournament=list(range(len(hypotheses)))
        random.Random(seed_key).shuffle(tournament)
        previous_comparison=(previous if previous and previous.get('status')=='comparing'
                             and previous.get('program')==frame.get('program')
                             and previous.get('coordinate_system')==frame.get('coordinate_system') else None)
        challenges=tournament[1:] if len(tournament)>1 else [None]
        if (previous_comparison and previous_comparison.get('tournament_order')==tournament
                and previous_comparison.get('champion_index') in tournament
                and type(previous_comparison.get('next_comparison_index')) is int
                and 1<=previous_comparison['next_comparison_index']<=len(challenges)+1):
            champion=previous_comparison['champion_index']
            start_comparison=previous_comparison['next_comparison_index']
            evidence=list(previous_comparison.get('evidence') or [])
            confidence=previous_comparison.get('model_confidence') or 'low'
            unresolved_comparisons=list(previous_comparison.get('unresolved_comparisons') or [])
            decisive_comparison_count=int(previous_comparison.get('decisive_comparison_count') or 0)
        else:
            champion=tournament[0];start_comparison=1;evidence=[];confidence='low'
            unresolved_comparisons=[];decisive_comparison_count=0
        raw='';tokens=0;truncated=False
        failed=None
        for comparison_index in range(start_comparison,len(challenges)+1):
            challenger=challenges[comparison_index-1]
            if update_progress:
                update_progress(project_id,progress=dict(stage='registration_model_selection',done=frame_index,
                    total=total_frames,current_fai=None,comparison_done=comparison_index,
                    comparison_total=len(challenges),
                    message=(f"本地模型比较 {frame.get('program')} / {frame.get('coordinate_system')} "
                             f"配准假设 {comparison_index}/{len(challenges)}；完整点组逐点核对")))
            pair=[hypotheses[champion]]
            if challenger is not None:pair.append(hypotheses[challenger])
            images, visual_evidence = _registration_visual_evidence(
                store, project_id, model, frame, pair)
            payload=dict(program=frame.get('program'),coordinate_system=frame.get('coordinate_system'),
                         source_point_count=len(source_rows),
                         full_source_points=[[row.get('name'),compact_vector(row.get('xyz'))]
                                             for row in source_rows],
                         source_conflicts=frame.get('conflicts',[]),
                         visual_evidence=visual_evidence,
                         hypotheses=pair)
            question=(
                ('你是此 MSOP 坐标系的配准决策模型。当前只有一个完整刚体变换假设，请独立判断它是否可信；'
                 '若证据不足、源点对应有误或仍有歧义，必须 ABSTAIN。' if challenger is None else
                 '你是此 MSOP 坐标系的配准决策模型。比较下面两个完整刚体变换假设，选择更符合原始命名 XYZ 点与 CAD 表面对应关系的一项。')
                +
                '几何程序没有决定胜者；它只生成候选、逐点对应和残差。普通候选的 all_source_point_mappings、特征族候选的 support_point_mappings 每行依次为 [点名, MSOP XYZ(mm), CAD表面 XYZ(mm), 点面距离(mm)]；'
                '特征族候选的 supporting feature group 是候选生成时用于抵抗混合特征离群点的子集，不代表坐标系只适用于该特征；最终仍需你结合原始图纸、完整MSOP点组和CAD叠图判断。'
                '同时检查可用无冲突点组和旋转平移的一致性。source_conflicts 是同名点的替代坐标记录，已排除在本次变换拟合外；不得把它们当成另一组独立测量点。'
                '图像只作补充证据；visual_evidence.status 表示图像不可用时，不得单凭缺少图像弃答。'
                '比较候选时以刚体不变量为准：同一刚体旋转可以让任意单个坐标轴的符号反向，这属于合法旋转，不能单独作为弃答理由；'
                '核对旋转矩阵为正规右手刚体变换、源点两两距离在变换前后保持、全组同名对应、inlier_ratio、p95/max/held-out 残差及是否存在局部系统错配。'
                '0.1 mm 是严格测量/常规几何候选门槛，不是显示坐标假设选择时必须满足的上限；候选仍须由你根据原图、可见实体、点组形状与多个候选间的相对证据判断。'
                '若某一候选由清楚的测量特征族支撑且整体空间关系更合理，可选择为显示坐标假设，即使其他特征族使全局残差变大；不得因此认为离群特征也已正确映射。'
                '下游仍会逐个 FAI 独立检查全部测量点、面距离、特征组共面性和相对形状，任何点未通过都不会获标。'
                '坐标系选择本身不是计量认证，不得据此声称 source_to_cad_registered 或 metrology_accepted。'
                '存在冲突点时，只能决定这些无冲突点支持的刚体变换；具体 FAI 是否采用某个冲突点版本会在后续独立的本地模型步骤中判定。'
                '若无冲突点不足以支持变换、两种变换均不可信、证据相互矛盾或仍有歧义，必须 ABSTAIN。'
                '若附件含图，前面的图是原始 MSOP 点位页，后面的图按 hypothesis_id 顺序显示候选变换投影到 CAD 的六向视图；'
                '红点编号按 source point mappings 中列出的点顺序对应；特征族候选的红点只绘制该特征族，普通候选绘制完整坐标系点组。请把图中点是否落在合理实体/测量面与数值残差一起核对；'
                '叠图只用于检查空间一致性，不能替代 MSOP 原始标注，也不得把投影点贴到错误可见面上。'
                '不可仅凭编号、排列顺序或最小平均残差选择。只可返回本次 hypotheses 的 hypothesis_id。'
                '只输出单行 JSON，不要写推理过程；evidence 限 120 个字符。字段：decision（SELECT_TRANSFORM 或 ABSTAIN）、'
                'hypothesis_id、program、coordinate_system、source_point_count、evidence、confidence（low/medium/high）。'
            )
            context=(f"Drawing {receipt.get('drawing_no')} | CAD sha256 {receipt.get('cad_sha256')} | "
                     f"MSOP sha256 {receipt.get('msop_sha256')}\n"+json.dumps(payload,ensure_ascii=False,separators=(',',':')))
            if len(context)+len(question)>48000:
                failed=dict(reason='registration_evidence_exceeds_context_budget');break
            # Talker chat, visual fallback, and registration share one local
            # model instance. Serialize each decode so concurrent requests do
            # not race CUDA buffers/KV cache or issue overlapping calls to the
            # local Strata endpoint.
            inference_lock=getattr(model,'_lock',None)
            with inference_lock if inference_lock is not None else nullcontext():
                messages=[{'role':'user','content':question,'attachments':[{'filename':'Complete MSOP-to-CAD hypothesis pair',
                                                                            'parsed_text':context,'images':images}]}]
                try:
                    raw,truncated,tokens=model._generate_once(
                        messages,max_context_chars=len(context)+len(question)+512,
                        max_new_tokens=768,max_images=len(images))
                except Exception as visual_error:
                    if not images:
                        raise
                    # Image inference may be locally configured but unable to
                    # allocate alongside the active Strata model on one GPU.
                    # Retry with complete numeric/MSOP text evidence; the local
                    # model must still explicitly select or abstain.
                    visual_evidence=dict(visual_evidence,
                        status='multimodal_inference_failed',
                        error=f'{type(visual_error).__name__}: {visual_error}'[:400])
                    _REGISTRATION_VISION_DISABLED_REASON=visual_evidence['error']
                    payload['visual_evidence']=visual_evidence
                    context=(f"Drawing {receipt.get('drawing_no')} | CAD sha256 {receipt.get('cad_sha256')} | "
                             f"MSOP sha256 {receipt.get('msop_sha256')}\n"+
                             json.dumps(payload,ensure_ascii=False,separators=(',',':')))
                    if len(context)+len(question)>48000:
                        failed=dict(reason='registration_evidence_exceeds_context_budget');break
                    messages[0]['attachments'][0]['parsed_text']=context
                    messages[0]['attachments'][0]['images']=[]
                    raw,truncated,tokens=model._generate_once(
                        messages,max_context_chars=len(context)+len(question)+512,
                        max_new_tokens=768,max_images=0)
            result=_object(raw) or {};selected_id=result.get('hypothesis_id')
            valid=(not truncated and result.get('decision')=='SELECT_TRANSFORM'
                   and result.get('program')==frame.get('program')
                   and result.get('coordinate_system')==frame.get('coordinate_system')
                   and result.get('source_point_count')==len(source_rows)
                   and result.get('confidence') in ('low','medium','high')
                   and selected_id in {hypothesis['hypothesis_id'] for hypothesis in pair})
            if not valid:
                if (not truncated and result.get('decision')=='ABSTAIN' and len(pair)>1):
                    # A tie between two weak candidates does not prove that
                    # every other transform is wrong. Keep the local model's
                    # abstention attached to this pair, then let it inspect the
                    # remaining alternatives. No algorithmic winner is added.
                    unresolved_comparisons.append(dict(
                        compared_hypotheses=[h['hypothesis_id'] for h in pair],
                        evidence=str(result.get('evidence') or raw)[:800]))
                    evidence.append(result.get('evidence') or raw[:800])
                    save_frame_decision(frame_index,dict(
                        program=frame.get('program'),coordinate_system=frame.get('coordinate_system'),
                        status='comparing',champion_index=champion,tournament_order=tournament,
                        next_comparison_index=comparison_index+1,model_confidence=confidence,evidence=evidence,
                        unresolved_comparisons=unresolved_comparisons,
                        decisive_comparison_count=decisive_comparison_count,
                        completed_comparisons=comparison_index,compared_hypothesis_count=len(hypotheses),
                        visual_evidence=visual_evidence))
                    continue
                failed=dict(reason='model_abstained_or_invalid',evidence=str(result.get('evidence') or raw)[:800],
                            truncated=truncated,compared_hypotheses=[h['hypothesis_id'] for h in pair])
                break
            champion=next(index for index in (champion,challenger)
                          if hypotheses[index]['hypothesis_id']==selected_id)
            decisive_comparison_count+=1
            confidence=result['confidence'];evidence.append(result.get('evidence'))
            save_frame_decision(frame_index,dict(
                program=frame.get('program'),coordinate_system=frame.get('coordinate_system'),
                status='comparing',champion_index=champion,tournament_order=tournament,
                next_comparison_index=comparison_index+1,model_confidence=confidence,evidence=evidence,
                unresolved_comparisons=unresolved_comparisons,
                decisive_comparison_count=decisive_comparison_count,
                completed_comparisons=comparison_index,compared_hypothesis_count=len(hypotheses),
                visual_evidence=visual_evidence))
        if failed:
            save_frame_decision(frame_index,dict(program=frame.get('program'),coordinate_system=frame.get('coordinate_system'),
                                                  status='abstained',source_point_count=len(source_rows),**failed))
            continue
        if decisive_comparison_count==0:
            save_frame_decision(frame_index,dict(program=frame.get('program'),
                coordinate_system=frame.get('coordinate_system'),status='abstained',
                source_point_count=len(source_rows),reason='all_registration_comparisons_abstained',
                evidence=[row.get('evidence') for row in unresolved_comparisons],
                unresolved_comparisons=unresolved_comparisons,
                unresolved_source_conflict_count=len(frame.get('conflicts',[]))))
            continue
        selected_id=hypotheses[champion]['hypothesis_id']
        selected_hypothesis=hypotheses[champion]
        selected_transform={key:selected_hypothesis.get(key) for key in (
            'rotation','translation_mm','p95_residual_mm','max_residual_mm',
            'held_out_max_mm','inlier_ratio','initialization','support_feature_group',
            'support_point_names','support_point_count')}
        selected_index=int(selected_id[1:]) if selected_id.startswith('H') else None
        if selected_index is not None:
            fit['model_selected_hypothesis_id']=selected_id
        save_frame_decision(frame_index,dict(program=frame.get('program'),coordinate_system=frame.get('coordinate_system'),
                                              status='selected',hypothesis_id=selected_id,hypothesis_index=selected_index,
                                              selected_transform=selected_transform,
                                              model_confidence=confidence,evidence=evidence,
                                              model_raw=raw[:1200],generated_tokens=tokens,
                                              compared_hypothesis_count=len(hypotheses),
                                              unresolved_source_conflict_count=len(frame.get('conflicts',[])),
                                              visual_evidence=visual_evidence,
                                              unresolved_comparisons=unresolved_comparisons,
                                              decisive_comparison_count=decisive_comparison_count))
    receipt['local_model_registration']=dict(status='complete',model_identity=identity_key,frames=decisions,
        authority='local_model_decision',source_to_cad_registered=False,metrology_accepted=False)
    if checkpoint:
        checkpoint(receipt)
    return receipt


def _restore_selected_candidate_precision(receipt):
    """Restore the exact generated transform identified by the local model.

    The model selects a hypothesis ID from rounded evidence. Reuse that ID
    to recover the source candidate's full-precision transform, avoiding
    an invalid SO(3) matrix from four-decimal model transcription.
    """
    frames={(frame.get('program'),frame.get('coordinate_system')):frame
            for frame in receipt.get('frames',[])}
    registration=receipt.get('local_model_registration') or {}
    for decision in registration.get('frames',[]):
        if decision.get('status')!='selected':
            continue
        scope=(decision.get('program'),decision.get('coordinate_system'))
        frame=frames.get(scope) or {}
        hypothesis_id=decision.get('hypothesis_id')
        option=None
        if isinstance(hypothesis_id,str) and hypothesis_id.startswith('H'):
            try:index=int(hypothesis_id[1:])
            except ValueError:index=-1
            fit=frame.get('triangle_refinement') or {}
            options=[fit.get('best'),*(fit.get('alternatives') or [])]
            if 0<=index<len(options):option=options[index]
        elif isinstance(hypothesis_id,str) and hypothesis_id.startswith('G'):
            try:
                family_id,option_id=hypothesis_id[1:].split('_',1)
                family_index=int(family_id);option_index=int(option_id)
            except ValueError:
                family_index=option_index=-1
            families=frame.get('feature_family_hypotheses') or []
            if 0<=family_index<len(families):
                fit=families[family_index].get('triangle_refinement') or {}
                options=[fit.get('best'),*(fit.get('alternatives') or [])]
                if 0<=option_index<len(options):option=options[option_index]
        if isinstance(option,dict) and isinstance(option.get('rotation'),list):
            decision['selected_transform']={key:option.get(key) for key in (
                'rotation','translation_mm','p95_residual_mm','max_residual_mm',
                'held_out_max_mm','inlier_ratio','initialization','support_feature_group',
                'support_point_names','support_point_count')}
    return receipt


def ensure_registration(store, project_id, model=None):
    project=store.get(project_id)
    msop=store.active_msop(project) or {}
    cad=(project.get('model') or {}).get('source_sha256')
    root=Path(__file__).resolve().parents[1]
    algorithm_sha256=registration_algorithm_sha256(root)
    output=store._dir(project_id)/'automatic-learning'/'registration.json'
    def checkpoint(receipt):
        output.parent.mkdir(parents=True,exist_ok=True)
        temp=output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(output)
    def current_receipt():
        if not output.is_file():
            return None
        digest=hashlib.sha256(output.read_bytes()).hexdigest()
        invalidate=getattr(store,'invalidate_stale_local_model_annotations',None)
        if invalidate:
            invalidate(project_id,digest)
        return output

    if output.is_file():
        receipt=json.loads(output.read_text())
        if (receipt.get('cad_sha256')==cad and receipt.get('msop_sha256')==msop.get('sha256')
                and receipt.get('algorithm_sha256')==algorithm_sha256):
            receipt=_local_model_registration_decisions(store,project_id,model,receipt,checkpoint=checkpoint)
            receipt=_restore_selected_candidate_precision(receipt)
            output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
            return current_receipt()
    if not cad or not msop.get('sha256'):
        raise ValueError('Automatic registration requires active CAD and MSOP sources')
    output.parent.mkdir(parents=True,exist_ok=True)
    progress=output.with_name('registration-progress.json')
    progress.unlink(missing_ok=True)
    with output.with_name('registration.log').open('w') as log:
        with subprocess.Popen([sys.executable,str(root/'scripts/auto_register_msop.py'),
             '--project',project_id,'--state-root',str(store.root.resolve()),
             '--output',str(output.resolve())],cwd=root,stdout=log,stderr=subprocess.STDOUT) as worker:
            while worker.poll() is None:
                status=dict(done=0,message='自动构建 MSOP 坐标系与 CAD 配准候选')
                if progress.is_file():
                    try:status.update(json.loads(progress.read_text()))
                    except (OSError,json.JSONDecodeError):pass
                store.update_local_model_marking(project_id,progress=dict(status,stage='registration'))
                time.sleep(2)
            if worker.returncode:
                raise RuntimeError('Automatic registration failed; see automatic-learning/registration.log')
    receipt=json.loads(output.read_text())
    if (receipt.get('cad_sha256')!=cad or receipt.get('msop_sha256')!=msop.get('sha256')
            or receipt.get('algorithm_sha256')!=algorithm_sha256):
        raise ValueError('Generated registration has stale source or algorithm hashes')
    receipt=_local_model_registration_decisions(store,project_id,model,receipt,checkpoint=checkpoint)
    receipt=_restore_selected_candidate_precision(receipt)
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    return current_receipt()
