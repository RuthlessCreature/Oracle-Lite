"""Evidence-only CAD body catalog and local-model physical-part selection.

No body name is inferred by code. Numeric IDs and geometry are shown to the
native model together with the source drawing before any measurement placement.
"""
from __future__ import annotations
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .cad_view_atlas import _valid_triangles,_view_projection,VIEWS


def cad_parts(mesh):
    grouped={}
    for face in mesh.get('faces',[]):
        if face.get('occt_face_valid') is False:continue
        body=face.get('body')
        if body is None:continue
        grouped.setdefault(body,[]).append(face)
    parts=[]
    for body,faces in sorted(grouped.items(),key=lambda row:str(row[0])):
        arrays=[np.asarray(face['positions'],float).reshape(-1,3) for face in faces if face.get('positions')]
        if not arrays:continue
        xyz=np.concatenate(arrays)
        if not len(xyz) or not np.isfinite(xyz).all():continue
        parts.append({'part_id':f'B{body}','body':body,'face_ids':[f['id'] for f in faces],
            'prototype_body':faces[0].get('prototype_body'),
            'face_count':len(faces),'surface_counts':dict(Counter(face.get('surface','unknown') for face in faces)),
            'bbox_mm':{'min':xyz.min(axis=0).tolist(),'max':xyz.max(axis=0).tolist()},
            'size_mm':np.ptp(xyz,axis=0).tolist(),'center_mm':((xyz.min(axis=0)+xyz.max(axis=0))/2).tolist()})
    return parts


def render_part_catalog(mesh,parts,directory):
    """Isolated original CAD bodies; no invented geometry or semantic labels."""
    output=[];faces={face['id']:face for face in mesh['faces']}
    try:font=ImageFont.truetype('DejaVuSans.ttf',13)
    except OSError:font=ImageFont.load_default()
    for offset in range(0,len(parts),4):
        batch=parts[offset:offset+4];key=hashlib.sha256(json.dumps(batch,sort_keys=True).encode()+mesh['source_sha256'].encode()+Path(__file__).read_bytes()).hexdigest()[:20]
        target=Path(directory)/f'cad-part-catalog-{key}.png'
        if not target.is_file():
            tile=220;row_height=248;image=Image.new('RGB',(tile*3,row_height*len(batch)),'white');draw=ImageDraw.Draw(image)
            for row,part in enumerate(batch):
                body_mesh={'faces':[faces[fid] for fid in part['face_ids']]};triangles,_=_valid_triangles(body_mesh,max_triangles=5000)
                draw.text((8,row*row_height+5),f"{part['part_id']} / original CAD body / size mm "+' x '.join(f'{x:.2f}' for x in part['size_mm']),fill='#233b35',font=font)
                for column,view in enumerate(('+Z','+X','+Y')):
                    normal,right,up,cr,cu,scale,_=_view_projection(triangles,view,tile,24)
                    x=(triangles@right-cr)*scale+tile/2+column*tile
                    y=(cu-triangles@up)*scale+tile/2+row*row_height+26
                    centers=triangles.mean(axis=1);normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                    shade=np.abs(normals@normal)/np.maximum(np.linalg.norm(normals,axis=1),1e-12)
                    for index in np.argsort(centers@normal):
                        intensity=int(100+90*(1-np.clip(shade[index],0,1)))
                        draw.polygon([(float(x[index,j]),float(y[index,j])) for j in range(3)],fill=(intensity,intensity+10,intensity+13))
                    draw.text((column*tile+5,row*row_height+28),view,fill='#35664e',font=font)
            target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.png.tmp');image.save(temp,format='PNG');temp.replace(target)
        output.append(str(target))
    return output


def _persist_part_selection(store,project,item,mesh,audit):
    audit={**audit,'fai':item.get('number'),'cad_sha256':mesh.get('source_sha256'),
        'document_sha256':next((doc.get('sha256') for doc in project.get('documents',[])
            if doc.get('id')==item.get('document_id')),None),
        'does_not_authorize_measurement_anchors':True}
    directory=store._dir(project['id'])/'automatic-learning'/'part-selections'
    directory.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(audit,ensure_ascii=False,indent=2).encode()
    receipt=directory/f'{hashlib.sha256(encoded).hexdigest()}.json'
    temporary=receipt.with_suffix('.json.tmp');temporary.write_bytes(encoded);temporary.replace(receipt)
    return {**audit,'receipt':str(receipt)}


def _candidate_geometry_support(mesh,candidate):
    """Expose only direct-source XYZ display hypotheses as non-authoritative body hints."""
    if not isinstance(candidate,dict):return []
    evidence=candidate.get('evidence') or {}
    groups=candidate.get('required_marker_groups') or evidence.get('required_marker_groups') or []
    direct_names={name for group in groups if isinstance(group,dict)
        for name in group.get('source_point_names',[]) if isinstance(name,str)}
    if not direct_names:return []
    face_by_id={face.get('id'):face for face in mesh.get('faces',[])}
    rows=[];seen=set()
    for row in evidence.get('point_correspondences') or []:
        if not isinstance(row,dict) or row.get('name') not in direct_names or row['name'] in seen:
            continue
        face=face_by_id.get(row.get('face_id'))
        if face is None:continue
        seen.add(row['name'])
        rows.append({key:row.get(key) for key in
            ('name','program','coordinate_system','source_xyz','transformed_xyz','position',
             'face_id','residual_mm') if key in row})
        rows[-1]['body_id']=f"B{face.get('body')}"
        rows[-1]['instance_path']=face.get('instance_path',[])
        rows[-1]['authority']='prior_local_model_display_hypothesis_only'
    return rows


def render_screening_montage(paths,output):
    """Combine every screened original-body sheet for later entity review."""
    paths=[Path(path) for path in dict.fromkeys(paths) if Path(path).is_file()]
    if not paths:return None
    target=Path(output)
    if target.is_file():return str(target)
    cell=(680,1030);columns=2;rows=(len(paths)+columns-1)//columns
    canvas=Image.new('RGB',(columns*cell[0],rows*cell[1]),'white');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:font=ImageFont.load_default()
    for index,path in enumerate(paths):
        with Image.open(path) as source:
            tile=source.convert('RGB');tile.thumbnail((cell[0]-16,cell[1]-42))
        x=(index%columns)*cell[0];y=(index//columns)*cell[1]
        draw.text((x+8,y+6),f'Original CAD body screening sheet {index+1}/{len(paths)}',fill='#233b35',font=font)
        canvas.paste(tile,(x+(cell[0]-tile.width)//2,y+32))
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix('.png.tmp')
    canvas.save(temporary,format='PNG');temporary.replace(target)
    return str(target)


def select_physical_parts(store,project,item,detail,model,mesh,source_images,*,geometry_hypothesis=None):
    from .local_matching import _object,_complete_visual_context_budget,_declared_marker_manifest,_fai_number
    parts=cad_parts(mesh);all_parts=parts
    geometry_support=_candidate_geometry_support(mesh,geometry_hypothesis)
    expected_source_names=sorted({row['name'] for row in geometry_support
        if isinstance(row.get('name'),str) and row['name']})
    hypothesis=geometry_hypothesis if isinstance(geometry_hypothesis,dict) else {}
    hypothesis_evidence=hypothesis.get('evidence') or {}
    for group in (hypothesis.get('required_marker_groups') or
                  hypothesis_evidence.get('required_marker_groups') or []):
        if isinstance(group,dict):
            expected_source_names.extend(name for name in group.get('source_point_names',[])
                if isinstance(name,str) and name)
    for group in (_declared_marker_manifest(detail.get('methods',[])) or []):
        expected_source_names.extend(name for name in group.get('source_point_names',[])
            if isinstance(name,str) and name)
    expected_source_names=sorted(set(expected_source_names))
    if not parts:return None,_persist_part_selection(store,project,item,mesh,
        {'status':'unavailable','reason':'CAD_body_identity_missing'})
    pages=render_part_catalog(mesh,parts,store._dir(project['id'])/'automatic-learning'/'part-catalog')
    images=list(dict.fromkeys([*source_images,*pages]))
    screening=[]
    if len(images)>16:
        candidates=[]
        for start in range(0,len(parts),24):
            batch=parts[start:start+24]
            batch_pages=render_part_catalog(mesh,batch,store._dir(project['id'])/'automatic-learning'/'part-catalog')
            batch_images=list(dict.fromkeys([*source_images,*batch_pages]))
            if len(batch_images)>16:
                return None,_persist_part_selection(store,project,item,mesh,{'status':'unavailable',
                    'reason':'complete_part_batch_image_budget_exceeded','screening':screening})
            payload={'fai':item['number'],'methods':detail.get('methods',[]),
                'direct_source_point_CAD_display_hypotheses':geometry_support,
                'visible_parts':[{k:v for k,v in part.items() if k!='face_ids'} for part in batch]}
            screen_question=('你是本地 CAD 零件候选筛选模型，只筛选当前一批原始body，不标点。'
                '原始MSOP和图纸指明本FAI的特征；后续图是当前批次body三视图。'
                '本阶段只识别承载被测特征的主体零件：例如螺母、导轨、挡片、窗扣本体。'
                '若特征或尺寸描述为A TO B、A-to-B、between A/B、从A到B、两面/两零件间距等关系，A和B是两个不同测量端；'
                '把当前批次中分别承载两端的主体都列为候选，不得只选一个端点对应的零件。'
                '即使FAI描述焊点，也不要选择小焊点body或凑出相同数量；焊点与主体的关联留给下一阶段。'
                '如附有direct_source_point_CAD_display_hypotheses，它只表示此前本地模型的显示假设，不能当真值或注册结果；'
                '可用其变换后点位与B编号的空间关系排除明显不相容主体，但必须服从本批原始body图像和图纸特征。'
                '请把形状和装配位置可能对应主体的所有B编号列为候选；重复主体实例全部保留。'
                '没有匹配候选时返回空数组。本批不能证明其他未展示body是否有关，也不能授权任何测量位置。'
                '只输出JSON：decision=SCREEN_PARTS,fai,part_ids(当前批次B编号数组，可空),evidence(简短依据)。')
            messages=[{'role':'user','content':screen_question,'attachments':[{
                'filename':'Original sources and one complete CAD body batch','parsed_text':json.dumps(payload,ensure_ascii=False),
                'images':batch_images}]}]
            raw,cut,tokens=model._generate_once(messages,max_context_chars=_complete_visual_context_budget(messages),
                max_new_tokens=1024,max_images=len(batch_images),temperature=0.)
            response=_object(raw) or {};ids=response.get('part_ids');valid_ids={part['part_id'] for part in batch}
            valid=(not cut and response.get('decision')=='SCREEN_PARTS' and _fai_number(response.get('fai'))==item['number']
                and isinstance(ids,list) and all(isinstance(value,str) and value in valid_ids for value in ids)
                and len(ids)==len(set(ids)) and bool(response.get('evidence')))
            screening.append({'visible_part_ids':sorted(valid_ids),'raw_output':raw,'truncated':cut,
                'valid':bool(valid),'catalog_images':batch_pages})
            # One malformed batch must not discard valid candidates from
            # earlier batches or prevent the independent full-catalog review.
            # Keep the bad response in the receipt, but never use its IDs.
            if not valid:continue
            candidates.extend(part for part in batch if part['part_id'] in ids)
        # If every batch abstains or returns malformed IDs, the global model
        # still gets the complete source-linked body table and may recover a
        # shortlist. No batch output itself authorizes a body.
        parts=candidates if candidates else all_parts
        pages=render_part_catalog(mesh,parts,store._dir(project['id'])/'automatic-learning'/'part-catalog')
        images=list(dict.fromkeys([*source_images,*pages]))
        if screening:
            payload={'fai':item['number'],'methods':detail.get('methods',[]),
                'direct_source_point_CAD_display_hypotheses':geometry_support,
                'all_original_bodies':[{k:v for k,v in part.items() if k!='face_ids'} for part in all_parts]}
            question=('你是本地 CAD 主体零件全局复核模型。此前分批候选过宽，尚未授权任何实体。'
                '请结合原图和完整原始body尺寸、位置、原型编号，选择承载本FAI测量特征的所有主体零件。'
                '若原文描述A TO B、A-to-B、between A/B或从A到B的距离，这是跨实体测量：必须在part_ids中同时保留两端的宿主零件，'
                '并在evidence中分别说明A端与B端对应哪些body；不能因一个测点有XYZ或更容易识别就只选单侧。'
                '如果有direct_source_point_CAD_display_hypotheses，它们是未注册的先前显示假设，只作空间相容性提示；'
                '只能使用与本FAI直接测量点名完全一致的行，不能把相似名、plane点或支撑点当成直接测量点。'
                '这是全局比较，不按每批凑数量。对于8 SPOTS PER NUT，应先找NUT主体，不是选8个小SPOTS；'
                '比较全部尺寸、surface_counts原始曲面拓扑与装配位置：有孔主体与简单小凸台应结合柱面和面数区分，不得单凭数量选择。'
                '勿将小焊点、薄面或整个大框架代替指定主体。主体候选的图像将在下一步独立验证。'
                '可以修正分批筛选的遗漏；只能选已有B编号，证据不足则ABSTAIN。'
                '只输出JSON：decision=SELECT_HOST_PARTS(或ABSTAIN),fai,part_ids,evidence。')
            messages=[{'role':'user','content':question,'attachments':[{
                'filename':'Original sources and complete global CAD body geometry table',
                'parsed_text':json.dumps(payload,ensure_ascii=False),'images':list(source_images)}]}]
            raw,cut,tokens=model._generate_once(messages,max_context_chars=_complete_visual_context_budget(messages),
                max_new_tokens=1024,max_images=len(source_images),temperature=0.)
            answer=_object(raw) or {};ids=answer.get('part_ids');lookup={p['part_id']:p for p in all_parts}
            valid=(not cut and answer.get('decision')=='SELECT_HOST_PARTS' and _fai_number(answer.get('fai'))==item['number']
                and isinstance(ids,list) and ids and all(isinstance(value,str) and value in lookup for value in ids)
                and len(ids)==len(set(ids)) and bool(answer.get('evidence')))
            screening.append({'stage':'global_host_comparison','raw_output':raw,'truncated':cut,'valid':bool(valid),
                'compared_part_ids':list(lookup)})
            if not valid:return None,_persist_part_selection(store,project,item,mesh,{'status':'unconfirmed',
                'reason':'global_host_identity_unconfirmed','screening':screening})
            parts=[lookup[value] for value in ids]
            pages=render_part_catalog(mesh,parts,store._dir(project['id'])/'automatic-learning'/'part-catalog')
            images=list(dict.fromkeys([*source_images,*pages]))
            if len(images)>16:return None,_persist_part_selection(store,project,item,mesh,{'status':'unavailable',
                'reason':'selected_part_comparison_image_budget_exceeded','screening':screening})
    # Face arrays are not a language-model input. Every numeric part keeps its
    # assembly-space bbox and prototype identity, including repeated bodies.
    context={'fai':item['number'],'methods':detail.get('methods',[]),
        'direct_source_point_CAD_display_hypotheses':geometry_support,
        'parts':[{k:v for k,v in part.items() if k!='face_ids'} for part in parts]}
    question=('你是本地 CAD 实体对应模型，本轮只选择当前FAI涉及的物理零部件，不选择测量点。'
        '前三类图为原始MSOP与图纸，其后的contact sheets为分离展示的原始CAD body三视图。B编号不含语义，必须自己结合形状、装配位置和引线识别。'
        '若方法或特征名称表示A TO B、A-to-B、between A/B、从A到B的距离/间隙，测量跨越两个物理端点；'
        '必须选择两端各自的宿主body，并在结构化 source_point_host_assignments 中逐个点名写出每个MSOP源点归属的B编号；单侧body不足以描述该测量。'
        '对离散测头接触点，一个源点通常只属于一个精确宿主body；不能因其配对端点位于另一面，就把两面的所有body都填给每个源点。'
        '当MSOP把同一X/Y位置的点按Z方向成对测距时，结合点名、测量公式和剖面图分别识别两个接触面；'
        '将每个端点分配给其实际接触面对应的body，而不是把整个候选body集合复制到每个点。'
        '只有原文明确显示测点位于多零件接缝/焊接交界时，才允许一个源点映射多个body，并须给出该点的交界证据。'
        '结合零件包围盒的装配坐标确认左右/上下，不能把每张缩放后的零件缩略图当成同一尺寸。'
        '若提供direct_source_point_CAD_display_hypotheses，只能把其中完全同名的直接测点作为空间相容性提示；这些点位未经注册，不是正确答案。'
        '选择当前原文特征所依附的主体零件body，例如螺母、导轨、挡片、窗扣；先识别主体，不要把小焊点body当成主体。'
        '包括PER NUT/PER FANG/PER SIDE的所有重复主体实例；不能只选择最容易看到的一个实例，也不能凭数量相等判定身份。'
        '不得用整个框架body代替原文指定螺母/导轨/挡片，不能因邻接而把无关零件算进来。没有充分对应证据时ABSTAIN。'
        '只输出JSON：decision(SELECT_PARTS/ABSTAIN),fai,part_ids(明确B编号数组),'
        'source_point_host_assignments(数组，每项含source_point_names原文点名数组、part_ids对应宿主B编号数组、evidence；'
        f'必须对以下每个直接测量点名恰好覆盖一次：{json.dumps(expected_source_names,ensure_ascii=False)}。'
        '跨零件距离的两个端点必须分别映射，禁止把不同端点全部映射到同一零件；没有可确认归属就ABSTAIN。),'
        'source_feature(原文特征名),evidence(一句说明引线/形状/装配位置依据),confidence(low/medium/high)。')
    messages=[{'role':'user','content':question,'attachments':[{'filename':'Original sources and complete numbered CAD bodies',
        'parsed_text':json.dumps(context,ensure_ascii=False),'images':images}]}]
    raw,truncated,tokens=model._generate_once(messages,max_context_chars=_complete_visual_context_budget(messages),max_new_tokens=1024,max_images=len(images),temperature=0.)
    parsed=_object(raw) or {};ids=parsed.get('part_ids');lookup={part['part_id']:part for part in parts}

    def validated_host_map(response):
        ids=response.get('part_ids');assignments=response.get('source_point_host_assignments')
        name_to_parts={};host_valid=True
        if expected_source_names:
            if not isinstance(assignments,list) or not assignments:host_valid=False
            else:
                for assignment in assignments:
                    if not isinstance(assignment,dict):host_valid=False;continue
                    names=assignment.get('source_point_names');host_ids=assignment.get('part_ids')
                    if (not isinstance(names,list) or not names or
                        any(not isinstance(name,str) or name not in expected_source_names for name in names) or
                        len(names)!=len(set(names)) or not isinstance(host_ids,list) or not host_ids or
                        any(not isinstance(value,str) or value not in (ids or []) for value in host_ids) or
                        len(host_ids)!=len(set(host_ids)) or not isinstance(assignment.get('evidence'),str) or
                        not assignment['evidence'].strip()):
                        host_valid=False;continue
                    for name in names:
                        if name in name_to_parts:host_valid=False
                        name_to_parts[name]=list(host_ids)
                if set(name_to_parts)!=set(expected_source_names):host_valid=False
                if any(not name_to_parts.get(name) for name in expected_source_names):host_valid=False
        return host_valid,name_to_parts

    host_assignments=parsed.get('source_point_host_assignments')
    host_assignments_valid,name_to_host_parts=validated_host_map(parsed)
    host_attempts=[{'raw_output':raw,'truncated':bool(truncated),'tokens':tokens,
        'host_map_valid':host_assignments_valid}]
    if expected_source_names and not host_assignments_valid and not truncated:
        supplied=sorted({name for row in host_assignments or [] if isinstance(row,dict)
            for name in row.get('source_point_names',[]) if isinstance(name,str)})
        correction=(question+'\n上一轮的逐点宿主映射没有通过源点清单校验。'
            f'仅以下 MSOP 测量点名是有效键：{json.dumps(expected_source_names,ensure_ascii=False)}；'
            f'上一轮出现的名称：{json.dumps(supplied,ensure_ascii=False)}。'
            '逐字复制有效源点名，不能用 Location Definitions 表中的相似名称替换方法页 Measure 点名；'
            '每个有效源点恰好出现一次，重新输出完整 JSON 和完整宿主关系。')
        retry_messages=[{'role':'user','content':correction,'attachments':messages[0]['attachments']}]
        try:
            retry_budget=_complete_visual_context_budget(retry_messages)
            retry_raw,retry_truncated,retry_tokens=model._generate_once(retry_messages,
                max_context_chars=retry_budget,max_new_tokens=1024,max_images=len(images),temperature=0.)
            host_attempts.append({'raw_output':retry_raw,'truncated':bool(retry_truncated),
                'tokens':retry_tokens,'host_map_valid':False})
            if not retry_truncated:
                retry_parsed=_object(retry_raw) or {};retry_valid,retry_map=validated_host_map(retry_parsed)
                host_attempts[-1]['host_map_valid']=retry_valid
                if retry_valid:
                    parsed=retry_parsed;raw=retry_raw;truncated=retry_truncated
                    tokens+=retry_tokens;host_assignments=parsed.get('source_point_host_assignments')
                    host_assignments_valid=True;name_to_host_parts=retry_map
                    ids=parsed.get('part_ids')
        except (ValueError,RuntimeError):
            pass
    valid=(not truncated and parsed.get('decision')=='SELECT_PARTS' and _fai_number(parsed.get('fai'))==item['number']
        and isinstance(ids,list) and ids and all(isinstance(value,str) and value in lookup for value in ids)
        and len(ids)==len(set(ids)) and bool(parsed.get('source_feature')) and bool(parsed.get('evidence'))
        and host_assignments_valid)
    if host_assignments_valid and expected_source_names:
        parsed=dict(parsed,source_point_host_assignments=host_assignments,
            source_point_host_part_ids=name_to_host_parts)
    audit={'status':'selected' if valid else 'unconfirmed','decision':parsed,'raw_output':raw,
        'truncated':truncated,'tokens':tokens,'cad_sha256':mesh['source_sha256'],
        'host_assignment_attempts':host_attempts,
        'document_sha256':next((doc.get('sha256') for doc in project.get('documents',[]) if doc.get('id')==item.get('document_id')),None),
        'screening':screening,'catalog_images':[{'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in pages],
        'does_not_authorize_measurement_anchors':True}
    audit=_persist_part_selection(store,project,item,mesh,audit)
    return ([lookup[value] for value in ids] if valid else None),audit
