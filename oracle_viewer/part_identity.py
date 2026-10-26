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
    from .local_matching import _object,_complete_visual_context_budget,_fai_number
    parts=cad_parts(mesh);all_parts=parts
    geometry_support=_candidate_geometry_support(mesh,geometry_hypothesis)
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
            if not valid:return None,_persist_part_selection(store,project,item,mesh,{'status':'unconfirmed',
                'reason':'local_part_screening_invalid','screening':screening})
            candidates.extend(part for part in batch if part['part_id'] in ids)
        if not candidates:return None,_persist_part_selection(store,project,item,mesh,{'status':'unconfirmed',
            'reason':'local_model_found_no_part_candidates','screening':screening})
        parts=candidates
        pages=render_part_catalog(mesh,parts,store._dir(project['id'])/'automatic-learning'/'part-catalog')
        images=list(dict.fromkeys([*source_images,*pages]))
        if screening:
            payload={'fai':item['number'],'methods':detail.get('methods',[]),
                'direct_source_point_CAD_display_hypotheses':geometry_support,
                'all_original_bodies':[{k:v for k,v in part.items() if k!='face_ids'} for part in all_parts]}
            question=('你是本地 CAD 主体零件全局复核模型。此前分批候选过宽，尚未授权任何实体。'
                '请结合原图和完整原始body尺寸、位置、原型编号，选择承载本FAI测量特征的所有主体零件。'
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
        '结合零件包围盒的装配坐标确认左右/上下，不能把每张缩放后的零件缩略图当成同一尺寸。'
        '若提供direct_source_point_CAD_display_hypotheses，只能把其中完全同名的直接测点作为空间相容性提示；这些点位未经注册，不是正确答案。'
        '选择当前原文特征所依附的主体零件body，例如螺母、导轨、挡片、窗扣；先识别主体，不要把小焊点body当成主体。'
        '包括PER NUT/PER FANG/PER SIDE的所有重复主体实例；不能只选择最容易看到的一个实例，也不能凭数量相等判定身份。'
        '不得用整个框架body代替原文指定螺母/导轨/挡片，不能因邻接而把无关零件算进来。没有充分对应证据时ABSTAIN。'
        '只输出JSON：decision(SELECT_PARTS/ABSTAIN),fai,part_ids(明确B编号数组),'
        'source_feature(原文特征名),evidence(一句说明引线/形状/装配位置依据),confidence(low/medium/high)。')
    messages=[{'role':'user','content':question,'attachments':[{'filename':'Original sources and complete numbered CAD bodies',
        'parsed_text':json.dumps(context,ensure_ascii=False),'images':images}]}]
    raw,truncated,tokens=model._generate_once(messages,max_context_chars=_complete_visual_context_budget(messages),max_new_tokens=1024,max_images=len(images),temperature=0.)
    parsed=_object(raw) or {};ids=parsed.get('part_ids');lookup={part['part_id']:part for part in parts}
    valid=(not truncated and parsed.get('decision')=='SELECT_PARTS' and _fai_number(parsed.get('fai'))==item['number']
        and isinstance(ids,list) and ids and all(isinstance(value,str) and value in lookup for value in ids)
        and len(ids)==len(set(ids)) and bool(parsed.get('source_feature')) and bool(parsed.get('evidence')))
    audit={'status':'selected' if valid else 'unconfirmed','decision':parsed,'raw_output':raw,
        'truncated':truncated,'tokens':tokens,'cad_sha256':mesh['source_sha256'],
        'document_sha256':next((doc.get('sha256') for doc in project.get('documents',[]) if doc.get('id')==item.get('document_id')),None),
        'screening':screening,'catalog_images':[{'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in pages],
        'does_not_authorize_measurement_anchors':True}
    audit=_persist_part_selection(store,project,item,mesh,audit)
    return ([lookup[value] for value in ids] if valid else None),audit
