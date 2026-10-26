"""Native-model review of measurement frames, distinct from dimension frames.

PDF column order is not semantic authority. Code enumerates printed scopes;
only the configured local model may assign the original XYZ groups to them.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path


def _page_explicitly_covers_fai(page_text, fai):
    """A nearby/shared point page is not automatically the current FAI's method."""
    text=re.sub(r'\s+',' ',page_text or '')
    headings=re.findall(r'\bFAI\s*(\d+)\s*(?:to|through|[-–])\s*(?:FAI\s*)?(\d+)\b',text,re.I)
    if any(int(start)<=fai<=int(end) for start,end in headings):return True
    return bool(re.search(r'\bFAI\s*'+str(int(fai))+r'\b',text,re.I))


def _scope_assignment_errors(parsed, *, fai, expected, programs, frames, quotes, methods, truncated=False, output_table_frames=None):
    """Validate source-only decisions without selecting a frame on the model's behalf."""
    errors=[]
    if truncated:errors.append('output_truncated')
    if parsed.get('decision')!='ASSIGN_SCOPES':errors.append('decision_must_be_ASSIGN_SCOPES_or_ABSTAIN')
    if parsed.get('fai')!=fai:errors.append('fai_identity_mismatch')
    assignments=parsed.get('groups')
    if (not isinstance(assignments,list) or not all(isinstance(a,dict) for a in assignments)
            or len(assignments)!=len(expected)
            or {a.get('group_id') for a in assignments}!=expected):
        errors.append('every_source_group_must_occur_exactly_once')
        return errors
    if not parsed.get('evidence'):errors.append('missing_evidence')
    for a in assignments:
        group=a.get('group_id')
        if a.get('program') not in programs:errors.append(f'{group}:program_not_in_printed_options')
        if a.get('coordinate_system') not in frames:errors.append(f'{group}:frame_not_in_printed_options')
        citation=quotes.get(a.get('source_quote_id'),a)
        quote=citation.get('source_quote')
        text=next((p.get('text','') for p in methods if p.get('page')==citation.get('source_page')),'')
        if (not isinstance(quote,str) or len(quote)<10
                or re.sub(r'\s+',' ',quote).strip() not in re.sub(r'\s+',' ',text).strip()):
            errors.append(f'{group}:quote_not_verbatim_on_cited_page')
        elif (not _page_explicitly_covers_fai(text,fai)
              and not any(p.get('page')==citation.get('source_page') and p.get('is_fai_method_page')
                          for p in methods)):
            errors.append(f'{group}:cited_page_does_not_identify_this_fai')
        elif not (re.search(r'\bMeasured\s+System\b|\bunder\s+alignment\b',quote,re.I)
                  or (a.get('scope_kind')=='coordinate_table_output_frame'
                      and a.get('coordinate_system') in (output_table_frames or {}).get(group,[])
                      and re.search(r'\bOutput\s+system\b',quote,re.I))):
            errors.append(f'{group}:quote_does_not_identify_measurement_frame')
        elif not re.search(r'\b'+re.escape(str(a.get('coordinate_system')))+r'\b',quote):
            errors.append(f'{group}:selected_frame_not_present_as_complete_token_in_quote')
        if not a.get('reason'):errors.append(f'{group}:missing_reason')
    return errors


def review_source_scopes(store,project,item,detail,model,points):
    from .local_matching import _object,_is_local_inference_model,_method_page_images,_source_point_family
    if not model.loaded or not _is_local_inference_model(model):
        return None,{'status':'unavailable','reason':'local_model_required'}
    methods=detail.get('methods') or []
    msop=store.active_msop(project) or {}
    all_pages=msop.get('method_pages') or []
    page_index={p.get('page'):p for p in all_pages}
    source_page_numbers={page for point in points for page in point.get('pages',[])}
    source_pages=[page_index[number] for number in sorted(source_page_numbers) if number in page_index]
    # Some MSOPs put the feature's XYZ/frame table on a separate construction
    # page, while the FAI method page carries only the operation. Include only
    # pages already cited by this FAI's extracted points; never search unrelated
    # pages globally or trust an inherited frame as a model decision.
    linked_pages=[]
    for page in source_pages:
        if _page_explicitly_covers_fai(page.get('text'),int(item['number'])):
            linked_pages.append(page)
    evidence_by_page={p.get('page'):{**p,'is_fai_method_page':False} for p in linked_pages}
    evidence_by_page.update({p.get('page'):{**p,'is_fai_method_page':True} for p in methods})
    scoped_evidence_pages=list(evidence_by_page.values())
    frames=sorted(set(re.findall(r'\bALI_[A-Z0-9_]+','\n'.join(p.get('text','') for p in scoped_evidence_pages))))
    programs=sorted({p.get('program') for p in [*methods,*points] if p.get('program')})
    if not frames:return None,{'status':'unavailable','reason':'source_scope_not_printed'}
    groups={}
    for point in points:
        key=(tuple(point.get('pages') or []),_source_point_family(point['name']),point.get('program'),point.get('coordinate_system'))
        groups.setdefault(key,[]).append(point)
    from .msop_tables import feature_coordinate_rows
    output_table_frames={}
    for i,(_key,rows) in enumerate(groups.items()):
        permitted=[]
        for page in scoped_evidence_pages:
            if re.search(r'\bMeasured\s+System\b',page.get('text',''),re.I):continue
            table={r['name']:r for r in feature_coordinate_rows(page) if r['axes']==['X','Y','Z']}
            if all(point['name'] in table and point.get('xyz') is not None
                   and all(abs(float(v)-float(w))<1e-9 for v,w in zip(table[point['name']]['values'],point['xyz']))
                   for point in rows):
                permitted.extend(table[point['name']]['coordinate_system'] for point in rows
                                 if table[point['name']].get('coordinate_system'))
        if permitted:output_table_frames[f'S{i+1}']=sorted(set(permitted))
    if not programs:
        if len(output_table_frames)!=len(groups) or any(p.get('program') for p in all_pages):
            return None,{'status':'unavailable','reason':'source_scope_not_printed'}
        programs=[None]
    payload={'fai':item['number'],'methods':methods,'allowed_coordinate_systems':frames,'allowed_programs':programs,
        'output_table_frames':output_table_frames,
        'groups':[{'group_id':f'S{i+1}','source_pages':list(key[0]),'prior_program':key[2],
            'prior_coordinate_system':key[3],'points':[{'name':p['name'],'xyz':p.get('xyz')} for p in rows]}
            for i,(key,rows) in enumerate(groups.items())]}
    expected={g['group_id'] for g in payload['groups']}
    quotes={}
    for page in scoped_evidence_pages:
        text=page.get('text','')
        for match in re.finditer(r'\bALI_[A-Z0-9_]+',text):
            qid=f'Q{len(quotes)+1}'
            quotes[qid]={'source_page':page['page'],'source_quote':text[max(0,match.start()-120):match.end()]}
    payload['source_coordinate_pages']=[{'page':p.get('page'),'text':p.get('text',''),
        'explicit_fai_link':_page_explicitly_covers_fai(p.get('text'),int(item['number']))}
        for p in source_pages]
    payload['verbatim_scope_quote_options']=quotes
    identity={'cad_sha256':project['model']['source_sha256'],'msop_sha256':msop['sha256'],
        'model':model.info(),'payload':payload,'implementation':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    path=store._dir(project['id'])/'automatic-learning'/'scope-reviews'/f'{key}.json'
    question=('You are the local MSOP coordinate-scope reviewer. PDF text extraction can interleave left-hand '
        'Measured System / Dimension System headers with right-hand method instructions. Do not infer scope '
        'from text order or trust prior_coordinate_system. Determine which MEASUREMENT alignment owns each '
        'entire original XYZ group; Dimension System is the reporting frame, not automatically the measuring frame. '
        'An explicit under alignment clause applies to its named feature group. Otherwise examine the original '
        'method and the specifically linked coordinate-definition pages together. A separate coordinate page can '
        'supply the printed frame when the FAI method page omits it; match exact point names and XYZ rows, then '
        "use the FAI method's measurement context to decide whether that frame is authoritative. Coordinate-page "
        'Dimension System labels alone do not prove the measurement frame. Never alter XYZ, names or split a group to improve CAD fit. '
        'Only use listed printed programs and alignments. Return ABSTAIN when conflicting evidence prevents a '
        'whole-group decision. Output JSON: decision=ASSIGN_SCOPES, fai, groups=[{group_id,program,'
        'coordinate_system,source_quote_id,reason}], evidence. Select source_quote_id from the provided Q identifiers '
        'showing the measurement-frame rule; those are literal original excerpts, not generated cleaned quotations. '
        'Select the quote about measurement or the explicit under-alignment clause, not the reporting header. Every group exactly once. '
        'For groups listed in output_table_frames, the model may instead confirm the literal coordinate-table '
        'output frame with scope_kind=coordinate_table_output_frame. This identifies the XYZ table frame only, '
        'not the probe measurement alignment. Select the exact Output system quote. When the only allowed '
        'program is null, preserve null; never invent UP or DOWN. '
        'This selects source semantics only; it does not authorize any CAD anchor.\n'+json.dumps(payload,ensure_ascii=False))
    def errors_for(parsed,cut):
        return _scope_assignment_errors(parsed,fai=item['number'],expected=expected,programs=programs,
            frames=frames,quotes=quotes,methods=scoped_evidence_pages,truncated=cut,
            output_table_frames=output_table_frames)
    if path.is_file():receipt=json.loads(path.read_text());parsed=receipt['decision']
    else:
        images=_method_page_images(store,project,detail,model,item['number'])
        if len(images)>16:return None,{'status':'unavailable','reason':'scope_source_image_budget_exceeded'}
        attempts=[]
        request=question
        for attempt in range(3):
            raw,cut,tokens=model._generate_once([{'role':'user','content':request,'attachments':[
                {'filename':'Original MSOP and drawing scope evidence','images':images}]}],
                max_context_chars=len(request)+1024,max_new_tokens=2048,max_images=len(images),temperature=0.)
            parsed=_object(raw) or {}
            errors=errors_for(parsed,cut)
            attempts.append({'attempt':attempt+1,'raw_output':raw,'decision':parsed,
                'truncated':cut,'generated_tokens':tokens,'validation_errors':errors})
            if not errors or parsed.get('decision')=='ABSTAIN':break
            # The validator reports contract failures, never supplies a preferred
            # alignment or changes coordinates. The local model must reconsider
            # the same original evidence and author the complete replacement.
            request=question+'\nLOCAL OUTPUT CONTRACT FEEDBACK:\n'+json.dumps({
                'previous_output':raw,'validation_errors':errors,
                'instruction':'Re-examine original evidence. Return a complete replacement JSON for all groups. '
                    'The selected frame must match a complete printed token in the cited measurement rule. '
                    'Do not choose a frame merely to satisfy validation; ABSTAIN if evidence is insufficient. '
                    'Do not change source XYZ or authorize CAD anchors.'},ensure_ascii=False)
        receipt={**identity,'question':question,'raw_output':raw,'truncated':cut,
            'generated_tokens':tokens,'decision':parsed,'attempts':attempts,'validation_errors':errors}
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    assignments=parsed.get('groups')
    errors=errors_for(parsed,receipt.get('truncated',False))
    valid=not errors
    if valid:
        for a in assignments:
            if a.get('source_quote_id') in quotes:
                a.update(quotes[a['source_quote_id']])
    audit={'status':'selected' if valid else 'unconfirmed','receipt':str(path),
        'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decision':parsed,
        'validation_errors':errors,'attempt_count':len(receipt.get('attempts',[])),
        'source_xyz_modified':False,'does_not_authorize_anchors':True}
    if not valid:return None,audit
    selected={a['group_id']:a for a in assignments};resolved=[]
    for i,rows in enumerate(groups.values()):
        a=selected[f'S{i+1}']
        for row in rows:
            resolved.append({**row,'program':a['program'],'coordinate_system':a['coordinate_system'],
                'scope_review_receipt':audit['receipt_sha256']})
    return resolved,audit


def new_scope_hypotheses(mesh,points,scopes):
    """Unpaired rigid candidates for reviewed scopes; final local review required."""
    import numpy as np
    from .auto_registration import surface_seed_candidates,refine_surface_fit
    from .local_matching import cached_surface_geometry,_project_model_selected_xyz_group,_planar_frame_hypotheses
    triangles,_ids,index=cached_surface_geometry(mesh);frames=[]
    for scope in sorted(scopes):
        rows=[p for p in points if (p.get('program'),p.get('coordinate_system'))==scope]
        xyz=np.asarray([p['xyz'] for p in rows],float)
        if len(xyz)<5 or np.linalg.matrix_rank(xyz-xyz.mean(0),tol=1e-6)<2:continue
        seeds=surface_seed_candidates(xyz,triangles,index=index,limit=8)
        fit=refine_surface_fit(xyz,triangles,seeds,index=index)
        candidates=[fit['best'],*fit['alternatives'],*seeds]
        scoped=[]
        for i,candidate in enumerate(candidates):
            scoped.append({k:v for k,v in candidate.items() if k not in ('cad_points_mm','residuals_mm')} | {
                'program':scope[0],'coordinate_system':scope[1],'hypothesis_id':f'REVIEWED_SCOPE_{i+1}',
                'requires_fresh_local_model_selection':True,'source_to_cad_registered':False})
        # Numerical ICP may tilt a nominal plane and harm a formerly coplanar
        # group. Retain analytic plane seeds, then optimize only tangential
        # rigid motion with ALL source points, without trimmed point subsets.
        planar=[]
        for seed in scoped:
            planar.extend(trial[0] for trial in _planar_frame_hypotheses(mesh,rows,[seed]))
        planar.extend(_plane_minimax_candidates(xyz,index,planar))
        valid=[]
        for frame in [*scoped,*planar]:
            projection,error=_project_model_selected_xyz_group(mesh,rows,[frame],{},[])
            if not error:valid.append(frame)
        frames.extend(valid[:12] if valid else scoped)
    return frames


def _plane_minimax_candidates(source,index,seeds):
    """Whole-plane rigid fit: keep the normal exact and include every point."""
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    source=np.asarray(source,float)
    if np.ptp(source[:,2])>1e-6:return []
    result=[];seen=set()
    for seed in seeds:
        r=np.asarray(seed['rotation'],float);t=np.asarray(seed['translation_mm'],float)
        signature=tuple(np.round(np.r_[r.ravel(),t],5))
        if signature in seen:continue
        seen.add(signature)
        normal=r[:,2];u=r[:,0];v=r[:,1];center=source.mean(0)@r.T+t
        def pose(params):
            delta=Rotation.from_rotvec(normal*params[0]).as_matrix()
            nr=delta@r;nt=center+u*params[1]+v*params[2]-source.mean(0)@nr.T
            return nr,nt
        def residual(params):
            nr,nt=pose(params);targets=source@nr.T+nt
            errors=np.asarray([target-index.nearest(target,exact=False)['point'] for target in targets])
            # Squared distances make distant points count more, without
            # dropping any source point or modifying its nominal coordinates.
            return (errors*np.linalg.norm(errors,axis=1)[:,None]).ravel()
        fit=least_squares(residual,[0.,0.,0.],bounds=([-np.deg2rad(.5),-.8,-.8],[np.deg2rad(.5),.8,.8]),
            max_nfev=24,diff_step=1e-4)
        nr,nt=pose(fit.x)
        result.append({**seed,'rotation':nr.tolist(),'translation_mm':nt.tolist(),
            'whole_plane_refinement':{'method':'all_point_plane_constrained_rigid_minimax',
                'point_count':len(source),'source_xyz_modified':False,'points_omitted':0,
                'requires_fresh_local_model_selection':True}})
        if len(result)>=8:break
    return result


def low_cardinality_translation_candidates(rows,frame,index,*,bound_mm=.8):
    """Translate an existing orientation; never infer a full pose from a line."""
    import numpy as np
    from scipy.optimize import least_squares
    xyz=np.asarray([row['xyz'] for row in rows],float)
    if not 2<=len(xyz)<5:return []
    r=np.asarray(frame['rotation'],float);t=np.asarray(frame['translation_mm'],float)
    targets=xyz@r.T+t
    def residual(shift):
        errors=np.asarray([p-index.nearest(p,exact=False)['point'] for p in targets+shift])
        return (errors*np.linalg.norm(errors,axis=1)[:,None]).ravel()
    fit=least_squares(residual,[0.,0.,0.],bounds=(-bound_mm,bound_mm),max_nfev=32)
    if not fit.success and fit.nfev>=32:return []
    return [{**frame,'translation_mm':(t+fit.x).tolist(),
        'low_cardinality_translation':{'point_count':len(rows),'rotation_unchanged':True,
            'delta_translation_mm':fit.x.tolist(),'axis_bound_mm':bound_mm,
            'source_xyz_modified':False,'source_to_cad_registered':False,
            'requires_fresh_local_model_selection':True}}]
