const $=id=>document.getElementById(id);
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const statuses={running:'正在处理',queued:'排队等候',partial:'本轮结束 · 仍有待解决项',completed:'本轮结束',failed:'任务失败',blocked:'等待可用证据',not_started:'尚未开始'};
const reasons={
 classified_source_relation_graph_unresolved:'原文点名已确认，定义与坐标关联仍待解决',
 classified_xyz_frame_evidence_unavailable:'原文具备 XYZ，当前缺少完整坐标系候选',
 no_fai_local_frame_candidate_passed_geometry_gate:'整组坐标候选未通过几何门限，已停止像素重试',
 source_entity_absent_at_current_assembly_stage:'MSOP 明确当前装配阶段不含该实体',
 model_xyz_group_review_abstained_or_invalid:'本地模型未确认整组坐标候选',
 local_part_identity_not_confirmed:'本地模型尚未确认对应零件及全部重复实例',
 marker_outside_local_model_selected_parts:'标记落在模型选定零件以外的实体上',
 local_entity_group_review_not_confirmed:'实体或完整点组未通过独立复核',
 local_drawing_location_manifest_not_confirmed:'原图的位置数量或重复实例未确认',
 local_source_manifest_not_confirmed:'原文直接测量点清单未确认',
 source_point_coverage_incomplete:'模型未完整覆盖原文测量点',
 model_visual_decision_invalid:'模型输出尚未通过完整性校验',
 model_selected_pixel_group_did_not_fully_hit_cad_surface:'点组未通过 CAD 表面或几何一致性检查',
 source_even_spacing_constraint_failed:'点位分布与原图等距要求不一致',
 candidate_id_selection_invalid:'模型选择的 CAD 候选与源点不对应',
 registration_reference_audit_failed:'配准参考点未通过核对',
 local_coordinate_variant_selection_failed:'同名点的坐标版本尚未消歧',
 source_coordinate_conflict:'源点坐标存在多个版本',
 no_explicit_cross_page_geometry_reference:'尚未找到明确的跨页几何关联',
 decision_not_auto_mark:'模型未确认可自动标记',
 local_vision_inference_failed:'本地视觉推理未成功完成',
 no_geometry_candidate:'尚无完整有效几何候选',
 visual_evidence_exceeds_context_budget:'完整证据超出本轮推理容量',
 local_multimodal_vision_model_unavailable:'本地视觉模型尚未就绪',
 required_xyz_rigid_fit_not_established:'源坐标与 CAD 的刚体对应尚未建立'
};
const projectData=new Map(),snapshots=new Map();let busy=false,lastDetail=0,events=[];
try{events=JSON.parse(localStorage.getItem('oracle-progress-events-v1')||'[]');if(!Array.isArray(events))events=[];}catch{}
function time(value){return new Date(value).toLocaleString('zh-CN',{hour12:false});}
async function get(path){const response=await fetch(path,{cache:'no-store',signal:AbortSignal.timeout(15000)});if(!response.ok)throw new Error(`服务返回 ${response.status}`);return response.json();}
function record(project,state){
 const coverage=state.coverage||{},progress=state.progress||{};
 const next={status:state.status,marked:coverage.marked,done:progress.done,current:progress.current_fai,message:progress.message};
 const previous=snapshots.get(project.id);snapshots.set(project.id,next);
 if(!previous)return;
 const changes=[];
 if(previous.marked!==next.marked)changes.push(`实际标记 ${previous.marked} → ${next.marked} 项`);
 if(previous.status!==next.status)changes.push(statuses[next.status]||next.status);
 if(previous.current!==next.current&&next.current)changes.push(`开始核对 FAI ${next.current}`);
 else if(previous.message!==next.message&&next.message)changes.push(next.message);
 if(changes.length){events.unshift({at:Date.now(),text:`${project.drawing_no} · ${changes.join('；')}`});events=events.slice(0,60);try{localStorage.setItem('oracle-progress-events-v1',JSON.stringify(events));}catch{}}
}
function renderEvents(){ $('changes').innerHTML=events.length?events.slice(0,20).map(e=>`<li><time>${escape(time(e.at))}</time>${escape(e.text)}</li>`).join(''):'<li class="muted">尚未观察到变化；任务继续运行时会自动记录。记录起于本页面首次同步。</li>'; }
function pendingReason(state,number){
 if(state.status==='queued')return ['等待本轮处理',''];
 if(state.status==='running'&&state.progress?.current_fai===number)return [state.progress.message||'正在核对原文与候选面',''];
 const result=(state.items||[]).find(row=>row.fai===number);
 if(!result)return [state.status==='running'?'等待本轮核对':'尚无本轮详细诊断',''];
 const detail=result.validation_failures?.[0]?.code||result.geometry_diagnostic?.reason;
 const code=detail||result.reason||result.result;
 const audit=result.local_model_audit||result.geometry_diagnostic||{};
 const evidence=audit.decision?.evidence||audit.local_model_decision?.evidence||result.evidence;
 return [reasons[code]||reasons[result.reason]||'模型尚未确认完整有效位置',code,evidence||audit.error||''];
}
function projectCard(project,state){
 const c=state.coverage||{},p=state.progress||{},queued=state.status==='queued';
 const items=projectData.get(project.id)?.items||[];
 const total=Number(c.total)||0,marked=Number(c.marked)||0;
 const percent=total?100*marked/total:0;
 const current=queued?'等待本地模型执行':p.current_fai?`正在核对 FAI ${p.current_fai}`:statuses[state.status]||'读取任务状态';
 const counts=queued?{}:state.counts||{},done=queued?0:Number(p.done??counts.processed)||0;
 const diagnostics=(c.unmarked_numbers||[]).map(n=>{
  const item=items.find(x=>x.number===n),[reason,code,evidence]=pendingReason(state,n);
  return `<tr><td><a href="/viewer?project=${encodeURIComponent(project.id)}&fai=${n}">FAI ${Number(n)}</a></td><td>${escape(item?.measurement_indicators?.map(v=>v.symbol).join(' / ')||'—')}</td><td class="reason">${escape(reason)}${code?`<small>${escape(code)}</small>`:''}${evidence?`<details class="native-detail"><summary>查看模型复核意见</summary><pre>${escape(evidence)}</pre></details>`:''}</td><td>${escape((item?.pages||[]).join(', ')||'—')}</td></tr>`;
 }).join('');
 const confidence={low:0,medium:0,high:0};
 for(const item of items){if(!c.unmarked_numbers?.includes(item.number)&&item.annotation?.status==='auto_marked'&&item.annotation.confidence in confidence)confidence[item.annotation.confidence]++;}
 return `<article class="project" data-project="${escape(project.id)}"><div class="project-head"><div><h3>${escape(project.drawing_no)}<span class="revision">Rev.${escape(project.revision)}</span></h3><div class="project-title">${escape(project.title)}</div></div><div class="head-actions"><span class="badge ${escape(state.status)}">${escape(statuses[state.status]||state.status)}</span><a href="/viewer?project=${encodeURIComponent(project.id)}">查看三维与原文 ↗</a></div></div><div class="project-body"><div><div class="coverage-title"><b>${marked} <span>/ ${total} 项已标记</span></b><span>${percent.toFixed(1)}%</span></div><div class="bar" role="progressbar" aria-label="FAI 标记覆盖" aria-valuenow="${marked}" aria-valuemin="0" aria-valuemax="${total}"><div style="width:${Math.min(100,Math.max(0,percent))}%"></div></div><div class="coverage-summary">仍未标记 ${Number(c.unmarked)||0} 项 · 本地模型标记 ${Number(c.local_model_marked)||0} 项</div></div><div class="task"><div class="task-label">当前任务</div><div class="task-current">${escape(current)}</div><div class="task-meta">${escape(p.message||state.message||'尚无任务说明')}<br>本轮已核对 ${done} / ${Number(state.total)||0} 项${queued?'':` · 通过 ${Number(counts.accepted)||0} · 未通过 ${Number(counts.abstain)||0}`}${state.updated_at?`<br>任务记录更新于 ${escape(time(state.updated_at))}`:''}${state.error?`<br>错误：${escape(state.error).slice(0,1000)}`:''}</div></div></div><div class="count-strip"><span>显示置信度：高 ${confidence.high} / 中 ${confidence.medium} / 低 ${confidence.low}（详情每 30 秒同步）</span><span>置信度来自模型及校验规则，不等于实测准确率</span></div><details data-pending="${escape(project.id)}"><summary>未标记清单与原因 · ${Number(c.unmarked)||0} 项</summary><div class="table-scroll">${diagnostics?`<table><thead><tr><th>测量项</th><th>指标</th><th>当前情况</th><th>MSOP 页</th></tr></thead><tbody>${diagnostics}</tbody></table>`:'<p class="muted">当前全部 FAI 均有标记；仍需核对位置与工程方案。</p>'}</div></details></article>`;
}
async function renderStrategies(projects,results){
 try{const update=await get('/viewer-static/method-update.json');$('method-update').textContent=update.message+' 新方法：测量与输出坐标系分离、整组平面约束、共享特征关系与分组原点组合求解。';}catch{$('method-update').textContent='';}

 const responses=await Promise.allSettled(projects.map(project=>get(`/api/viewer/projects/${project.id}/fai-strategies`)));
 const groups=new Map();let done=0,total=0,classified=0;const stages=[];const trials=[];const activeIds=new Set(results.filter(row=>row.state?.status==='running').map(row=>row.project.id));
 for(let index=0;index<responses.length;index++){
  const response=responses[index],project=projects[index];if(response.status!=='fulfilled')continue;
  const state=response.value;for(const trial of state.route_trials||[]){trials.push({...trial,project,routeLabel:state.routes[trial.route]});if(trial.status==='running'){activeIds.add(project.id);stages.push(`${project.drawing_no} 正在复核 FAI ${trial.fai}`);}}if(state.source_stale){stages.push(`${project.drawing_no} 分类证据已过期`);continue;}
  done+=Number(state.done)||0;total+=Number(state.total)||0;
  if(state.status==='running'){activeIds.add(project.id);stages.push(`${project.drawing_no} 正在分类 FAI ${(state.current_fai||[]).join('、')}`);}
  const coverage=results.find(row=>row.project.id===project.id)?.state?.coverage;
  for(const item of state.items||[]){
   classified++;const pending=coverage?coverage.unmarked_numbers?.includes(item.fai):null;
   const view=$('strategy-view').value;
   if(view!=='method'&&pending!==true)continue;
   const key=view==='failure'?view+'/'+item.failure_class:view==='route'?view+'/'+item.next_route:item.measurement_family+'/'+item.evidence_class;
   const name=view==='failure'?(item.failure_class==='none'?'失败诊断尚未记录':state.failure_classes[item.failure_class]):view==='route'?state.routes[item.next_route]:state.families[item.measurement_family];
   const evidence=view==='method'?state.evidence_classes[item.evidence_class]:'当前未标记项';
   if(!groups.has(key))groups.set(key,{name,evidence,items:[],marked:0,pending:0,unknown:0});
   const group=groups.get(key);
   if(pending===true)group.pending++;else if(pending===false)group.marked++;else group.unknown++;
   group.items.push({project,item,pending,route:state.routes[item.next_route],failure:state.failure_classes[item.failure_class]});
  }
 }
 const trialReasons={main_registration_in_progress:'主任务正在重建配准，尚无完整帧可供本次独立试验使用',reviewed_measurement_scope_has_no_frame_seed:'本地模型确认的测量坐标系尚无完整配准候选',local_measurement_scope_review_unconfirmed:'本地模型尚未确认测量坐标归属',model_xyz_group_review_abstained_or_invalid:'整组几何候选已进入模型复核，语义确认或输出尚未通过',global_host_identity_unconfirmed:'全局对比后，模型仍未确认主体零件',projected_group_shape_drift_exceeded_gate:'完整点组的几何形状变化超出门限，未放宽校验条件',no_fai_local_frame_candidate_passed_geometry_gate:'尚无整组候选通过几何校验',selected_part_comparison_image_budget_exceeded:'候选主体仍过多，完整对比资料超出图像容量',local_part_screening_invalid:'本地模型未完整确认当前批次候选',local_model_found_no_part_candidates:'本地模型未确认可用的主体候选'};
 $('strategy-trials').innerHTML=trials.length?'<h3 class="trial-title">新路线最近一次验证 · 不计入标记覆盖</h3>'+trials.map(trial=>`<p class="trial-row"><b>${escape(trial.project.drawing_no)} / FAI ${trial.fai}</b> · ${escape(trial.routeLabel)}<br>${trial.status==='deferred'?'等待主任务结束后重试':trial.status==='running'?'正在处理：'+escape(trial.stage||'本地模型与整组几何复核'):trial.status==='accepted'?'本地模型已通过完整点组复核，实际标记数以项目统计为准':trial.status==='selected'?'模型已选择候选主体，后续实体与点组复核未完成':'本阶段尚未通过'}${trial.reason?'：'+escape(trialReasons[trial.reason]||trial.reason):''}${trial.decision?.part_ids?.length?' · 主体 '+escape(trial.decision.part_ids.join('、')):''}${trial.source_point_count?'<br>完整原文点组 '+Number(trial.source_point_count)+' 点':''}${Number.isFinite(trial.geometry_diagnostic?.max_pairwise_shape_error_mm)?' · 形状误差 '+Number(trial.geometry_diagnostic.max_pairwise_shape_error_mm).toFixed(3)+' mm / 门限 '+Number(trial.geometry_diagnostic.gate_mm)+' mm':''}</p>`).join(''):'';
 $('active-projects').textContent=`${activeIds.size} 个项目`;
 if(stages.length)$('active-description').textContent='本地模型分类或完整点组复核进行中';
 $('strategy-progress').textContent=total?`已分类 ${classified}/${total} 项${stages.length?' · '+stages.join('；'):done>=total?' · 本轮分类结束':''}`:'等待本地模型完成分类';
 const expandedStrategies=new Set([...document.querySelectorAll('[data-strategy][open]')].map(el=>el.dataset.strategy));
 $('strategy-groups').innerHTML=[...groups.values()].sort((a,b)=>b.pending-a.pending||b.items.length-a.items.length).map(group=>`<details class="strategy-class" data-strategy="${escape(group.name+'/'+group.evidence)}"><summary><strong>${escape(group.name)}</strong><span>${escape(group.evidence)}</span><small>已有标记 ${group.marked} · 待解决 ${group.pending}${group.unknown?' · 状态未同步 '+group.unknown:''}</small></summary><div class="table-scroll"><table><thead><tr><th>项目 / FAI</th><th>当前状态</th><th>模型建议路线与依据</th></tr></thead><tbody>${group.items.sort((a,b)=>Number(b.pending)-Number(a.pending)||a.item.fai-b.item.fai).map(({project,item,pending,route,failure})=>`<tr><td><a href="/viewer?project=${encodeURIComponent(project.id)}&fai=${item.fai}">${escape(project.drawing_no)}<br>FAI ${item.fai}</a></td><td>${pending===null?'状态未同步':pending?'待解决':'已有标记'}<small class="classification-failure">${escape(failure)}</small></td><td><b>${escape(route)}</b><br>${escape(item.reason)}<small class="source-quote">${(item.source_quotes||[]).map(q=>`MSOP p.${q.page}：${escape(q.text)}`).join('<br>')}</small></td></tr>`).join('')}</tbody></table></div></details>`).join('')||'<p class="muted">本地模型正在整理分类；有经过原文引用核验的结果后逐批显示。</p>';
 for(const el of document.querySelectorAll('[data-strategy]'))el.open=expandedStrategies.has(el.dataset.strategy);
}

async function refresh(force=false){
 if(busy)return;busy=true;$('refresh').disabled=true;
 try{
  const [all,service,training]=await Promise.all([get('/api/viewer/projects'),get('/api/status'),get('/api/viewer/cloud-training').catch(()=>null)]);
  if(training){
   const pipeline=training.pipeline||{},p=training.progress||{},data=training.dataset||{},round=training.round||{};
   const stale=pipeline.status!=='stopped'&&(training.connection_error||training.synced_at&&Date.now()/1000-training.synced_at>60);
   const stages={train:'LoRA 训练',evaluate:'独立项目验证','evaluate-vllm':'云端运行时对照验证',evaluated:'验证结束',stopped_by_user:'已按要求停止',failed:'训练或验证任务未完成'};
   $('cloud-training-state').textContent=stale?'同步中断 · 以下保留上次记录':`${pipeline.model||'Qwen'} · ${stages[pipeline.stage]||pipeline.stage||'尚未开始'} · ${pipeline.status==='complete'?'完成':pipeline.status==='failed'?'失败':pipeline.status==='running'?'进行中':pipeline.status==='stopped'?'已停止':'等候进程'}`;
   const loss=Number(p.metrics?.loss);
   $('cloud-training-progress').textContent=pipeline.status==='stopped'?'云端训练及后续任务已停止，继续使用本机 Strata 处理 FAI。':p.error||pipeline.error||(pipeline.status==='complete'?`完整对照验证已结束；${pipeline.source_gate_passed?'原文读取检查通过':'本轮候选未达到全部采用条件'}。Viewer 当前仍使用原底座，候选尚未设为默认模型。`:p.global_step?`训练步骤 ${p.global_step}/${p.max_steps} · Loss ${Number.isFinite(loss)?loss.toFixed(5):'—'}`:p.done?`已处理 ${p.done}/${p.total} 页`:pipeline.stage==='train'?'训练进程已启动，正在加载底座与适配器。':'正在读取任务进度。');
   const manifestTask=data.training_scope==='FAI_direct_measurement_manifest';
   $('cloud-training-data').textContent=`${manifestTask?'FAI 测量清单专项 · ':''}训练 ${data.samples?.train??'—'} ${manifestTask?'条样本':'页'} / ${data.point_count?.train??'—'} 个原文点${training.training?.status==='complete'?`（已完成 ${training.training.steps}/${training.training.max_steps} 步）`:''} · 验证 ${data.samples?.validation??'—'} ${manifestTask?'条样本':'页'} / ${data.point_count?.validation??'—'} 个原文点 · 按项目隔离验证 · 当前适配器${round.adapter_deployed?'已用于清单确认':pipeline.deployment_status==='deployed'?'已部署':'尚未部署'}`;
   if(round.stage==='marking')$('cloud-training-progress').textContent=`训练及对照验证结束 · 正在重跑 ${round.project} · ${round.project_progress?.progress?.message||'等待模型'} · ${round.project_progress?.coverage?.marked??'—'}/${round.project_progress?.coverage?.total??'—'} 项已有完整标记`;
   else if(round.stage==='waiting_for_current_queue')$('cloud-training-progress').textContent+=' 等当前任务队列完成后加载修复并重跑三个项目。';
   else if(round.stage==='finished')$('cloud-training-progress').textContent=`本轮重跑结束 · ${round.all_FAI_fully_marked?'所有FAI均已有完整标记':'仍有未通过项，详见各项目诊断'} · ${(round.results||[]).map(r=>`${r.project} ${r.coverage?.marked}/${r.coverage?.total}`).join('；')}`;
   const metrics=pipeline.metrics||p.metrics||{};
   $('cloud-training-validation').textContent=['base','adapter'].filter(k=>metrics[k]).map(k=>{const m=metrics[k];return `${k==='base'?'底座':'候选'}已验证 ${m.pages} 页：${manifestTask?'清单决策完整正确':'完整坐标正确'} ${m.exact_pages} 条，裸 JSON 合规 ${m.valid_json_pages} 页，${manifestTask?'正确测量点名':'正确坐标点'} ${m.correct_points}/${m.expected_points}`;}).join('；');
   $('cloud-training-gpu').textContent=pipeline.status==='stopped'?'云端 GPU 进度已停止同步；当前推理后端以项目服务状态为准。':`GPU 分工：0 号卡训练或验证，1 号卡在线推理。${(training.gpu||[]).map(v=>{const [id,used,total,util]=v.split(',').map(x=>x.trim());return `${id} 号卡：显存 ${used}/${total} MiB，利用率 ${util}%`;}).join('；')}`;
  }else $('cloud-training-state').textContent='训练记录暂不可读取';
  const projects=all.filter(p=>p.drawing_no!=='817-07781');
  const detailDue=force||Date.now()-lastDetail>30000;
  const results=await Promise.all(projects.map(async project=>{
   try{
    const [state,detail]=await Promise.all([get(`/api/viewer/projects/${project.id}/local-model-marking`),detailDue||!projectData.has(project.id)?get(`/api/viewer/projects/${project.id}`):Promise.resolve(null)]);
    if(detail)projectData.set(project.id,detail);record(project,state);return {project,state};
   }catch(error){return {project,error};}
  }));
  if(detailDue)lastDetail=Date.now();
  const open=new Set([...document.querySelectorAll('[data-pending][open]')].map(n=>n.dataset.pending));
  const openNative=new Set([...document.querySelectorAll('.native-detail[open]')].map(n=>n.closest('tr')?.querySelector('a')?.href));
  $('projects').innerHTML=results.map(({project,state,error})=>error?`<div class="empty">${escape(project.drawing_no)}：状态读取失败，${escape(error.message)}。等待重新连接。</div>`:projectCard(project,state)).join('')||'<div class="empty">当前没有纳入范围的项目。</div>';
  for(const el of document.querySelectorAll('[data-pending]'))el.open=open.has(el.dataset.pending);
  for(const el of document.querySelectorAll('.native-detail'))el.open=openNative.has(el.closest('tr')?.querySelector('a')?.href);
  const good=results.filter(x=>x.state),sum=key=>good.reduce((n,x)=>n+(Number(x.state.coverage?.[key])||0),0);
  const active=good.filter(x=>x.state.status==='running'),queued=good.filter(x=>x.state.status==='queued');
  $('total-marked').textContent=`${sum('marked')} / ${sum('total')}`;$('total-pending').textContent=sum('unmarked');
  $('active-projects').textContent=`${active.length} 个项目`;
  $('active-description').textContent=`${queued.length} 个项目排队${results.length!==good.length?' · 有项目暂时无法读取':''}`;
  const model=service.viewer_model||{};
  $('model-state').textContent=service.viewer_model_error?'模型异常':model.loaded?'已加载':'未加载';
  $('model-name').textContent=[model.inference_backend,model.model_alias].filter(Boolean).join(' · ')||'未取得模型信息';
  $('scope-note').textContent=all.some(p=>p.drawing_no==='817-07781')?'817-07781 因三维资料问题，按当前约定暂不纳入本页统计。':'展示当前 Viewer 项目。';
  $('connection').textContent=good.length===results.length?'● 已连接本地服务':'部分项目连接异常';
  $('last-update').textContent=`最近同步 ${new Date().toLocaleTimeString('zh-CN',{hour12:false})}`;
  $('fetch-state').textContent='状态每 5 秒同步 · 项目详情每 30 秒同步';
  $('error').hidden=good.length===results.length;
  if(!$('error').hidden)$('error').textContent='部分项目读取失败，汇总只包含成功读取的项目；页面会自动重试。';
  renderEvents();
  await renderStrategies(projects,results);
 }catch(error){
  $('connection').textContent='连接中断';$('error').hidden=false;
  $('error').textContent=`暂时无法读取进度：${error.message}。以下保留最后一次成功读取的数据，5 秒后自动重试。`;
  $('fetch-state').textContent='数据暂未更新，请以最近同步时间为准';
 }finally{busy=false;$('refresh').disabled=false;}
}
$('refresh').addEventListener('click',()=>refresh(true));
$('strategy-view').addEventListener('change',()=>refresh());
refresh(true);setInterval(()=>refresh(),5000);
