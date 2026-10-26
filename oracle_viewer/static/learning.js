const el=id=>document.getElementById(id);
const escape=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const open=document.createElement('button');open.id='automatic-learning';open.textContent='自动配准与训练';el('export').before(open);
const dialog=document.createElement('dialog');dialog.className='engineering-dialog';dialog.innerHTML='<div class="wf-header"><h2>自动配准与训练</h2><button id="learning-close">关闭 ×</button></div><div id="learning-state" class="wf-content" aria-live="polite"></div><div class="wf-actions"><button id="learning-start">开始 / 继续运行</button></div>';document.body.append(dialog);
let pid=null,timer=null;
const phaseNames={source_adapter_training:'原文输出层适配训练',source_adapter_evaluation:'原文输出层适配验证',source_labels:'原文训练样本',project_split:'独立项目测试集',geometry_labels:'CAD位置训练样本',weight_training:'模型权重训练',evaluation:'训练后效果验证',deployment:'新模型上线'};
const states={collecting_native_features:'正在采集本地模型特征',evaluating_native_output:'正在验证完整原文输出',rejected:'验证未通过，未启用',source_adapter_rejected_geometry_pending:'原文适配候选验证未通过，CAD位置训练等待有效配准',source_adapter_validated_geometry_pending:'原文输出适配已验证，CAD位置训练等待有效配准',not_started:'尚未启动',queued:'正在排队',running:'正在运行',complete:'已完成',prepared_training_blocked:'资料准备已完成，训练环境尚未就绪',pending_verified_geometry:'等待有效配准和特征核验',waiting_trainable_backend:'缺少匹配的可训练模型与加载链路',failed:'运行未完成，可继续'};
function render(s){
 if(s.cloud_training){
  const cloud=s.cloud_training,run=cloud.pipeline||{},progress=cloud.progress||{},data=cloud.dataset||{};
  const stages={train:'正在训练 LoRA',evaluate:'正在对照底座验证','evaluate-vllm':'正在用云端运行时对照验证',evaluated:'训练与验证结束',failed:'任务未完成'};
  const progressText=run.status==='complete'?`已完成训练 ${cloud.training?.steps??'—'} 步及独立项目对照验证`:progress.global_step?`${progress.global_step} / ${progress.max_steps} 个训练步骤`:progress.done?`${progress.done} / ${progress.total} 页`:'正在加载训练或验证进程';
  el('learning-start').textContent='重新执行当前项目配准与资料检查';
  el('learning-start').disabled=['queued','running'].includes(s.status);
  el('learning-state').innerHTML=`<h3>云端 Qwen3.8-27B</h3><p>${escape(stages[run.stage]||run.stage||'读取任务')} · ${escape(progressText)}</p><p>训练 ${escape(data.samples?.train)} 页 / ${escape(data.point_count?.train)} 个原文坐标点；独立项目验证 ${escape(data.samples?.validation)} 页 / ${escape(data.point_count?.validation)} 点。GPU 0 训练或验证，GPU 1 提供在线推理。</p><p>本轮是 MSOP 图文坐标读取训练，适配器${run.deployment_status==='deployed'?'已部署':'尚未部署'}。FAI 几何配准与方案质量需要单独验证。</p><p>XML 是根据虚拟拍摄图、对应测量项和像素/空间坐标生成的最终产物；不作为方案生成或本轮训练的输入。</p>${progress.error||run.error?`<p>${escape(progress.error||run.error)}</p>`:''}<p>当前项目配准记录：${escape(states[s.status]||s.status)}。下方按钮只重新检查本项目，不会重复启动云端训练。</p><a href="/viewer-static/progress.html">查看实时训练步骤、损失及 GPU 状态 ↗</a>`;
  return;
 }
 el('learning-start').textContent='开始 / 继续运行';
 const c=s.curriculum||{},p=s.registration_progress||{};el('learning-start').disabled=['queued','running'].includes(s.status);
 const h=s.native_head_pilot;
 const live=s.native_training_progress;
 const refinement=s.native_head_refinement, rr=refinement?.report||{}, rp=refinement?.progress;
 const refinementNames={full_heldout_prior:'原适配器：原文回归',synthetic_prior:'原适配器：格式变化测试',full_heldout_candidate:'新适配器：原文回归',synthetic_candidate:'新适配器：格式变化测试','repeat-prior-1':'原适配器：第 1 次复测','repeat-prior-2':'原适配器：第 2 次复测','repeat-prior-3':'原适配器：第 3 次复测','repeat-candidate-1':'新适配器：第 1 次复测','repeat-candidate-2':'新适配器：第 2 次复测','repeat-candidate-3':'新适配器：第 3 次复测'};
 const refinementStage={stale_artifact:'适配产物已改变，原验证记录失效',trained_not_evaluated:'参数训练完成，等待完整回答验证',evaluating_full_answers:'正在验证完整回答',validated_source_candidate:'完整回答验证通过，候选未默认启用',retained_prior_after_validation:'完整回答未达到采用条件，保留原版本'};
 const refinementHtml=refinement?`<section><h3>本轮原文输出适配训练</h3><p>${escape(refinementStage[refinement.stage]||refinement.stage)}。训练 ${escape(rr.train_pages)} 页 / 内部验证 ${escape(rr.dev_pages?.length)} 页；新增 ${escape(rr.new_trainable_parameters)} 个可训练参数。训练 ${escape(rr.steps)} 步，由内部验证选用第 ${escape(rr.selected_step)} 步。</p>${rp?`<p>${escape(refinementNames[rp.stage]||rp.stage)}：${escape(rp.done)} / ${escape(rp.total)} 页。</p>`:''}${rr.full_heldout?`<p>原文回归：旧版 ${escape(rr.full_heldout.prior?.correct_points)} / ${escape(rr.full_heldout.prior?.expected_points)} 点，新版 ${escape(rr.full_heldout.adapter?.correct_points)} / ${escape(rr.full_heldout.adapter?.expected_points)} 点；字段结构合格页 ${escape(rr.full_heldout.prior?.schema_valid_pages)} → ${escape(rr.full_heldout.adapter?.schema_valid_pages)}。${escape(rr.promotion_reason)}</p>`:''}<p>这是原文读取与输出格式训练；主体权重未变更，也未训练 CAD 点位。817-07781 为重复回归项目，不是首次盲测。</p></section>`:'';
 const liveHtml=live?`<p>原文输出训练进度：${escape(live.example_done)} / ${escape(live.examples_total)} 页，已采集 ${escape(live.samples_done)} 个下一词位置。</p>`:'';
 const expanded=s.native_head_expanded;
 const expandedHtml=expanded?`<p><strong>第二轮全量原文适配</strong>：${expanded.stage==='feature_capture'?`正在采集原生特征，已采集 ${escape(expanded.samples_done)} 个下一词位置（训练 ${escape(expanded.train_samples_done)} / 测试 ${escape(expanded.test_samples_done)}）`:expanded.stage==='rejected_after_validation'?`已训练，独立项目坐标 ${escape(expanded.report?.full_heldout?.adapter?.correct_points)} / ${escape(expanded.report?.full_heldout?.adapter?.expected_points)}，低于第一轮，本轮未启用`:`已训练 ${escape(expanded.report?.parameters)} 个参数、${escape(expanded.report?.steps)} 步；训练 ${escape(expanded.report?.train_samples)} / 测试 ${escape(expanded.report?.test_samples)} 个下一词位置，尚未默认启用`}。</p>`:'';
 const pilot=h?`<p><strong>Strata 原文坐标输出适配试验</strong>：已训练 ${escape(h.parameters)} 个参数、${escape(h.steps)} 步。主体权重未改动；适配器${h.deployed?'已启用':'未默认启用'}。</p><p>独立项目 ${escape(h.test_samples)} 个下一词测试位置，正确率 ${(100*h.test_top1_before).toFixed(1)}% → ${(100*h.test_top1_after).toFixed(1)}%。这是小样本输出试验，不能作为 CAD 配准或 FAI 放置精度结论。</p>`:'';
 el('learning-state').innerHTML=`<p>${escape(states[s.status]||s.status)}${s.source_stale?' · 源文件已改变，需重新运行':''}</p><p>${escape(p.message||'运行后将逐组检查程序和坐标系')} ${p.total?escape(`${p.done} / ${p.total}`):''}</p><table><thead><tr><th>步骤</th><th>状态</th></tr></thead><tbody>${Object.entries(s.phases||{}).map(([k,v])=>`<tr><td>${escape(phaseNames[k]||k)}</td><td>${escape(states[v]||v)}</td></tr>`).join('')}</tbody></table><p>训练样本 ${escape(c.train_count??0)} 条 / 独立项目测试样本 ${escape(c.test_count??0)} 条。原文读取样本与CAD位置标签分开校验；不能用不合格配准作为正确位置。</p><table><thead><tr><th>项目</th><th>原文样本</th></tr></thead><tbody>${Object.entries(c.by_project||{}).map(([k,v])=>`<tr><td>${escape(k)}</td><td>${escape(v)}</td></tr>`).join('')}</tbody></table>${pilot}${expandedHtml}${refinementHtml}${liveHtml}<p>完整模型与几何训练：${s.weight_training_started?'已启动':'未启动'}；模型权重：${s.weights_changed?'已有变更':'未改动'}。</p><p>${escape((s.preflight?.reasons||[]).join('；'))}</p>${s.source_adapter_error?`<p>${escape(s.source_adapter_error)}</p>`:''}${s.error?`<p>${escape(s.error)}</p>`:''}<a href="/api/viewer/projects/${encodeURIComponent(pid)}/automatic-learning" download="automatic-learning.json">下载运行记录 JSON</a>`;
}
async function refresh(){
 if(!pid||!dialog.open)return;
 try{const response=await fetch(`/api/viewer/projects/${pid}/automatic-learning`);const s=await response.json();if(!response.ok)throw Error(s.detail||'读取失败');render(s);if(['queued','running'].includes(s.status)||(s.cloud_training?s.cloud_training.pipeline?.status==='running':s.native_head_expanded?.stage==='feature_capture'||s.native_head_refinement?.stage==='evaluating_full_answers'))timer=setTimeout(refresh,2500);}
 catch(e){el('learning-state').textContent='读取运行状态失败：'+e.message;}
}
open.onclick=()=>{pid=el('projects').value;if(!pid)return;clearTimeout(timer);dialog.showModal();refresh();};
el('learning-close').onclick=()=>{clearTimeout(timer);dialog.close();};
el('learning-start').onclick=async()=>{el('learning-start').disabled=true;try{const response=await fetch(`/api/viewer/projects/${pid}/automatic-learning`,{method:'POST'});const s=await response.json();if(!response.ok)throw Error(s.detail||'启动失败');render(s);timer=setTimeout(refresh,1000);}catch(e){el('learning-state').textContent='启动失败：'+e.message;el('learning-start').disabled=false;}};
dialog.addEventListener('close',()=>clearTimeout(timer));
window.addEventListener('oracle-project-loaded',()=>dialog.close());
