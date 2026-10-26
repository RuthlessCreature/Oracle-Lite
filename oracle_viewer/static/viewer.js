import * as THREE from 'three';
import {wording,errorMessage} from './copy.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {RoomEnvironment} from './vendor/RoomEnvironment.js';
const $=id=>document.getElementById(id), base='/api/viewer';
let project=null, number=null, detail=null, docs=[], selected=new Set(), faces=new Map(), meshes=[], requestId=0, planText='', mode='source', busy=false;
const scene=new THREE.Scene(), group=new THREE.Group();scene.add(group);
const camera=new THREE.PerspectiveCamera(40,1,.01,100000);camera.up.set(0,0,1);
let renderer,controls,renderDirty=true,contextLost=false,renderedFrames=0,lastFrameTime=0;
const renderPixelBudget=1500000;
function invalidateRender(){renderDirty=true;}
function installEnvironment(){const old=scene.environment,room=new RoomEnvironment(),pmrem=new THREE.PMREMGenerator(renderer);scene.environment=pmrem.fromScene(room,.04).texture;room.dispose();pmrem.dispose();old?.dispose();}
try{renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:'default'});renderer.setPixelRatio(1);renderer.setClearColor(0x172737,0);renderer.localClippingEnabled=true;renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.85;installEnvironment();$('canvas-host').prepend(renderer.domElement);controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.addEventListener('change',invalidateRender);
renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();contextLost=true;notice('三维显示已中断。请关闭不需要的页面后刷新；测量方法和方案仍可查看。');});
renderer.domElement.addEventListener('webglcontextrestored',()=>{contextLost=false;installEnvironment();invalidateRender();});}
catch(e){notice('浏览器无法显示三维模型，请检查硬件加速设置；原始资料和测量方法仍可查看。');}
scene.add(new THREE.HemisphereLight(0xf0f6ff,0x526474,.65));
const light=new THREE.DirectionalLight(0xffffff,1.5);light.position.set(120,-150,260);scene.add(light);
const fill=new THREE.DirectionalLight(0xa2caff,.6);fill.position.set(-180,100,80);scene.add(fill);
const clipPlane=new THREE.Plane(new THREE.Vector3(0,0,-1),0);
let box=null,transparent=false;
const selectedPoints=new Map();let markerGroups=[];
const annotationOf=item=>item?.annotation;
const anchorsOf=item=>item?.annotation?.origin==='local_model_candidate'&&item?.annotation?.status==='auto_marked'?item.annotation.anchors||[]:[];
const marked=item=>item?.annotation?.origin==='local_model_candidate'&&item?.annotation?.status==='auto_marked'&&!!item?.annotation?.anchors?.length;
function printedSpecification(heading){const value=(heading||'').replace(/^\s*FAI\s+\d+(?:\s*[~–-]\s*\d+)?(?:\s*\(SPC\s+[^)]*\))?\s*/i,'').trim();return /^[-+]?\d+(?:\.\d+)?\s+(?:±\s*\d+(?:\.\d+)?|[+-]\d+(?:\.\d+)?(?:\s*\/\s*[+-]?\d+(?:\.\d+)?)?)(?:\s*(?:mm|deg|°))?$/i.test(value)?value:'';}
function measurementLabel(item){const feature=item.measurement_features?.[0]||item.features?.[0]||item.headings?.[0]||'未明确';const specs=item.specifications||[];const indicators=item.measurement_indicators||[];const metrics=item.measurement_metrics||[];const fmt=s=>`${s.nominal} ${s.upper}/${s.lower}${s.unit?` ${s.unit}`:''}`;const fmtMetric=m=>[m.mean,m.minimum?`MIN ${m.minimum}`:'',m.allowed_range&&m.allowed_range!=='-'?`ALLOW ${m.allowed_range}`:'',m.unit].filter(Boolean).join(' · ');const metricText=metrics.map(fmtMetric).join('; '),printed=(item.headings||[]).map(printedSpecification).filter(Boolean).join(' · ');return {feature,spec:specs.length?specs.map(fmt).join(' · '):metricText||printed||'SEE MSOP',full:specs.length?specs.map(fmt).join('; '):metricText||printed||'SEE MSOP',symbols:indicators.map(x=>x.symbol).filter(Boolean).join(' · '),indicatorDetails:indicators.map(x=>`${x.symbol} ${x.label} (${x.source||''})`).join('; ')};}
function markerTitle(item,pending){const m=measurementLabel(item),state=item.annotation?.status==='auto_marked'?'自动标记':pending?'待复核':'已复核';return `FAI ${item.number}${m.symbols?` · ${m.symbols}`:''} · ${m.feature} · ${m.full}${m.indicatorDetails?` · ${m.indicatorDetails}`:''} · P.${item.pages.join(',')}${item.probes?.length?` · 测头 ${item.probes.join(' / ')}`:''}${item.annotation?.origin==='local_model_candidate'?` · 置信度 ${({low:'低',medium:'中',high:'高'})[item.annotation.confidence]||'未明确'} (模型估计)`:''} · ${state}`;}
function updateCounts(){$('location-count').textContent=`${project.marked_count??project.located_count} / ${project.fai_count} 已标记 · ${project.reviewed_count||0} 位置已复核`;}

function notice(text){$('notice').textContent=errorMessage(text);$('notice').hidden=false;clearTimeout(notice.timer);notice.timer=setTimeout(()=>$('notice').hidden=true,6500);}
async function api(path,options={}){const r=await fetch(path.startsWith('/api/')?path:base+path,options);const data=await r.json();if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;}
const post=(path,data)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function projectPath(s=''){return `/projects/${project.id}${s}`;}
function setMode(value){invalidateRender();mode=value;$('three-area').hidden=value==='source';$('source-area').hidden=value==='3d';$('stage-views').classList.toggle('compare',value==='compare');for(const id of ['3d','source','compare'])$('view-'+id).classList.toggle('active',value===id);resize();}
function setTab(value){$('method-panel').hidden=value!=='method';$('plan-panel').hidden=value!=='plan';$('method-tab').classList.toggle('active',value==='method');$('plan-tab').classList.toggle('active',value==='plan');}
function resize(){if(!renderer)return;const host=$('canvas-host');const w=host.clientWidth,h=host.clientHeight;if(w&&h){renderer.setPixelRatio(Math.min(devicePixelRatio,1,Math.sqrt(renderPixelBudget/(w*h))));renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();invalidateRender();}}
new ResizeObserver(resize).observe($('canvas-host'));
function updateCameraRange(){if(!box||!controls)return;const radius=box.getBoundingSphere(new THREE.Sphere()).radius,distance=camera.position.distanceTo(controls.target),near=Math.max(Math.min(radius*.01,distance*.05),.001),far=Math.max(distance+radius*3,near*100);if(Math.abs(camera.near-near)>camera.near*.05||Math.abs(camera.far-far)>camera.far*.05){camera.near=near;camera.far=far;camera.updateProjectionMatrix();}}
function fit(direction=new THREE.Vector3(1,-1,.8)){if(!box||!controls)return;const damping=controls.enableDamping;controls.enableDamping=false;controls.update();const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());const radius=Math.max(size.x,size.y,size.z)*.5;const distance=radius/Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*1.3/Math.min(camera.aspect,1);controls.target.copy(center);camera.position.copy(center).add(direction.normalize().multiplyScalar(distance));updateCameraRange();controls.update();controls.enableDamping=damping;}
function paint(){invalidateRender();const linked=new Set(marked(detail?.item)?detail.item.annotation.face_ids||[]:[]),color=new THREE.Color();for(const m of meshes){const colors=m.geometry.getAttribute('color');for(const range of m.userData.ranges){color.set(selected.has(range.id)?0xf0bb62:linked.has(range.id)?0x20b8b2:0x8f9da8);for(let i=range.vertexStart;i<range.vertexEnd;i++)colors.setXYZ(i,color.r,color.g,color.b);}colors.needsUpdate=true;m.material.opacity=transparent ? .32 : 1;m.material.transparent=transparent;m.material.depthWrite=!transparent;m.material.clippingPlanes=$('clip').checked?[clipPlane]:[];}updateBindingControls();}
function faceAtHit(hit){const tri=hit.faceIndex;let lo=0,hi=hit.object.userData.ranges.length-1;while(lo<=hi){const mid=(lo+hi)>>1,r=hit.object.userData.ranges[mid];if(tri<r.triangleStart)hi=mid-1;else if(tri>=r.triangleEnd)lo=mid+1;else return r.id;}return null;}
function makeBodyMesh(bodyFaces){let vertices=0,indexCount=0;for(const f of bodyFaces){vertices+=f.positions.length/3;indexCount+=f.indices.length;}const xyz=new Float32Array(vertices*3),colors=new Float32Array(vertices*3),index=new Uint32Array(indexCount),ranges=[];let vertexOffset=0,indexOffset=0;for(const f of bodyFaces){xyz.set(f.positions,vertexOffset*3);for(let i=0;i<f.indices.length;i++)index[indexOffset+i]=f.indices[i]+vertexOffset;ranges.push({id:f.id,vertexStart:vertexOffset,vertexEnd:vertexOffset+f.positions.length/3,triangleStart:indexOffset/3,triangleEnd:(indexOffset+f.indices.length)/3});vertexOffset+=f.positions.length/3;indexOffset+=f.indices.length;}const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(xyz,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));geometry.setIndex(new THREE.BufferAttribute(index,1));geometry.computeVertexNormals();geometry.computeBoundingSphere();const material=new THREE.MeshStandardMaterial({color:0xffffff,vertexColors:true,roughness:.52,metalness:.28,envMapIntensity:.45,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:meshes.length*2,side:THREE.DoubleSide});const mesh=new THREE.Mesh(geometry,material);mesh.userData.ranges=ranges;return mesh;}
function updateBindingControls(){$('focus-fai').disabled=!anchorsOf(detail?.item).length;$('picked').textContent=selected.size?`已选择 ${selected.size} 个 CAD 面用于查看`:'FAI 标记由本地模型自动完成；人工选点不计入覆盖';}
function renderMarkers(){invalidateRender();const container=$('markers');container.replaceChildren();markerGroups=[];const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('aria-hidden','true');container.append(svg);for(const item of project?.items||[]){const anchors=anchorsOf(item);if(!anchors.length)continue;const pending=false;const cls=pending?' pending-review':'';const m=measurementLabel(item);const button=el('button','marker'+cls+(number===item.number?' active':''));button.dataset.fai=item.number;button.append(el('strong','marker-fai',`FAI ${item.number}${m.symbols?`  ${m.symbols}`:''}`),el('span','marker-feature',m.feature),el('span','marker-spec',m.spec));button.title=markerTitle(item,pending);button.hidden=true;button.onclick=async()=>{if(button.dragged)return;await selectFAI(item.number);focusFAI();};const graphics=anchors.map(()=>{const line=document.createElementNS(svg.namespaceURI,'line'),pin=document.createElementNS(svg.namespaceURI,'circle');line.setAttribute('class','marker-leader'+cls);pin.setAttribute('class','marker-pin'+cls);pin.setAttribute('r','3.5');svg.append(line,pin);return {line,pin};});const entry={item,anchors,button,graphics};markerGroups.push(entry);enableMarkerDrag(entry);container.append(button);}}
function animate(time=0){requestAnimationFrame(animate);if(!renderer||contextLost||document.hidden||mode==='source'||!$('workflow-panel').hidden)return;controls.update();if(!renderDirty||time-lastFrameTime<1000/30)return;renderDirty=false;lastFrameTime=time;updateCameraRange();projectMarkers();const css=renderer.getSize(new THREE.Vector2());renderer.setViewport(0,0,css.x,css.y);renderer.setScissorTest(false);
if(previewOn){const aspect=(previewCamera.right-previewCamera.left)/(previewCamera.top-previewCamera.bottom),pw=Math.min(css.x,css.y*aspect),ph=pw/aspect;renderer.clear();renderer.setViewport((css.x-pw)/2,(css.y-ph)/2,pw,ph);renderer.setScissor((css.x-pw)/2,(css.y-ph)/2,pw,ph);renderer.setScissorTest(true);}renderer.render(scene,previewOn?previewCamera:camera);renderer.setScissorTest(false);renderedFrames++;}animate();
document.addEventListener('visibilitychange',invalidateRender);
new MutationObserver(invalidateRender).observe($('workflow-panel'),{attributes:true,attributeFilter:['hidden']});
if(renderer){let down=null;renderer.domElement.addEventListener('pointerdown',e=>down={x:e.clientX,y:e.clientY});renderer.domElement.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down.x,e.clientY-down.y)>5||e.button!==0)return;const r=renderer.domElement.getBoundingClientRect(),p=new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),ray=new THREE.Raycaster();ray.setFromCamera(p,camera);const hit=ray.intersectObjects(meshes).find(h=>!$('clip').checked||clipPlane.distanceToPoint(h.point)>=0);if(!e.shiftKey)selected.clear();selectedPoints.clear();if(hit){const id=faceAtHit(hit);if(selected.has(id)){selected.delete(id);selectedPoints.delete(id);}else{selected.add(id);selectedPoints.set(id,hit.point.toArray());}}paint();});}
async function loadGeometry(){for(const mesh of meshes){group.remove(mesh);mesh.geometry.dispose();mesh.material.dispose();}meshes=[];faces.clear();selected.clear();selectedPoints.clear();box=null;const model=project.model,available=['ready','preview'].includes(model?.status),imported=!!model?.filename,unavailable=imported&&!available;$('geometry-empty').hidden=available&&!!renderer;$('geometry-empty-title').textContent=unavailable?'CAD 已导入，但暂时无法显示':'请导入三维模型';$('geometry-reason').textContent=unavailable?`文件“${model.filename}”已保存，但当前解码器无法读取其几何。原始 CAD 文件仍在项目资料中。`:model?.reason||'请导入STEP或X_T等三维CAD文件。';$('geometry-error-details').hidden=!(unavailable&&model?.error);$('geometry-error').textContent=model?.error||'';$('import-step').textContent=unavailable?'更换 CAD 文件':'导入 CAD 模型';$('geometry-footnote').textContent=unavailable?'原始 CAD 已保存在项目中；修复解码器或更换文件后即可继续查看测量位置。FAI 清单和原始资料仍可使用。':'导入模型后可查看测量位置。未标记的FAI仍保留在左侧清单中。';$('geometry-state').textContent=available?`${model.source_format||'CAD'} · ${model.status==='preview'?'几何预览':'CAD模型'} · mm`:unavailable?'CAD 已导入 · 解码失败':'未导入三维模型';if(available&&renderer){const pid=project.id;$('stage-info').textContent='正在加载零件模型…';meshAbortController=new AbortController();let data;try{data=await api(projectPath('/mesh'),{signal:meshAbortController.signal});}catch(e){if(e.name==='AbortError')return;throw e;}if(project.id!==pid)return;const bodies=new Map();for(const face of data.faces){const {positions,indices,...metadata}=face;faces.set(face.id,metadata);if(!bodies.has(face.body))bodies.set(face.body,[]);bodies.get(face.body).push(face);}for(const bodyFaces of bodies.values()){const mesh=makeBodyMesh(bodyFaces);group.add(mesh);meshes.push(mesh);}box=new THREE.Box3().setFromObject(group);clipPlane.constant=(box.min.z+box.max.z)/2;fit();const size=box.getSize(new THREE.Vector3());$('stage-info').textContent=`${data.body_count} 个实例 · ${data.face_count} 个面 · ${size.x.toFixed(2)} × ${size.y.toFixed(2)} × ${size.z.toFixed(2)} mm · ${data.metrology_ready===false?'模型精度待验证':'仅用于显示和位置核对'}`;paint();}else $('stage-info').textContent=unavailable?`CAD 文件已导入，Parasolid schema 解码未通过；原始文件已保留。`: '可先查看图纸和MSOP，导入CAD后再核对三维位置';renderMarkers();updateBindingControls();}
async function loadProjects(preferred){const all=await api('/projects'),list=all.filter(p=>p.drawing_no!=='817-07781');$('projects').replaceChildren(...list.map(p=>{const o=el('option','',`${p.drawing_no||p.title} · Rev ${p.revision||'—'}`);o.value=p.id;return o;}));if(list.length){const defaultProject=list.find(p=>p.drawing_no==='613-13599')||list[0];await loadProject(preferred&&list.some(p=>p.id===preferred)?preferred:defaultProject.id);}else{$('project-title').textContent='新建项目并导入资料';notice('新建项目后导入 MSOP 和 CAD。');}}
function renderProgramFiles(){
 const files=project?.reference_files||[],programs=project?.measurement_programs||[];
 const button=$('program-files');button.hidden=!files.length;button.textContent=`测量程序 / XML · ${files.filter(f=>f.filename.toLowerCase().endsWith('.xml')).length}`;
 $('program-warning').replaceChildren(...(project?.source_warnings||[]).map(w=>el('div','program-warning',w)));
 const summaries=$('program-summary');summaries.replaceChildren();
 for(const p of programs){const card=el('article','program-card'),matched=p.matched_msop_fai_numbers||[],unmatched=p.unmatched_msop_fai_numbers||[];card.append(el('h3','',p.name),el('p','',`${p.program_type} · ${p.fai_group_count} 个 FAI 编号组 · ${p.point_count} 个坐标点（${p.xyz_point_count} 个含 XYZ）`),el('p','',`MSOP编号关联：${matched.length}/${p.fai_group_count}${unmatched.length?` · XML 中未出现在 MSOP 清单：${unmatched.map(n=>'FAI '+n).join(', ')}`:''}`),el('p','',`工位/程序段：${(p.sections||[]).join(' / ')} · XML 内部标识：${p.drawing_id||'未识别'}`));summaries.append(card);}
 const list=$('program-files-list');list.replaceChildren();
 for(const f of files){const row=el('div','program-file-row'),info=el('div','',`${f.filename} · ${f.description||f.category}`),a=el('a','button','下载原始文件');a.href=base+projectPath(`/reference-files/${f.id}/source`);a.download=f.filename;row.append(info,a);list.append(row);}
}
function openProgramFiles(){renderProgramFiles();$('program-dialog').showModal();}
let projectLoadEpoch=0,meshAbortController=null;
async function loadProject(id){const epoch=++projectLoadEpoch;meshAbortController?.abort();clearPlanningScene();window.dispatchEvent(new Event('oracle-project-loaded'));requestId++;const incoming=await api(`/projects/${id}`);if(epoch!==projectLoadEpoch)return;project=incoming;$('marking-status').hidden=true;$('projects').value=id;number=null;detail=null;planText='';$('plan-result').textContent='';$('project-title').textContent=project.title;$('revision').textContent=`图纸 ${project.drawing_no||'未指定'} · 版本 ${project.revision||'未指定'}`;$('count').textContent=project.fai_count;updateCounts();$('coverage').textContent=wording(project.coverage);docs=project.documents.filter(d=>d.role!=='cad');$('source-doc').replaceChildren(...docs.map(d=>{const o=el('option','',(d.role==='msop'?'MSOP · ':'图纸 · ')+d.filename);o.value=d.id;return o;}));$('source-empty').hidden=!!docs.length;$('source-image').hidden=!docs.length;renderProgramFiles();renderItems();await loadGeometry();if(epoch!==projectLoadEpoch)return;const requestedFAI=new URLSearchParams(location.search).get('project')===id?Number(new URLSearchParams(location.search).get('fai')):null;const first=project.items.find(i=>i.number===requestedFAI)||project.items.find(i=>anchorsOf(i).length)||project.items[0];if(first)await selectFAI(first.number);else{$('fai-title').textContent='请导入测量SOP';$('facts').replaceChildren();$('methods').replaceChildren();$('dependencies').replaceChildren();$('warnings').replaceChildren();if(docs.length)showPage(docs[0].id,1);}if(epoch!==projectLoadEpoch)return;setMode(['ready','preview'].includes(project.model?.status)?'3d':'source');void ensureLocalModelMarking(id);}
const markingPollProjects=new Set();
function showMarkingStatus(id,state){
  if(project?.id!==id)return;
  const area=$('marking-status');
  area.hidden=false;
  const labels={queued:'等待自动标记',running:'正在自动标记',completed:'自动标记完成',partial:'本轮定位已结束，仍有未标记项',blocked:'自动标记需要补充证据',failed:'自动标记失败'};
  const c=state.coverage;
  const coverage=c?`已标记 ${c.marked}/${c.total} 项`:'';
  const phases={method_classification:'正在识别测量方法与定位依据',registration:'正在核对工件坐标系',registration_model_selection:'正在由本地模型核对配准位置'};
  const classification=state.classification_progress;
  const phase=phases[state.progress?.stage];
  const message=state.error?errorMessage(state.error):(state.status==='running'?(phase||state.progress?.message):state.message)||state.progress?.message||'';
  const classificationText=classification?`测量方法已核对 ${classification.done}/${classification.total} 项`:'';
  const comparisonText=state.progress?.comparison_total?`配准候选已核对 ${state.progress.comparison_done??0}/${state.progress.comparison_total} 组`:'';
  area.title=[labels[state.status]||'自动标记状态',message,classificationText,comparisonText,coverage].filter(Boolean).join('\n');
  const terminal=['partial','blocked'].includes(state.status);
  area.textContent=[labels[state.status]||'自动标记状态',terminal?'':classificationText||comparisonText||message,coverage].filter(Boolean).join(' · ');
  if(terminal&&c?.unmarked){const button=el('button','','继续处理未标记项');button.type='button';button.onclick=()=>void ensureLocalModelMarking(id,true);area.append(' ',button);}
}

async function ensureLocalModelMarking(id,retry=false){
  if(markingPollProjects.has(id))return;
  markingPollProjects.add(id);
  showMarkingStatus(id,{status:"queued",message:"正在检查本地模型和项目资料"});
  try{
    let classificationProgress=null,lastClassificationRead=0;
    let state=await post(`/projects/${id}/local-model-marking${retry?'?retry=true':''}`,{});
    showMarkingStatus(id,state);
    if(state.status==='queued'||state.status==='running'){
      notice('正在自动解析 MSOP 点位并生成三维标记。');
      do{
        await new Promise(r=>setTimeout(r,2000));
        state=await api(`/projects/${id}/local-model-marking`);
        if(project?.id!==id)return;
        if(state.progress?.stage==='method_classification'&&Date.now()-lastClassificationRead>10000){
          try{const c=await api(`/projects/${id}/fai-strategies`);classificationProgress={done:c.done??c.items?.length??0,total:c.total??project.fai_count};lastClassificationRead=Date.now();}catch{}
        }else if(state.progress?.stage!=='method_classification')classificationProgress=null;
        showMarkingStatus(id,{...state,classification_progress:classificationProgress});
        if(state.coverage&&state.coverage.marked!==(project.marked_count??project.located_count)){
          const updated=await api(`/projects/${id}`);
          if(project?.id!==id)return;
          project=updated;updateCounts();renderItems();renderMarkers();
          if(number)await selectFAI(number);
        }
        if(project?.id===id&&state.status==='running'&&state.progress)notice(`${state.progress.message} · 已处理 ${state.progress.done}/${state.total??'—'} 项`);
      }while(state.status==='queued'||state.status==='running');
    }
    if(project?.id!==id)return;
    project=await api(`/projects/${id}`);
    updateCounts();renderItems();renderMarkers();
    if(number)await selectFAI(number);
    const c=state.coverage;
    const coverage=c?`已标记 ${c.marked}/${c.total} 项 · 未标记 ${c.unmarked} 项`:'未取得标记覆盖统计';
    if(state.status==='completed')notice(`${coverage}。本地模型已自动完成全部 FAI 的位置标记；置信度显示在各测量项旁。`);
    else if(state.status==='blocked'||state.status==='partial'){const reason=(state.message||'').replace(/^已标记\s*\d+\/\d+\s*项[；;]\s*未标记\s*\d+\s*项[，,]?/,'').trim();notice(`${coverage}。${reason||'部分位置仍未能由自动流程确定；当前覆盖不完整，继续作为算法缺口处理。'}`);}
    else notice(`自动标记未完成：${state.error||state.message||'未知错误'} · ${coverage}`);
  }catch(e){if(project?.id===id){showMarkingStatus(id,{status:'failed',error:e.message});notice(`无法开始自动标记：${e.message}`);}}
  finally{markingPollProjects.delete(id);}
}
function renderItems(){const q=$('search').value.toLowerCase().replace(/\s+/g,''),filter=$('filter').value,requestedFAI=/^(?:fai)?0*(\d+)$/i.exec(q);const items=(project?.items||[]).filter(i=>{const hay=JSON.stringify(i).toLowerCase().replace(/\s+/g,'');return (!q||(requestedFAI?i.number===Number(requestedFAI[1]):hay.includes(q)))&&(filter==='all'||filter==='located'&&marked(i)||filter==='pending'&&!marked(i)||filter==='review'&&i.annotation?.status==='pending_review'||filter==='conflict'&&i.warnings.length);});$('items').replaceChildren();for(const item of items){const b=el('button','item'+(number===item.number?' active':''));b.dataset.fai=item.number;let label=marked(item)?'本地模型自动标记':item.annotation?.status==='stale'?'原位置待特征复核':'未标记';if(item.annotation?.origin==='local_model_candidate')label+=` · 置信度：${({low:'低',medium:'中',high:'高'})[item.annotation.confidence]||'未给出'}`;b.append(el('strong','',`FAI ${item.number}`),el('span','state'+(marked(item)?' located':''),label+(item.warnings.length?' · 资料需核对':'')));const spec=item.specifications[0],metric=item.measurement_metrics?.[0];if(spec)b.append(el('div','spec',`${spec.nominal}  ${spec.upper}/${spec.lower} · 单位见原文`));else if(metric)b.append(el('div','spec',`${item.measurement_indicators?.[0]?.symbol||''} ${measurementLabel(item).spec}`));b.append(el('small','',item.measurement_features?.[0]||item.features[0]||item.headings[0]),el('small','',`第 ${item.pages.join('、')} 页 · ${item.probes.join(' / ')||'测头见原文'}`));b.onclick=()=>selectFAI(item.number);$('items').append(b);}if(!items.length)$('items').append(el('p','empty-list','没有找到测量项。请调整搜索或筛选；若尚未导入MSOP，请先导入资料。'));}
function showPage(docId,page){const doc=docs.find(d=>d.id===docId);if(!doc)return;page=Math.max(1,Math.min(doc.page_count,Number(page)||1));$('source-doc').value=docId;$('source-page').value=page;$('source-page').max=doc.page_count;$('page-count').textContent='/ '+doc.page_count;$('source-image').src=`${base}${projectPath(`/documents/${docId}/pages/${page}.png`)}`;$('download-source').href=base+projectPath(`/documents/${docId}/source`);}
function fact(label,value){const f=el('div','fact');f.append(el('span','',label),el('strong','',value||'原文未明确'));return f;}
async function selectFAI(n){const pid=project.id,req=++requestId;const d=await api(`/projects/${pid}/fai/${n}`);if(req!==requestId||project.id!==pid)return;number=n;detail=d;selected.clear();selectedPoints.clear();$('fai-title').textContent='FAI '+n;$('binding-status').textContent=marked(d.item)?'本地模型已自动标记':d.item.annotation?.status==='stale'?'位置已退出覆盖 · 等待自动复核':'尚未获得本地模型标记';$('binding-status').className='badge'+(marked(d.item)?' located':'');$('binding-status').title='置信度是模型估计；自动位置标记不等同计量结果。';if(d.item.annotation?.origin==='local_model_candidate')$('binding-status').textContent+=` · 置信度：${({low:'低',medium:'中',high:'高'})[d.item.annotation.confidence]||'未给出'}`;$('warnings').replaceChildren(...d.item.warnings.map(w=>el('div','warning',w)));const grid=el('div','fact-grid');for(const method of d.methods){const spec=method.specification;const indicators=(method.measurement_indicators||[]).map(x=>`${x.symbol} ${x.label}`).join(' · ');grid.append(fact(`FAI ${n} · P.${method.page}`,method.heading||method.features.join(' / ')||'请查看原文'),fact('测量指标',indicators||'原文未明确'),fact('名义值 / 公差',spec?`${spec.nominal}  ${spec.upper}/${spec.lower}${spec.unit?` ${spec.unit}`:''}`:(printedSpecification(method.heading)||'查看原页标注')),fact('测量坐标系',method.dimension_system),fact('测头 / 程序',[method.probe,method.program].filter(Boolean).join(' / ')));}for(const metric of d.item.measurement_metrics||[]){const ind=(d.item.measurement_indicators||[]).map(x=>`${x.symbol} ${x.label}`).join(' · ');const values=[metric.mean,metric.minimum?`MIN ${metric.minimum}`:'',metric.allowed_range&&metric.allowed_range!=='-'?`ALLOW ${metric.allowed_range}`:'',metric.unit].filter(Boolean).join(' · ');grid.append(fact('测量类型',ind||'未明确'),fact('位置 / 特征',metric.location||d.item.measurement_features?.[0]||'请查看原文'),fact('要求 / 单位',values),fact('资料来源',metric.source||d.item.visual_evidence||'MSOP页面提取'));}$('facts').replaceChildren(grid);$('methods').replaceChildren();for(const method of d.methods){const card=el('div','method-card'),b=el('button','',`第 ${method.page} 页 ↗`);b.onclick=()=>{showPage(d.item.document_id,method.page);setMode(mode==='compare'?'compare':'source');};card.append(b,el('h3','',method.heading),el('pre','',method.text.trim()));$('methods').append(card);}$('dependencies').replaceChildren();for(const dep of d.dependencies.filter(p=>!d.item.pages.includes(p.page))){const b=el('button','dependency');b.append(el('span','',`${dep.reference||dep.heading} · ${dep.program||'程序见原文'}`),el('span','',`p.${dep.page}`));b.onclick=()=>{showPage(d.item.document_id,dep.page);setMode(mode==='compare'?'compare':'source');};$('dependencies').append(b);}if(!$('dependencies').children.length)$('dependencies').append(el('p','empty-list','未识别到关联基准页，请对照MSOP核对。'));$('binding-info').textContent=marked(d.item)?'此位置由本地模型自动标记；置信度是模型自评，不代表计量验收。':d.item.annotation?.status==='stale'?'原位置缺少当前特征对应证据或存在冲突，已退出有效覆盖；由本地模型重新核对。':'尚无有效的本地模型自动标记。';if(d.item.annotation?.origin==='local_model_candidate'){const a=d.item.annotation;$('binding-info').textContent=`${a.anchors.length} 个标记点 · 本地模型结果 · ${a.status==='auto_marked'?'自动定位完成':a.status==='stale'?'等待本地模型重新定位':'未完成'}。置信度：${({low:'低',medium:'中',high:'高'})[a.confidence]||'未给出'}（模型自评，未经校准）。`; }const verification=d.item.annotation?.feature_verification;
if(verification?.status==='confirmed'){
 const names=[...new Set((verification.feature_links||[]).map(x=>x.source_name))];
 $('binding-info').textContent+=` 原文与CAD实例逐点复核已通过；共享特征：${names.join(' / ')||'原文无命名特征，保留逐点实体证据'}。`;
}
$('talker').href=`/?project=${pid}&fai=${n}`;const tasks=project.tasks.filter(t=>t.fai===n);const latest=tasks.at(-1);showTask(latest);renderItems();paint();renderMarkers();if(docs.length)showPage(d.item.document_id,d.item.pages[0]);}
function showTask(task){planText=task?.result?.text||'';$('plan-result').textContent=(task?.result?.validation_warnings?.length?'需核对：\n'+task.result.validation_warnings.join('\n')+'\n\n':'')+planText;$('download-plan').hidden=!planText;$('task-status').textContent=task?task.status==='complete'?`${task.stale?'资料已更新，请重新编制方案 · ':''}初步方案 · ${task.result.engine==='shared-local-model'?'本地生成':'依据原文计算'}${task.result.truncated?' · 输出达到长度上限，请继续细化':''}`:task.status==='failed'?task.error:'任务已提交，正在编制方案':'';}
async function refresh(){const n=number;project=await api(projectPath());updateCounts();renderItems();if(n)await selectFAI(n);}
$('search').oninput=renderItems;$('filter').onchange=renderItems;$('projects').onchange=()=>loadProject($('projects').value).catch(e=>notice(e.message));$('view-3d').onclick=()=>setMode('3d');$('view-compare').onclick=()=>setMode('compare');$('marker-mode').onchange=renderMarkers;$('focus-fai').onclick=focusFAI;$('source-zoom').onchange=()=>{$('source-image').style.width=$('source-zoom').value+'%';$('source-image').style.maxWidth='none';};$('view-source').onclick=()=>setMode('source');$('method-tab').onclick=()=>setTab('method');$('plan-tab').onclick=()=>setTab('plan');$('fit').onclick=()=>fit();$('top').onclick=()=>fit(new THREE.Vector3(0,0,1));$('front').onclick=()=>fit(new THREE.Vector3(0,-1,0));$('transparent').onclick=()=>{transparent=!transparent;paint();};$('clip').onchange=paint;$('clip-value').oninput=()=>{invalidateRender();if(box)clipPlane.constant=box.min.z+(box.max.z-box.min.z)*Number($('clip-value').value)/100;};$('source-doc').onchange=()=>showPage($('source-doc').value,1);$('source-page').onchange=()=>showPage($('source-doc').value,$('source-page').value);$('prev-page').onclick=()=>showPage($('source-doc').value,Number($('source-page').value)-1);$('next-page').onclick=()=>showPage($('source-doc').value,Number($('source-page').value)+1);
function chooseImport(role){if(!project)return notice('请先新建项目');if(role)$('import-role').value=role;$('file').accept=$('import-role').value==='cad'?'.step,.stp,.x_t,.x_b':'.pdf';$('file').click();}
$('import').onclick=()=>chooseImport();$('import-step').onclick=()=>chooseImport('cad');$('file').onchange=async()=>{const file=$('file').files[0];if(!file||!project)return;const pid=project.id;const fd=new FormData();fd.append('role',$('import-role').value);fd.append('file',file);$('import').disabled=true;notice('正在导入并解析资料…');try{await api(`/projects/${pid}/documents`,{method:'POST',body:fd});await loadProject(pid);notice('资料已导入。若更换了CAD或MSOP，请重新核对标记位置和方案。');}catch(e){notice(e.message);}finally{$('import').disabled=false;$('file').value='';}};
$('program-files').onclick=openProgramFiles;$('program-close').onclick=()=>$('program-dialog').close();$('new-project').onclick=async()=>{const title=prompt('项目名称');if(!title)return;const drawing_no=prompt('图纸编号（可留空）')||'',revision=prompt('图纸版本（可留空）')||'';try{const data=await post('/projects',{title,drawing_no,revision});await loadProjects(data.id);}catch(e){notice(e.message);}};
$('export').onclick=()=>{if(project)location.href=base+projectPath('/export');};

$('ask').onclick=async()=>{if(!number||busy)return;busy=true;$('ask').disabled=true;const pid=project.id,n=number;try{const task=await post(`/projects/${pid}/fai/${n}/ask`,{question:$('question').value});setTab('plan');$('task-status').textContent='任务已提交，正在编制方案';let current;do{await new Promise(r=>setTimeout(r,1200));current=await api(`/projects/${pid}/tasks/${task.id}`);if(project.id===pid&&number===n)showTask(current);}while(current.status==='queued'||current.status==='running');if(project.id===pid)await refresh();}catch(e){notice(e.message);}finally{busy=false;$('ask').disabled=false;}};
$('download-plan').onclick=()=>{const url=URL.createObjectURL(new Blob([planText],{type:'text/markdown;charset=utf-8'}));const a=el('a');a.href=url;a.download=`FAI${number}-AOI-draft.md`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
api('/api/status').then(s=>{const m=s.viewer_model||s;$('model-status').textContent=m.loaded?(m.inference_backend==='vllm-self-hosted'?'● 云端模型已连接':'● 本地模型已连接'):'○ 推理模型未就绪';$('model-status').title=`推理模型：${m.model_alias||'未提供'}\n运行后端：${m.inference_backend||'未提供'}`;}).catch(()=>$('model-status').textContent='无法连接本地服务');
loadProjects(new URLSearchParams(location.search).get('project')).catch(e=>notice(e.message));
// Read-only instrumentation for isolated application render regression tests.
if(new URLSearchParams(location.search).get('render-test')==='1')window.__oracleViewerDiagnostics=()=>({
  project:project?.id,mode,meshCount:meshes.length,faceCount:faces.size,
  drawCalls:renderer?.info.render.calls,triangles:renderer?.info.render.triangles,gpuResources:{...renderer?.info.memory},
  camera:camera.position.toArray(),cameraNear:camera.near,cameraFar:camera.far,target:controls?.target.toArray(),
  selected:[...selected],selectedPoints:[...selectedPoints],markers:markerGroups.map(g=>({fai:g.item.number,visible:!g.button.hidden,status:g.item.annotation?.status,anchors:g.anchors.map(a=>a.position),label:[g.button.style.left,g.button.style.top]})),clipped:$('clip').checked,clipPosition:clipPlane.constant,
  transparent,renderedFrames,contextLost,pixelRatio:renderer?.getPixelRatio(),drawingBuffer:renderer?.getDrawingBufferSize(new THREE.Vector2()).toArray(),gpuResources:renderer?{...renderer.info.memory,programs:renderer.info.programs.length}:null,retainedSourceVertexArrays:[...faces.values()].filter(f=>f.positions||f.indices).length,previewOn,previewFrame:previewOn?{position:previewCamera.position.toArray(),up:previewCamera.up.toArray(),forward:previewCamera.getWorldDirection(new THREE.Vector3()).toArray()}:null,simulationOn:$("simulate-light").checked,planningHead,planningAtoms:[...planningAtoms],planningObjectCount:planningGroup.children.length,environment:!!scene.environment,box:box?.getSize(new THREE.Vector3()).toArray()
});

function saveAnnotation(n,anchors,status,note,origin=project.items.find(i=>i.number===n)?.annotation?.origin||'manual',context={id:project.id,cad_sha256:project.model.source_sha256,document_id:project.items.find(i=>i.number===n).document_id}){return post(`/projects/${context.id}/fai/${n}/annotation`,{anchors,status,note,origin,confidence:project.items.find(i=>i.number===n)?.annotation?.confidence||null,cad_sha256:context.cad_sha256,document_id:context.document_id});}
function focusFAI(){const anchors=anchorsOf(detail?.item);if(!anchors.length||!controls)return;if(mode==='source')setMode('3d');const center=new THREE.Vector3();for(const a of anchors)center.add(new THREE.Vector3(...a.position));center.divideScalar(anchors.length);const direction=camera.position.clone().sub(controls.target).normalize();const damping=controls.enableDamping;controls.enableDamping=false;controls.update();controls.target.copy(center);camera.position.copy(center).add(direction.multiplyScalar(Math.max(20,...anchors.map(a=>center.distanceTo(new THREE.Vector3(...a.position))*5))));$('clip').checked=false;paint();controls.update();controls.enableDamping=damping;}
function enableMarkerDrag(entry){let drag=null;const {button,item}=entry;button.onpointerdown=e=>{if(e.button!==0||!item.annotation)return;e.preventDefault();e.stopPropagation();button.setPointerCapture(e.pointerId);drag={context:{id:project.id,cad_sha256:project.model.source_sha256,document_id:item.document_id},x:e.clientX,y:e.clientY,original:[...(entry.anchors[0].label_offset||[24,-24])],offset:[parseFloat(button.style.left)-Number(entry.graphics[0].line.getAttribute('x1')),parseFloat(button.style.top)-Number(entry.graphics[0].line.getAttribute('y1'))],moved:false};button.dragged=false;};button.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(dx,dy)>4){drag.moved=true;button.dragged=true;entry.anchors[0].label_offset=[drag.offset[0]+dx,drag.offset[1]+dy];invalidateRender();}};button.onpointerup=async e=>{if(!drag)return;const state=drag;drag=null;if(button.hasPointerCapture(e.pointerId))button.releasePointerCapture(e.pointerId);if(state.moved){try{await saveAnnotation(item.number,entry.anchors,item.annotation.status,item.annotation.note,item.annotation.origin,state.context);if(project.id===state.context.id)await refresh();}catch(error){entry.anchors[0].label_offset=state.original;notice(error.message);}}setTimeout(()=>button.dragged=false,150);};button.onpointercancel=()=>{if(drag)entry.anchors[0].label_offset=drag.original;drag=null;};}
function projectMarkers(){if(previewOn){for(const g of markerGroups){g.button.hidden=true;for(const x of g.graphics)x.line.style.display=x.pin.style.display='none';}return;}const host=$('canvas-host'),w=host.clientWidth,h=host.clientHeight,placed=[],markerMode=$('marker-mode').value;const groups=[...markerGroups].sort((a,b)=>(b.item.number===number)-(a.item.number===number)||a.item.number-b.item.number);for(const g of groups){const points=g.anchors.map((a,index)=>{const v=new THREE.Vector3(...a.position);const behind=v.clone().sub(camera.position).dot(camera.getWorldDirection(new THREE.Vector3()))<=0;const clipped=$('clip').checked&&clipPlane.distanceToPoint(v)<0;v.project(camera);return {x:(v.x*.5+.5)*w,y:(-v.y*.5+.5)*h,visible:(!planningHead||!planningAtoms.size||planningAtoms.has(`${g.item.number}:${index}`))&&!behind&&!clipped&&Math.abs(v.x)<=1&&Math.abs(v.y)<=1&&Math.abs(v.z)<=1};});const inPlan=!planningHead||planningFAIs.has(g.item.number);const modeAllows=markerMode!=='none'&&(markerMode!=='active'||g.item.number===number);const visible=points.some(p=>p.visible)&&inPlan&&modeAllows;g.button.hidden=!visible;for(const graphic of g.graphics){graphic.line.style.display=visible?'':'none';graphic.pin.style.display=visible?'':'none';}if(!visible)continue;const origin=points.find(p=>p.visible),offset=g.anchors[0].label_offset||[24,-24],bw=g.button.offsetWidth||65,bh=g.button.offsetHeight||25;let x=origin.x+offset[0],y=origin.y+offset[1];const overlap=(cx,cy)=>placed.some(r=>Math.abs(cx-r.x)<(bw+r.w)/2+6&&Math.abs(cy-r.y)<(bh+r.h)/2+5);for(let step=0;step<20;step++){x=Math.max(bw/2+5,Math.min(w-bw/2-5,origin.x+offset[0]+(step%2)*70));y=Math.max(bh/2+5,Math.min(h-bh/2-5,origin.y+offset[1]+Math.ceil(step/2)*(bh+7)));if(!overlap(x,y))break;}g.button.style.left=x+'px';g.button.style.top=y+'px';placed.push({x,y,w:bw,h:bh});for(let i=0;i<points.length;i++){const p=points[i],{line,pin}=g.graphics[i];line.style.display=pin.style.display=p.visible?'':'none';line.setAttribute('x1',p.x);line.setAttribute('y1',p.y);line.setAttribute('x2',x);line.setAttribute('y2',y);pin.setAttribute('cx',p.x);pin.setAttribute('cy',p.y);}}}
// The planning scene uses the same validated configuration as the candidate BOM.
const planningGroup=new THREE.Group();scene.add(planningGroup);let planningHead=null,planningFAIs=new Set(),planningAtoms=new Set(),planningEpoch=0;
function disposeGroupContents(root){const geometries=new Set(),materials=new Set(),textures=new Set();root.traverse(o=>{if(o.geometry)geometries.add(o.geometry);for(const m of o.material?(Array.isArray(o.material)?o.material:[o.material]):[]){materials.add(m);if(m.map)textures.add(m.map);}});for(const g of geometries)g.dispose();for(const t of textures)t.dispose();for(const m of materials)m.dispose();root.clear();}
function clearPlanningScene(){if(planningHead==='ASSEMBLY'||planningHead==='STATION')$('planning-layout').hidden=true;activeHead=null;activeStationPose=null;activeStationPreview=null;stationSetupGroups.clear();previewOn=false;previewCamera=null;if(controls)controls.enabled=true;$('camera-preview').textContent='虚拟拍摄图';$('save-virtual-shot').hidden=true;disposeGroupContents(virtualShotGroup);disposeGroupContents(simulationGroup);$('simulate-light').checked=false;applySimulation();for(const id of ['simulation-control','camera-preview','save-virtual-shot','physical-report'])$(id).hidden=true;planningEpoch++;disposeGroupContents(planningGroup);$('render-mode-badge').hidden=true;planningHead=null;planningFAIs.clear();planningAtoms.clear();$('clear-planning').hidden=true;if(meshes.length)box=new THREE.Box3().setFromObject(group);invalidateRender();}
function sceneLabel(text,position,scale=120,parent=planningGroup){const c=document.createElement('canvas');c.width=1024;c.height=180;const ctx=c.getContext('2d');ctx.fillStyle='#123349';ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle='#ffffff';ctx.font='40px sans-serif';text.split('|').slice(0,2).forEach((line,i)=>ctx.fillText(line.trim(),18,64+i*65,988));const texture=new THREE.CanvasTexture(c),sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false}));sprite.position.copy(position);sprite.scale.set(scale,scale*21/120,1);parent.add(sprite);return sprite;}
function stationShotMarkers(numbers){
 disposeGroupContents(virtualShotGroup);const wanted=new Set(numbers.map(Number));
 const fov=activeHead?.scene?.fov_xy_mm||activeStationPose?.fov_mm||[20,20],roi=activeHead?.optics?.roi_px||activeStationPose?.image_size_px||[2000,2000],sx=fov[0]/roi[0],sy=fov[1]/roi[1];
 const right=new THREE.Vector3(1,0,0).applyQuaternion(previewCamera.quaternion),up=new THREE.Vector3(0,1,0).applyQuaternion(previewCamera.quaternion);
 const points=activeHead?.capture_points||((project?.items||[]).filter(i=>wanted.has(Number(i.number))).flatMap(item=>anchorsOf(item).map((a,index)=>({id:`${item.number}:${index}`,number:item.number,position:a.position}))));
 for(const point of points){
  const at=new THREE.Vector3(...point.position),clip=at.clone().project(previewCamera);if(Math.abs(clip.x)>1||Math.abs(clip.y)>1||Math.abs(clip.z)>1)continue;
  const dot=new THREE.Mesh(new THREE.CircleGeometry(2.5*sx,12),new THREE.MeshBasicMaterial({color:0xff493f,depthTest:false}));dot.position.copy(at);dot.quaternion.copy(previewCamera.quaternion);dot.renderOrder=500;virtualShotGroup.add(dot);
  const label=`${point.number}.${Number(String(point.id).split(':')[1])+1}`,c=document.createElement('canvas'),ctx=c.getContext('2d');ctx.font='13px sans-serif';c.width=Math.ceil(ctx.measureText(label).width)+8;c.height=22;ctx.font='13px sans-serif';ctx.lineWidth=3;ctx.strokeStyle='#102030';ctx.fillStyle='#fff';ctx.strokeText(label,4,15);ctx.fillText(label,4,15);
  const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));sprite.scale.set(c.width*sx,c.height*sy,1);sprite.position.copy(at).addScaledVector(right,(c.width/2+5)*sx).addScaledVector(up,8*sy);sprite.renderOrder=501;virtualShotGroup.add(sprite);
 }
 virtualShotGroup.visible=previewOn;
}
function selectStationPreview(value='station:all'){
 if(!activeStationPreview)return;let setupId=null,pose=null;
 if(value.startsWith('setup:'))setupId=value.slice(6);
 else if(value.startsWith('pose:')){const poseId=value.slice(5);for(const s of activeStationPreview.setups||[]){const found=(s.device_placement?.poses||[]).find(p=>p.id===poseId);if(found){setupId=s.id;pose=found;break;}}}
 for(const [id,g] of stationSetupGroups)g.visible=!setupId||id===setupId;
 const setup=(activeStationPreview.setups||[]).find(s=>s.id===setupId);if(setup&&!pose)pose=setup.device_placement?.poses?.[0]||null;activeStationPose=pose;
 const numbers=pose?.fai_numbers||setup?.fai_numbers||activeStationPreview.fai_numbers||[];planningFAIs=new Set(numbers.map(Number));planningAtoms=new Set(numbers.flatMap(n=>{const item=project?.items?.find(i=>Number(i.number)===Number(n));return anchorsOf(item).map((_,i)=>`${n}:${i}`);}));renderMarkers();stationShotMarkers(numbers);
 const canShoot=!!pose?.fov_mm?.length&&!!pose?.basis_y_mm?.length;$('camera-preview').disabled=!canShoot;$('camera-preview').hidden=false;$('camera-preview').textContent='虚拟拍摄图';$('save-virtual-shot').hidden=true;
 const pp=pose?.fov_mm?.map(v=>Number(v).toFixed(2)).join(' × ');$('stage-info').textContent=pose?`${setup?.station_id||'Station'} / ${setup?.id} · ${pose.id} · FOV ${pp} mm · FAI ${numbers.join(', ')}`:setup?`${setup.station_id||'Station'} / ${setup.id} · ${numbers.length} 项 FAI · ${setup.device_placement?.unresolved_devices?.length||0} 个设备位姿未解算`: `工站总览 · ${stationSetupGroups.size} 组 Setup · 选择上方下拉菜单切换子方案`;
 const b=new THREE.Box3().setFromObject(group);if(setupId){const visible=stationSetupGroups.get(setupId);if(visible)b.union(new THREE.Box3().setFromObject(visible));}else if(stationSetupGroups.size)b.union(new THREE.Box3().setFromObject(planningGroup));if(!b.isEmpty()){box=b;fit(new THREE.Vector3(.15,-1,.65));}
}
$('clear-planning').onclick=()=>{clearPlanningScene();fit();$('stage-info').textContent='零件模型 · 标记用于位置核对';};
window.addEventListener('oracle-workflow-fai',async e=>{await selectFAI(e.detail);setMode('source');});
let shotOverview=null;
let planningLayouts=[],activeHead=null,previewOn=false,previewCamera=null,activeStationPreview=null,activeStationPose=null;
const simulationGroup=new THREE.Group();scene.add(simulationGroup);const virtualShotGroup=new THREE.Group();scene.add(virtualShotGroup);const stationSetupGroups=new Map();const defaultLights=scene.children.filter(o=>o.isLight);
function applySimulation(){const on=$('simulate-light').checked;defaultLights.forEach(l=>l.visible=!on);simulationGroup.visible=on;virtualShotGroup.visible=previewOn;for(const m of meshes)m.material.envMapIntensity=on?0:.45;planningGroup.visible=!previewOn;if(activeHead||activeStationPose)$('stage-info').textContent=previewOn?'CAD 几何虚拟投影 · 非真实曝光或实拍':`${activeHead?.id||activeStationPose?.id} · ${on?'照明布置预览':'器件布置'} · 安装待验证`;invalidateRender();}
$('simulate-light').onchange=()=>{applySimulation();notice('照明预览用于查看光源方向和布置。反光、透射和曝光效果需通过实物试拍确认。');};
$('camera-preview').onclick=()=>{if(!activeHead&&!activeStationPose)return;previewOn=!previewOn;if(previewOn){let target,front,up,w,h,far;if(activeStationPose){const p=activeStationPose,d=new THREE.Vector3(...p.view_direction).normalize();target=new THREE.Vector3(...p.target_mm);front=target.clone().addScaledVector(d,-Number(p.working_distance_mm||100));up=new THREE.Vector3(...p.basis_y_mm).normalize();[w,h]=p.fov_mm;far=Math.max(5000,Number(p.working_distance_mm||100)*10);}else{const s=activeHead.scene;[w,h]=s.fov_xy_mm;target=new THREE.Vector3(...s.target_mm);front=new THREE.Vector3(...s.lens_front_mm);up=surfaceFrame(s).y;far=activeHead.optics.working_distance_mm+500;}previewCamera=new THREE.OrthographicCamera(-w/2,w/2,h/2,-h/2,.01,far);previewCamera.position.copy(front);previewCamera.up.copy(up);previewCamera.lookAt(target);previewCamera.updateMatrixWorld();stationShotMarkers(activeHead?.fai_numbers||activeStationPose?.fai_numbers||[]);}controls.enabled=!previewOn;$('camera-preview').textContent=previewOn?'退出虚拟拍摄图':'虚拟拍摄图';$('save-virtual-shot').hidden=!previewOn;applySimulation();notice('当前为CAD几何正投影示意，不包含真实光学畸变、景深、反射、曝光和噪声；不能替代相机实拍或计量验收。');};
$('save-virtual-shot').onclick=()=>{
 if(!previewOn||!renderer||!previewCamera)return;
 const roi=activeHead?.optics?.roi_px||activeStationPose?.image_size_px;
 if(!roi||roi.length!==2||roi.some(v=>!Number.isInteger(v)||v<1)){notice('本拍摄位置缺少已计算的图像 ROI，暂不能导出与坐标表一致的虚拟图。');return;}
 const [w,h]=roi;if(Math.max(w,h)>renderer.capabilities.maxTextureSize){notice('图像尺寸超过当前浏览器的渲染能力；未缩放输出，避免改变坐标比例。');return;}
 const previous=renderer.getRenderTarget(),target=new THREE.WebGLRenderTarget(w,h,{colorSpace:THREE.SRGBColorSpace});
 try{
  renderer.setRenderTarget(target);renderer.render(scene,previewCamera);
  const pixels=new Uint8Array(w*h*4);renderer.readRenderTargetPixels(target,0,0,w,h,pixels);
  const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d'),frame=x.createImageData(w,h),stride=w*4;
  for(let row=0;row<h;row++)frame.data.set(pixels.subarray((h-row-1)*stride,(h-row)*stride),row*stride);
  x.putImageData(frame,0,0);x.fillStyle='rgba(9,28,43,.86)';x.fillRect(0,h-36,w,36);x.fillStyle='#fff';x.font='16px sans-serif';
  const fov=activeHead?.scene?.fov_xy_mm||activeStationPose?.fov_mm||[];
  x.fillText(`CAD PROJECTION | NOT A REAL CAPTURE | ${w} x ${h} px | FOV ${fov.map(v=>Number(v).toFixed(3)).join(' x ')} mm | ${activeStationPose?.id||activeHead?.id||''}`,14,h-12);
  const shot=activeHead,pid=project?.id;
  c.toBlob(async blob=>{if(!blob){notice('虚拟图编码失败，请重新导出。');return;}
   try{
    let url;
    if(shot?.workflow_id){const form=new FormData();form.append('workflow_id',shot.workflow_id);form.append('image',blob,'virtual-shot.png');const response=await fetch(`${base}/projects/${pid}/workflow/virtual-shot-image/${encodeURIComponent(shot.id)}`,{method:'POST',body:form});const data=await response.json();if(!response.ok)throw Error(data.detail||'虚拟图保存失败');url=data.download_url;}
    else url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download=`oracle-virtual-shot-${shot?.id||activeStationPose?.id||'preview'}.png`;document.body.append(a);a.click();a.remove();if(url.startsWith('blob:'))setTimeout(()=>URL.revokeObjectURL(url),60000);notice(`虚拟图已生成：${w} × ${h} px，与相机镜头ROI一致。`);
   }catch(e){notice('虚拟图导出失败：'+e.message);}
  },'image/png');
 }catch(e){notice('虚拟图导出失败：'+e.message);}finally{renderer.setRenderTarget(previous);target.dispose();invalidateRender();}
};
$('physical-report').onclick=()=>{if(!activeHead)return;const p=activeHead.physical_checks;notice(p?`初步干涉检查：${p.collisions.length?'发现干涉，需调整':'简化模型未发现干涉'} · ${p.collisions.length} 处已检出干涉 · 安装待验证。当前检查基于简化模型，需补全夹具、支架、接头、线缆和运动空间。`:'该方案尚无安装检查记录，请重新编制方案。');};

window.addEventListener('oracle-project-loaded',()=>{shotOverview=null;planningLayouts=[];$('planning-layout').hidden=true;$('planning-layout').replaceChildren();});
window.addEventListener('oracle-workflow-layouts',e=>{planningLayouts=e.detail.heads;const picker=$('planning-layout');picker.replaceChildren();for(const h of planningLayouts){const option=document.createElement('option');option.value=h.id;option.textContent=`${e.detail.primary.includes(h.id)?'主方案':'照明对比'} ${h.id} · ${wording(h.lighting_strategy||'backlight')} · FAI ${h.fai_numbers.join(',')}`;picker.append(option);}picker.hidden=!planningLayouts.length;});
 $('planning-layout').onchange=()=>{if($('planning-layout').value==='shots:overview'&&shotOverview){window.dispatchEvent(new CustomEvent('oracle-workflow-shot-overview',{detail:shotOverview}));return;}if(planningHead==='STATION'){selectStationPreview($('planning-layout').value);return;}const head=planningLayouts.find(h=>h.id===$('planning-layout').value);if(head)window.dispatchEvent(new CustomEvent('oracle-workflow-scene',{detail:head}));};
window.addEventListener('oracle-workflow-scene',async e=>{
 clearPlanningScene();const epoch=planningEpoch,pid=project?.id,head=e.detail,s=head.scene;activeHead=head;for(const id of ['simulation-control','camera-preview','physical-report'])$(id).hidden=false;planningHead=head.id;$('planning-layout').value=head.id;planningFAIs=new Set(head.fai_numbers);planningAtoms=new Set(head.coverage_atoms||[]);$('clear-planning').hidden=false;setMode('3d');await selectFAI(head.fai_numbers[0]);if(epoch!==planningEpoch||project?.id!==pid)return;$('clip').checked=false;paint();
 const target=new THREE.Vector3(...s.target_mm),front=new THREE.Vector3(...s.lens_front_mm),flange=new THREE.Vector3(...s.camera_flange_mm),normal=new THREE.Vector3(...s.normal),lightFace=new THREE.Vector3(...s.light_face_mm);
 const metal=new THREE.MeshStandardMaterial({color:0x303841,metalness:.85,roughness:.28}),camMat=new THREE.MeshStandardMaterial({color:0x376985,metalness:.65,roughness:.28});
 const lens=new THREE.Mesh(new THREE.CylinderGeometry(s.lens_diameter_mm/2,s.lens_diameter_mm/2,s.lens_length_mm,64),metal);lens.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),normal);lens.position.copy(front).add(flange).multiplyScalar(.5);planningGroup.add(lens);
 const glass=new THREE.Mesh(new THREE.CircleGeometry(s.lens_diameter_mm*.43,64),new THREE.MeshStandardMaterial({color:0x103d53,metalness:.35,roughness:.18,side:THREE.DoubleSide}));glass.quaternion.copy(surfaceFrame(s).quaternion);glass.position.copy(front).addScaledVector(normal,-.05);planningGroup.add(glass);
 const cameraBody=new THREE.Mesh(new THREE.BoxGeometry(...s.camera_body_mm),camMat);cameraBody.quaternion.copy(surfaceFrame(s).quaternion);cameraBody.position.copy(flange).addScaledVector(normal,s.camera_body_mm[2]/2);planningGroup.add(cameraBody);
 const direction=new THREE.Vector3(...(s.light_direction||[0,0,1])),lampGroup=new THREE.Group();lampGroup.position.copy(lightFace);lampGroup.quaternion.copy(surfaceFrame({normal:direction.toArray()}).quaternion);planningGroup.add(lampGroup);
 const emittingMat=new THREE.MeshStandardMaterial({color:0xffaaa0,emissive:0xfe4835,emissiveIntensity:1,side:THREE.DoubleSide});
 if(s.light_shape==='ring'){
  const shape=new THREE.Shape();shape.absarc(0,0,s.light_outer_mm/2,0,Math.PI*2,false);const hole=new THREE.Path();hole.absarc(0,0,s.light_inner_mm/2,0,Math.PI*2,true);shape.holes.push(hole);
  const body=new THREE.Mesh(new THREE.ExtrudeGeometry(shape,{depth:s.light_height_mm,bevelEnabled:false,curveSegments:32}),metal.clone());body.position.z=-s.light_height_mm;lampGroup.add(body);
  const emitter=new THREE.Mesh(new THREE.RingGeometry(s.light_inner_mm/2+3,s.light_outer_mm/2-4,64),emittingMat);emitter.position.z=.1;lampGroup.add(emitter);
 }else{
  const lamp=new THREE.Mesh(new THREE.BoxGeometry(...s.light_body_mm),metal.clone());lamp.position.z=-s.light_body_mm[2]/2;lampGroup.add(lamp);
  const emitter=new THREE.Mesh(new THREE.PlaneGeometry(...s.light_active_mm),emittingMat);lampGroup.add(emitter);
 }
 planningGroup.add(new THREE.ArrowHelper(direction,lightFace,Math.min(40,lightFace.distanceTo(target)),0xffa351,5,2.5));
 const [w,h]=s.fov_xy_mm,corners=fieldCorners(s),points=[];
 for(let i=0;i<4;i++){const a=corners[i],b=corners[(i+1)%4];points.push(a,b,a,a.clone().addScaledVector(normal,head.optics.working_distance_mm));}
 const rays=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0x2ce1b4,transparent:true,opacity:.8}));planningGroup.add(rays);
 const arrow=new THREE.ArrowHelper(normal.clone().negate(),front,head.optics.working_distance_mm,0x34e3c0,6,3);planningGroup.add(arrow);invalidateRender();
 sceneLabel(`${head.camera} / ${head.lens} | 外形示意`,flange.clone().add(new THREE.Vector3(55,0,20)));
 sceneLabel(`${head.light} | ${wording(head.lighting_strategy||"transmitted_backlight")} / 待试拍`,lightFace.clone().add(new THREE.Vector3(65,0,-20)));
 sceneLabel(`FAI ${head.fai_numbers.join(', ')} | FOV ${w.toFixed(2)} x ${h.toFixed(2)} mm`,target.clone().add(new THREE.Vector3(65,0,0)));
 const basisX=new THREE.Vector3(1,0,0).applyQuaternion(lampGroup.quaternion),basisY=new THREE.Vector3(0,1,0).applyQuaternion(lampGroup.quaternion);
 for(let i=0;i<4;i++){const angle=i*Math.PI/2,radius=s.light_shape==='ring'?(s.light_inner_mm+s.light_outer_mm)/4:Math.min(...s.light_active_mm)/4;
  const illumination=new THREE.PointLight(0xff7060,1000,Math.max(500,lightFace.distanceTo(target)*5),2);illumination.position.copy(lightFace).addScaledVector(basisX,radius*Math.cos(angle)).addScaledVector(basisY,radius*Math.sin(angle));simulationGroup.add(illumination);}
 simulationGroup.add(new THREE.AmbientLight(0xffffff,.05));applySimulation();
 box=new THREE.Box3().setFromObject(group).union(new THREE.Box3().setFromObject(planningGroup));fit(new THREE.Vector3(1,-1,.5));
 $('stage-info').textContent=`${head.id} · ${wording(head.lighting_strategy||"backlight")} · 工作距离 ${head.optics.working_distance_mm} mm · 采样间距 ${head.optics.sampling_um_px.toFixed(3)} μm/px · 成像效果待验证`;
 notice('当前为初步设备布置，设备按外形尺寸显示。点击“方案设计与设备库”可查看选型依据和待验证事项。');
});

// Whole assembly: one mesh per physical unit, no GPU lights per exposure.
let activeAssembly=null;
const returnAssembly=document.createElement('button');returnAssembly.id='return-assembly';returnAssembly.textContent='返回主方案';returnAssembly.hidden=true;$('clear-planning').after(returnAssembly);
window.addEventListener('oracle-project-loaded',()=>{activeAssembly=null;returnAssembly.hidden=true;});
returnAssembly.onclick=async()=>{try{const record=activeAssembly;if(!record)return;const latest=await api(projectPath(`/workflow/assembly/${record.scheme_id}`));if(!latest||latest.stale||latest.id!==record.id)throw Error('主方案已变化，请回到 05 加载当前总装');window.dispatchEvent(new CustomEvent('oracle-workflow-assembly',{detail:{...latest,sourceHeads:record.sourceHeads}}));}catch(e){notice(e.message);}};
const tagAssemblyObjects=(start,unit)=>planningGroup.children.slice(start).forEach(o=>{o.userData.assemblyUnit=unit;});
window.addEventListener('oracle-workflow-scene',e=>{const explicit=Object.hasOwn(e.detail,'parentAssembly'),parent=explicit?e.detail.parentAssembly:activeAssembly?.result.camera_units.some(c=>c.head_ids.includes(e.detail.id))?activeAssembly:null;activeAssembly=parent;returnAssembly.hidden=!parent;$('camera-preview').disabled=false;});
$('clear-planning').addEventListener('click',()=>{returnAssembly.hidden=true;});
function assemblyBody(head,{cameraUnit=true,lightUnit=true,color=0x428faf}={}){
 const s=head.scene,n=new THREE.Vector3(...s.normal),front=new THREE.Vector3(...s.lens_front_mm),flange=new THREE.Vector3(...s.camera_flange_mm);
 const mat=new THREE.MeshStandardMaterial({color,metalness:.55,roughness:.35});
 if(cameraUnit){
  const lens=new THREE.Mesh(new THREE.CylinderGeometry(s.lens_diameter_mm/2,s.lens_diameter_mm/2,s.lens_length_mm,20),mat);lens.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),n);lens.position.copy(front).add(flange).multiplyScalar(.5);lens.userData.assemblyCamera=true;planningGroup.add(lens);
  const body=new THREE.Mesh(new THREE.BoxGeometry(...s.camera_body_mm),mat);body.quaternion.copy(surfaceFrame(s).quaternion);body.position.copy(flange).addScaledVector(n,s.camera_body_mm[2]/2);body.userData.assemblyCamera=true;planningGroup.add(body);
  const face=new THREE.Mesh(new THREE.CircleGeometry(s.lens_diameter_mm*.43,20),new THREE.MeshStandardMaterial({color:0x103d53,metalness:.5,roughness:.2,side:THREE.DoubleSide}));face.quaternion.copy(surfaceFrame(s).quaternion);face.position.copy(front).addScaledVector(n,-.05);face.userData.assemblyCamera=true;planningGroup.add(face);
 }
 if(lightUnit){
  const lamp=new THREE.Group(),d=new THREE.Vector3(...s.light_direction);lamp.position.set(...s.light_face_mm);lamp.quaternion.copy(surfaceFrame({normal:d.toArray()}).quaternion);planningGroup.add(lamp);
  const emitting=new THREE.MeshBasicMaterial({color:0xffca7a,transparent:true,opacity:.6,side:THREE.DoubleSide});
  if(s.light_shape==='panel'){
   const body=new THREE.Mesh(new THREE.BoxGeometry(...s.light_body_mm),mat);body.position.z=-s.light_body_mm[2]/2;lamp.add(body);lamp.add(new THREE.Mesh(new THREE.PlaneGeometry(...s.light_active_mm),emitting));
  }else{
   const shape=new THREE.Shape();shape.absarc(0,0,s.light_outer_mm/2,0,Math.PI*2,false);const hole=new THREE.Path();hole.absarc(0,0,s.light_inner_mm/2,0,Math.PI*2,true);shape.holes.push(hole);
   const body=new THREE.Mesh(new THREE.ExtrudeGeometry(shape,{depth:s.light_height_mm,bevelEnabled:false,curveSegments:16}),mat);body.position.z=-s.light_height_mm;lamp.add(body);lamp.add(new THREE.Mesh(new THREE.RingGeometry(s.light_inner_mm/2,s.light_outer_mm/2,24),emitting));
  }
 }
 return flange;
}
function surfaceFrame(s){const n=new THREE.Vector3(...s.normal).normalize();let x=s.basis_x?new THREE.Vector3(...s.basis_x):Math.abs(n.x)<.9?new THREE.Vector3(1,0,0):new THREE.Vector3(0,1,0);x.addScaledVector(n,-x.dot(n)).normalize();const y=s.basis_y?new THREE.Vector3(...s.basis_y):new THREE.Vector3().crossVectors(n,x);return {x,y,n,quaternion:new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x,y,n))};}
function fieldCorners(s,target=s.target_mm){const [w,h]=s.fov_xy_mm,{x,y}=surfaceFrame(s),center=new THREE.Vector3(...target);return [[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]].map(([a,b])=>center.clone().addScaledVector(x,a).addScaledVector(y,b));}
function assemblyLines(points,color,segments=false){const geometry=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p)));const material=new THREE.LineBasicMaterial({color,transparent:true,opacity:.5});planningGroup.add(segments?new THREE.LineSegments(geometry,material):new THREE.Line(geometry,material));}
// Use calculated exposure poses rather than unresolved station CAD origins.
window.addEventListener('oracle-workflow-shot-overview',e=>{
 const heads=e.detail.heads.filter(h=>h.scene?.lens_front_mm?.length===3&&h.scene?.fov_xy_mm?.length===2);
 if(!heads.length){notice('当前方案没有可预览的拍摄位姿。');return;}
 clearPlanningScene();shotOverview={...e.detail,heads};activeAssembly=null;returnAssembly.hidden=true;
 planningLayouts=heads;activeHead=heads[0];planningHead='SHOT_OVERVIEW';
 planningFAIs=new Set(heads.flatMap(h=>h.fai_numbers));planningAtoms=new Set(heads.flatMap(h=>h.coverage_atoms||[]));
 setMode('3d');$('clip').checked=false;paint();$('clear-planning').hidden=false;
 // One current device pose and all exposure footprints; never simultaneous installation.
 assemblyBody(activeHead);
 for(const h of heads){const corners=fieldCorners(h.scene);assemblyLines([...corners,corners[0]].map(v=>v.toArray()),0x39c7b2);}
 assemblyLines([activeHead.scene.lens_front_mm,activeHead.scene.target_mm],0x39c7b2);
 const picker=$('planning-layout');picker.replaceChildren();
 const overview=document.createElement('option');overview.value='shots:overview';overview.textContent=`拍摄方案总览 · ${heads.length} 个拍摄位置`;picker.append(overview);
 for(const h of heads){const o=document.createElement('option');o.value=h.id;o.textContent=`拍摄位置 ${h.id} · FOV ${h.scene.fov_xy_mm.map(v=>Number(v).toFixed(2)).join(' × ')} mm · FAI ${h.fai_numbers.join(',')}`;picker.append(o);}
 picker.hidden=false;picker.value='shots:overview';
 $('camera-preview').hidden=false;$('camera-preview').disabled=false;$('camera-preview').title=`按当前展示的 ${activeHead.id} 相机、镜头和 ROI 生成虚拟图；其他位置通过下拉框切换`;
 $('physical-report').hidden=false;$('simulation-control').hidden=true;
 box=new THREE.Box3().setFromObject(group).union(new THREE.Box3().setFromObject(planningGroup));fit(new THREE.Vector3(1,-1,.65));
 $('render-mode-badge').hidden=false;$('render-mode-badge').textContent=`拍摄方案 · 当前器件姿态 ${activeHead.id}`;
 $('stage-info').textContent=`${heads.length} 个顺序拍摄位置 · 当前显示 ${activeHead.id} 相机 / 镜头 / 光源的计算外形 · 下拉切换逐位置；非同时安装总装`;
 notice(`已加载 ${heads.length} 个拍摄位置，当前器件为 ${activeHead.id}。绿色框为各位置视野；可切换位置或直接生成当前虚拟拍摄图。器件使用规格包络，厂商 CAD 安装坐标尚未核实的部分不冒充实体模型。`);invalidateRender();
});
window.addEventListener('oracle-workflow-assembly',e=>{
 const a=e.detail.result;if(!a)return;clearPlanningScene();activeAssembly=e.detail;activeHead=null;planningHead='ASSEMBLY';planningFAIs=new Set(a.selected_fai);planningAtoms.clear();setMode('3d');$('clip').checked=false;paint();$('clear-planning').hidden=false;
 returnAssembly.hidden=true;const picker=$('planning-layout');picker.replaceChildren();let option=document.createElement('option');option.value='';option.textContent='本方案 · 查看整机布置';picker.append(option);for(const s of e.detail.schemes||[]){const opt=document.createElement('option');opt.value=`scheme:${s.id}`;opt.textContent=`切换方案 · ${s.title} · ${s.station_count} 工站`;opt.selected=s.id===e.detail.scheme_id;picker.append(opt);}
 for(const [i,c] of a.camera_units.entries()){
  const start=planningGroup.children.length,h=c.head,s=h.scene,color=[0x448fc0,0x59b39b,0xb795d5,0xe8aa65][i%4];
  const flange=assemblyBody(h,{lightUnit:false,color});sceneLabel(`${c.id} / ${c.camera} | ${c.lens} / ${c.head_ids.length} 个拍摄位置`,flange.clone().add(new THREE.Vector3(42,0,16+i*14)),78);
  assemblyLines(c.targets_mm,color); // Relative target sequence only, not an approved motion path.
  for(const pose of c.poses||c.targets_mm.map(target=>({scene:{...s,target_mm:target}}))){const corners=fieldCorners(pose.scene);assemblyLines([...corners,corners[0]].map(p=>p.toArray()),color);}
  tagAssemblyObjects(start,c.id);option=document.createElement('option');option.value=c.id;option.textContent=`子方案 · ${c.id} · ${c.head_ids.length} 个拍摄位置${c.rotation_required?' / 需旋转或翻面':''} / FAI ${c.fai_numbers.join(',')}`;picker.append(option);
 }
 for(const l of a.light_units){
  const start=planningGroup.children.length;
  if(l.mode==='merged_panel_requirement'){
   const [w,h]=l.active_requirement_mm,d=new THREE.Vector3(...l.direction),area=new THREE.Mesh(new THREE.PlaneGeometry(w,h),new THREE.MeshBasicMaterial({color:0xffb640,transparent:true,opacity:.25,side:THREE.DoubleSide,depthWrite:false}));area.position.set(...l.face_mm);area.quaternion.copy(surfaceFrame({normal:d.toArray(),basis_x:l.basis_x,basis_y:l.basis_y}).quaternion);planningGroup.add(area);
   const border=new THREE.LineSegments(new THREE.EdgesGeometry(area.geometry),new THREE.LineBasicMaterial({color:0xffb640}));border.position.copy(area.position);border.quaternion.copy(area.quaternion);planningGroup.add(border);
   sceneLabel(`${l.id} / 共用发光面积 ${w.toFixed(1)} x ${h.toFixed(1)} mm | 面积需求 / 型号待选`,area.position.clone().add(new THREE.Vector3(35,0,-25)),88);
  }else{assemblyBody(l.head,{cameraUnit:false});sceneLabel(`${l.id} / ${l.item_id} | 共用移动光源`,new THREE.Vector3(...l.head.scene.light_face_mm).add(new THREE.Vector3(45,0,-28)));}tagAssemblyObjects(start,l.id);
 }
 for(const s of a.depth_stations){const start=planningGroup.children.length,centers=s.search_regions.map(r=>r.search_center_mm).filter(Boolean);if(centers.length){assemblyLines(centers,0xcb82df);sceneLabel(`${s.id} / ${s.sensor_ids.join(',')} | 测量区域 / 安装位置待确认`,new THREE.Vector3(...centers[0]).add(new THREE.Vector3(-65,0,30)),72);}tagAssemblyObjects(start,`depth:${s.id}`);option=document.createElement('option');option.value=`depth:${s.id}`;option.textContent=`子方案 · ${s.id} · 三维测量区域（安装位置待确认）`;picker.append(option);}
 picker.hidden=false;$('camera-preview').hidden=false;$('camera-preview').disabled=true;$('physical-report').hidden=false;$('simulation-control').hidden=true;
 box=new THREE.Box3().setFromObject(group).union(new THREE.Box3().setFromObject(planningGroup));fit(new THREE.Vector3(1,-1,.65));
 $('stage-info').textContent=`整机布置 · ${a.camera_units.length} 个相机组 / ${a.light_units.length} 个光源 · ${a.selected_fai.length} 项FAI · ${a.collisions.length} 处干涉提示 · 安装待验证`;
 notice('当前显示整机的相机组和共用光源。橙色区域表示所需发光面积，光源型号尚未确定。可选择子方案查看对应设备和测量点。');invalidateRender();
});
const singleLayoutChange=$('planning-layout').onchange;
$('planning-layout').onchange=()=>{
 if(planningHead!=='ASSEMBLY'){singleLayoutChange();return;}
 if($('planning-layout').value.startsWith('scheme:')){window.dispatchEvent(new CustomEvent('oracle-switch-integrated-scheme',{detail:{scheme_id:$('planning-layout').value.slice(7)}}));return;}
 const a=activeAssembly.result,value=$('planning-layout').value,c=a.camera_units.find(c=>c.id===value),depth=a.depth_stations.find(s=>`depth:${s.id}`===value);if(previewOn)$('camera-preview').click();activeHead=c?.head||null;
 const lights=c?a.light_units.filter(l=>l.head_ids.some(id=>c.head_ids.includes(id))).map(l=>l.id):[];
 planningGroup.children.forEach(o=>{const unit=o.userData.assemblyUnit;o.visible=!value||unit===value||lights.includes(unit);});returnAssembly.hidden=!value;$('camera-preview').disabled=!c;
 const atoms=c?c.head_ids.flatMap(id=>activeAssembly.sourceHeads?.find(h=>h.id===id)?.coverage_atoms||[]):depth?depth.search_regions.flatMap(r=>r.marker_atom_refs||[]):[];planningFAIs=new Set(c?.fai_numbers||(depth?atoms.map(id=>Number(id.split(':')[0])):a.selected_fai));planningAtoms=new Set(atoms);renderMarkers();
 $('stage-info').textContent=c?`子方案 ${c.id} · ${c.head_ids.length} 个拍摄位置 · 共用光源 · 相对运动待验证`:depth?`子方案 ${depth.id} · 三维测量区域 · 传感器安装待确认`:'主方案 · 完整设备布置';invalidateRender();
};
const singlePhysicalReport=$('physical-report').onclick;
$('physical-report').onclick=()=>{if(planningHead!=='ASSEMBLY'){singlePhysicalReport();return;}const a=activeAssembly.result;notice(`总装静态包络：${a.collisions.length} 处干涉；相机及光源运动包络：${a.motion_screen?.potential_interference_count??'未检查'} 段需避让；${a.light_units.filter(l=>l.status==='required_emitting_area_only_no_verified_supplier_body').length} 组光源仅有面积需求。完整工装、线缆、运动轴和安装仍待验证。`);};
const previousDebug=window.__oracleViewerDiagnostics;
if(previousDebug)window.__oracleViewerDiagnostics=()=>({...previousDebug(),assembly:planningHead==='ASSEMBLY'?{cameraUnits:activeAssembly?.result.camera_units.length,lightUnits:activeAssembly?.result.light_units.length,status:activeAssembly?.result.status,selectedUnit:$('planning-layout').value,visibleUnits:[...new Set(planningGroup.children.filter(o=>o.visible).map(o=>o.userData.assemblyUnit))]}:null});

window.addEventListener('oracle-station-regions',e=>{
 const station=e.detail;if(!station)return;$('clear-planning').click();
 planningHead='STATION';planningFAIs=new Set(station.fai_numbers||[]);$('clear-planning').hidden=false;$('render-mode-badge').textContent='方案 CAD 外形预览 · 尚未生成安装位姿';$('render-mode-badge').hidden=false;setMode('3d');renderMarkers();
 $('stage-info').textContent=`${station.id} ${station.name} · ${station.setups.length} 组 Setup · CAD 与FAI区域清单视图 · 安装位姿未生成`;
 notice('此视图按工站/Setup列出下载的STEP外形和FAI区域；当前方案未提供设备安装位姿，因此不代表实际摆放或避碰结果。');invalidateRender();
 if(station.setups.some(s=>s.device_ids?.length))window.dispatchEvent(new CustomEvent('oracle-station-device-preview',{detail:{setups:station.setups,project_id:station.project_id,view:station.view||'setup'}}));
});

window.addEventListener('oracle-station-device-preview',async e=>{
 const request=++planningEpoch,detail=e.detail||{},setups=detail.setups||[];disposeGroupContents(planningGroup);disposeGroupContents(virtualShotGroup);stationSetupGroups.clear();activeStationPreview=detail;activeStationPose=null;previewOn=false;previewCamera=null;controls.enabled=true;$('camera-preview').textContent='虚拟拍摄图';$('save-virtual-shot').hidden=true;
 $('render-mode-badge').textContent=`${detail.view==='overview'?'工站总方案':'Setup'} · 器件位姿与镜头同轴包络预览`;$('render-mode-badge').hidden=false;
 const specs=await api('/library');if(request!==planningEpoch)return;const byId=new Map(specs.items.map(i=>[i.id,i]));let x=box?.max.x??0;x+=60;const centerY=box?.getCenter(new THREE.Vector3()).y??0,centerZ=box?.getCenter(new THREE.Vector3()).z??0,missing=[];let shown=0,posed=0;
 const addPoseEnvelope=(pose,color,parent)=>{if(!pose.fov_mm||!pose.basis_x_mm||!pose.basis_y_mm)return;const t=new THREE.Vector3(...pose.target_mm),p=new THREE.Vector3(...(pose.axis_point_world_mm||pose.position_mm)),bx=new THREE.Vector3(...pose.basis_x_mm),by=new THREE.Vector3(...pose.basis_y_mm),w=pose.fov_mm[0]/2,h=pose.fov_mm[1]/2,corners=[t.clone().addScaledVector(bx,-w).addScaledVector(by,-h),t.clone().addScaledVector(bx,w).addScaledVector(by,-h),t.clone().addScaledVector(bx,w).addScaledVector(by,h),t.clone().addScaledVector(bx,-w).addScaledVector(by,h)],points=[];for(let i=0;i<4;i++){points.push(p,corners[i],corners[i],corners[(i+1)%4]);}const g=new THREE.BufferGeometry().setFromPoints(points),line=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color,transparent:true,opacity:.72}));line.userData.stationPoseEnvelope=true;parent.add(line);};
 for(const setup of setups){const setupGroup=new THREE.Group();setupGroup.userData.stationSetup=setup.id;stationSetupGroups.set(setup.id,setupGroup);planningGroup.add(setupGroup);for(const id of setup.device_ids||[]){const item=byId.get(id);if(!item)continue;try{
   const r=await fetch(`${base}/library/items/${encodeURIComponent(id)}/preview-mesh`);if(!r.ok){const err=await r.json().catch(()=>({}));missing.push(`${id}: ${err.detail||'无可用本地实体CAD'}`);continue;}
   const data=await r.json();if(request!==planningEpoch)return;if(!data.brep_valid||data.source_format!=='STEP')throw Error('CAD 网格未经有效实体校验');
   const bodies=new Map();for(const f of data.faces){if(!bodies.has(f.body))bodies.set(f.body,[]);bodies.get(f.body).push(f);}
   const prototype=new THREE.Group();for(const rows of bodies.values()){let nv=0,ni=0;for(const f of rows){nv+=f.positions.length/3;ni+=f.indices.length;}const p=new Float32Array(nv*3),ix=new Uint32Array(ni);let vo=0,io=0;for(const f of rows){p.set(f.positions,vo*3);for(let i=0;i<f.indices.length;i++)ix[io+i]=f.indices[i]+vo;vo+=f.positions.length/3;io+=f.indices.length;}const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(p,3));g.setIndex(new THREE.BufferAttribute(ix,1));g.computeVertexNormals();prototype.add(new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0x91a5b6,metalness:.38,roughness:.42,side:THREE.DoubleSide})));}
   // Use the complete source-CAD transform: its origin, optical-axis datum point and roll
   // datum are already resolved by the plan against the referenced STEP frame.
   const placement=setup.device_placement,poses=(placement?.poses||[]).filter(p=>p.device_id===id);if(!placement||!poses.length){const blocked=(placement?.unresolved_devices||[]).find(x=>x.device_id===id);missing.push(`${id}: ${blocked?.reason||'方案没有保存可追溯的完整CAD位姿，未放入渲染场景'}`);continue;}const shownPoses=poses;
   for(const pose of shownPoses){const root=prototype.clone(true),rawBox=new THREE.Box3().setFromObject(root);if(pose){root.quaternion.set(...pose.quaternion_xyzw);root.position.set(...pose.cad_origin_world_mm);addPoseEnvelope(pose,setup.channel==='2d_vision'?0x35c2ce:0xffb24a,setupGroup);posed++;sceneLabel(`${setup.station_id||'Station'}/${setup.id} · ${id} · 光轴/原点按厂家CAD · WD ${pose.working_distance_mm.toFixed(1)} mm · FAI ${pose.fai_numbers.join(',')}`,new THREE.Vector3(...pose.axis_point_world_mm),Math.max(45,Math.min(120,rawBox.getSize(new THREE.Vector3()).length()*.22)),setupGroup);
     for(const assy of (placement?.assemblies||[]).filter(a=>a.camera_pose_id===pose.id)){const axis=new THREE.Vector3(...assy.lens_axis_world).normalize(),center=new THREE.Vector3(...assy.lens_center_mm),diameter=Number(assy.lens_outer_diameter_mm||44),length=Number(assy.lens_length_mm||137.5);const geo=new THREE.CylinderGeometry(diameter/2,diameter/2,length,48,1,true),mat=new THREE.MeshStandardMaterial({color:0x49bdc9,transparent:true,opacity:.18,wireframe:true,roughness:.35,depthWrite:false}),env=new THREE.Mesh(geo,mat);env.position.copy(center);env.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),axis);env.userData.stationPoseEnvelope=true;setupGroup.add(env);const front=new THREE.Vector3(...assy.lens_front_mm),back=new THREE.Vector3(...assy.lens_back_mm);for(const pt of [front,back]){const ring=new THREE.Mesh(new THREE.TorusGeometry(diameter/2,.65,8,48),new THREE.MeshBasicMaterial({color:0x5ee1eb,transparent:true,opacity:.8}));ring.position.copy(pt);ring.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),axis);ring.userData.stationPoseEnvelope=true;setupGroup.add(ring);}sceneLabel(`${assy.lens_id} · 厂家尺寸包络 / STEP待补`,center,45,setupGroup);}
    }
    setupGroup.add(root);shown++;
   }
  }catch(err){missing.push(`${id}: ${err.message}`);}}}
 if(request!==planningEpoch)return;planningHead='STATION';const picker=$('planning-layout');picker.replaceChildren();let option=document.createElement('option');option.value='station:all';option.textContent='工站总览 · 全部 Setup';picker.append(option);for(const setup of setups){option=document.createElement('option');option.value=`setup:${setup.id}`;option.textContent=`${setup.station_id||'Station'} / ${setup.id} · ${setup.channel} · FAI ${setup.fai_numbers.join(',')}`;picker.append(option);for(const pose of setup.device_placement?.poses||[]){const shot=document.createElement('option');shot.value=`pose:${pose.id}`;shot.textContent=`拍摄位置 ${pose.id} · FAI ${pose.fai_numbers.join(',')} · FOV ${pose.fov_mm?.map(n=>Number(n).toFixed(1)).join(' × ')||'未定'} mm`;picker.append(shot);}}picker.hidden=false;
 for(const id of ['camera-preview','physical-report'])$(id).hidden=false;$('simulation-control').hidden=true;$('camera-preview').disabled=true;$('clear-planning').hidden=false;
 const sceneBox=new THREE.Box3().setFromObject(group);if(shown)sceneBox.union(new THREE.Box3().setFromObject(planningGroup));if(!sceneBox.isEmpty()){box=sceneBox;fit(shown?new THREE.Vector3(.15,-1,.65):undefined);}
 const lensCount=setups.reduce((n,s)=>n+(s.device_placement?.assemblies||[]).length,0);
 selectStationPreview('station:all');$('stage-info').textContent=`${detail.view==='overview'?'项目工站总览':'Setup 器件预览'} · ${posed} 个厂家CAD坐标系位姿 / ${shown} 件真实 STEP CAD · ${lensCount} 组相机镜头同轴包络${missing.length?` · ${missing.length} 件位姿/CAD未解决`:''}`;
 notice(missing.length?`相机与镜头按同轴接口及厂家尺寸计算；镜头以线框规格包络显示，非厂家STEP外形。未核验光轴基准点或缺少STEP的设备不生成假位姿：${missing.join('；')}`:`已按每个设备原始CAD坐标系放置；镜头按C-mount和厂家尺寸叠加为线框包络（厂家STEP未入库）。视锥按目录FOV生成，夹具碰撞和线缆空间仍待验证。`);invalidateRender();
});

// Export actual renderer frames, restoring the view after capture.
window.oracleCaptureAssemblyFrames=async record=>{
 const oldPosition=camera.position.clone(),oldQuaternion=camera.quaternion.clone(),oldTarget=controls.target.clone(),oldMode=$('marker-mode').value,oldBackground=scene.background;
 window.dispatchEvent(new CustomEvent('oracle-workflow-assembly',{detail:record}));
 scene.background=new THREE.Color(0x142838);
 const frames={};const frame=async()=>{await new Promise(resolve=>requestAnimationFrame(resolve));renderer.render(scene,camera);return renderer.domElement.toDataURL('image/png');};
 try{
  $('marker-mode').value='active';frames.overview=await frame();
  const center=new THREE.Vector3(...(record.result.light_units[0].face_mm||record.result.light_units[0].head.scene.light_face_mm));center.z=0;
  camera.position.copy(center).add(new THREE.Vector3(0,0,500));controls.target.copy(center);camera.lookAt(center);controls.update();planningGroup.traverse(o=>{if(o.userData.assemblyCamera)o.visible=false;});frames.top=await frame();planningGroup.traverse(o=>{if(o.userData.assemblyCamera)o.visible=true;});
  for(const c of record.result.camera_units){
   const s=c.head.scene,[w,h]=s.fov_xy_mm,pc=new THREE.OrthographicCamera(-w/2,w/2,h/2,-h/2,.01,c.head.optics.working_distance_mm+500);
   pc.position.set(...s.lens_front_mm);pc.up.copy(surfaceFrame(s).y);pc.lookAt(new THREE.Vector3(...s.target_mm));pc.updateMatrixWorld();planningGroup.visible=false;
   renderer.render(scene,pc);frames[c.id]=renderer.domElement.toDataURL('image/png');planningGroup.visible=true;
  }
  return frames;
 }finally{
  scene.background=oldBackground;planningGroup.traverse(o=>{if(o.userData.assemblyCamera)o.visible=true;});planningGroup.visible=true;camera.position.copy(oldPosition);camera.quaternion.copy(oldQuaternion);controls.target.copy(oldTarget);controls.update();$('marker-mode').value=oldMode;invalidateRender();
 }
};

// Snapshot the current CAD/FAI renderer for the complete station review deck.
// This captures measured-region evidence, never invents installation poses.
window.oracleCaptureStationFrames=async record=>{
 if(!renderer||contextLost||!project||project.id!==record.project_id&&record.project_id)throw Error('当前项目三维显示尚未就绪');
 const exportProject=project.id;
 const scopes=[['overview',record.task_snapshot.map(t=>t.number)],...record.result.stations.flatMap(st=>[['station_'+st.id,st.fai_numbers],...st.setups.map(u=>['setup_'+u.id,u.fai_numbers])])];
 if(scopes.length>128)throw Error('本次超过128个渲染视图，请拆分工站后导出；不能静默漏掉Setup');
 const old={position:camera.position.clone(),quaternion:camera.quaternion.clone(),target:controls.target.clone(),aspect:camera.aspect,near:camera.near,far:camera.far,background:scene.background,size:renderer.getSize(new THREE.Vector2()),ratio:renderer.getPixelRatio(),visible:planningGroup.visible,simulationVisible:simulationGroup.visible,clipping:renderer.localClippingEnabled,damping:controls.enableDamping};
 const savedMeshes=meshes.map(m=>({mesh:m,colors:m.geometry.getAttribute('color').array.slice(),opacity:m.material.opacity,transparent:m.material.transparent,depthWrite:m.material.depthWrite}));
 const frames={};
 try{
  for(const saved of savedMeshes){saved.mesh.material.opacity=1;saved.mesh.material.transparent=false;saved.mesh.material.depthWrite=true;}
  planningGroup.visible=false;simulationGroup.visible=false;renderer.localClippingEnabled=false;scene.background=new THREE.Color(0xf0f4f6);controls.enableDamping=false;controls.update();
  renderer.setPixelRatio(1);renderer.setSize(1400,820,false);renderer.setViewport(0,0,1400,820);renderer.setScissorTest(false);camera.aspect=1400/820;camera.updateProjectionMatrix();
  // Fit CAD only; legacy equipment is excluded from the current station plan.
  const cadBox=new THREE.Box3().setFromObject(group),center=cadBox.getCenter(new THREE.Vector3()),extent=cadBox.getSize(new THREE.Vector3());
  camera.position.copy(center).add(new THREE.Vector3(1,-1,.85).normalize().multiplyScalar(Math.max(extent.x,extent.y,extent.z)*.65/Math.tan(THREE.MathUtils.degToRad(camera.fov/2))));controls.target.copy(center);camera.lookAt(center);camera.near=.01;camera.far=100000;camera.updateProjectionMatrix();controls.update();
  for(const [key,numbers] of scopes){
   if(project.id!==exportProject||project.id!==$('projects').value)throw Error('项目已切换，请重新导出');
   const wantedFaces=new Set(record.task_snapshot.filter(t=>numbers.includes(t.number)).flatMap(t=>t.anchors.map(a=>a.face_id)));const frameColor=new THREE.Color();
   for(const saved of savedMeshes){const colors=saved.mesh.geometry.getAttribute('color');for(const range of saved.mesh.userData.ranges){frameColor.set(wantedFaces.has(range.id)?0x20b8b2:0x8f9da8);for(let i=range.vertexStart;i<range.vertexEnd;i++)colors.setXYZ(i,frameColor.r,frameColor.g,frameColor.b);}colors.needsUpdate=true;}
   await new Promise(resolve=>requestAnimationFrame(resolve));renderer.setViewport(0,0,1400,820);renderer.setScissorTest(false);renderer.render(scene,camera);
   const canvas=document.createElement('canvas');canvas.width=1400;canvas.height=820;const ctx=canvas.getContext('2d');ctx.drawImage(renderer.domElement,0,0);ctx.font='bold 22px "Noto Sans CJK SC", sans-serif';
   const wanted=new Set(numbers),placed=[];
   for(const task of record.task_snapshot.filter(t=>wanted.has(t.number))){
    const points=task.anchors.map(a=>new THREE.Vector3(...a.position).project(camera)).filter(v=>Math.abs(v.x)<1&&Math.abs(v.y)<1&&Math.abs(v.z)<1);
    if(!points.length)continue;
    ctx.fillStyle='#007C85';for(const v of points){ctx.beginPath();ctx.arc((v.x*.5+.5)*1400,(-v.y*.5+.5)*820,3.5,0,Math.PI*2);ctx.fill();}
    const point=points[0],px=(point.x*.5+.5)*1400,py=(-point.y*.5+.5)*820,label=`FAI ${task.number} ${(task.symbols||[]).join(' ')}`,width=ctx.measureText(label).width+16;
    let x=Math.min(1400-width-10,Math.max(10,px+18)),y=Math.max(38,Math.min(770,py-18));
    for(let attempt=0;attempt<80&&placed.some(r=>x<r.x+r.w+3&&x+width>r.x-3&&y<r.y+31&&y+29>r.y-3);attempt++){y+=32;if(y>770){y=45;x=Math.max(10,x-width-25);}}
    placed.push({x,y,w:width});ctx.strokeStyle='#007C85';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(x,y+12);ctx.stroke();ctx.fillStyle='#FFFFFF';ctx.fillRect(x,y,width,29);ctx.fillStyle='#174250';ctx.fillText(label,x+8,y+23);
   }
   ctx.font='18px "Noto Sans CJK SC",sans-serif';ctx.fillStyle='#435E6B';ctx.fillText('CAD 标记投影；实际光路、可见性和安装位姿待验证',18,803);frames[key]=canvas.toDataURL('image/png');
  }
  return frames;
 }finally{
  for(const saved of savedMeshes){if(!meshes.includes(saved.mesh))continue;const color=saved.mesh.geometry.getAttribute('color');color.array.set(saved.colors);color.needsUpdate=true;saved.mesh.material.opacity=saved.opacity;saved.mesh.material.transparent=saved.transparent;saved.mesh.material.depthWrite=saved.depthWrite;}
  scene.background=old.background;planningGroup.visible=old.visible;simulationGroup.visible=old.simulationVisible;renderer.localClippingEnabled=old.clipping;renderer.setPixelRatio(old.ratio);renderer.setSize(old.size.x,old.size.y,false);renderer.setViewport(0,0,old.size.x,old.size.y);renderer.setScissorTest(false);if(project.id===exportProject){camera.position.copy(old.position);camera.quaternion.copy(old.quaternion);controls.target.copy(old.target);camera.aspect=old.aspect;camera.near=old.near;camera.far=old.far;camera.updateProjectionMatrix();controls.update();}else resize();controls.enableDamping=old.damping;invalidateRender();
 }
};
