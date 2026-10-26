import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
const input=process.argv[2],dir=path.resolve(process.argv[3]);
const skill=process.env.ORACLE_PRESENTATION_SKILL_DIR||path.join(os.homedir(),'.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations');
const require=createRequire(path.join(process.env.RUNTIME_NODE_MODULES,'package.json'));
const {Presentation,PresentationFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const {describe,parameterLabel,parameterValue}=await import('../oracle_viewer/static/copy.js');
const src=JSON.parse(await fs.readFile(input,'utf8')),r=src.record,result=r.result,project=src.project;
// Font metrics drive pagination. Explicit newlines are not rewrapped by the renderer.
const fontPath=execFileSync('fc-match',['-f','%{file}','Noto Sans CJK SC'],{encoding:'utf8'}).trim();
const characters=[...new Set(JSON.stringify(src)+await fs.readFile(new URL(import.meta.url),'utf8'))];
const metrics=JSON.parse(execFileSync(process.env.RUNTIME_PYTHON,['-c','import sys,json; from PIL import ImageFont; f=ImageFont.truetype(sys.argv[1],100); print(json.dumps({c:f.getlength(c)/100 for c in json.load(sys.stdin)}))',fontPath],{input:JSON.stringify(characters),encoding:'utf8',maxBuffer:4000000}));
const p=Presentation.create({slideSize:{width:1280,height:720}}),font='Noto Sans CJK SC',tables=[],faiIndex={},setupIndex={},nodeIndex={},sections=[];
const owners=new Map(result.stations.flatMap(st=>st.setups.flatMap(u=>u.fai_numbers.map(n=>[n,{station:st,setup:u}]))));
const routes=new Map(r.routes.map(t=>[t.number,t])),devices=new Map(src.catalog.items.map(d=>[d.id,d]));
await fs.mkdir(path.join(dir,'.build'),{recursive:true});await fs.mkdir(path.join(dir,'.build/verified'),{recursive:true});await fs.mkdir(path.join(dir,'output'),{recursive:true});
const val=x=>x===null||x===undefined?'未提供':typeof x==='object'?JSON.stringify(x):String(x);
function txt(s,value,x,y,w,h,size=24,color='#243D49',bold=false){const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});t.text=value;t.text.style={typeface:font,fontSize:size,color,bold,wrap:'none',autoFit:'none',insets:0};return t;}
function page(title,chapter='工程评审',subtitle=''){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 txt(s,chapter,60,30,1160,27,18,'#007C85',true);txt(s,title,60,73,1160,65,43,'#173543',true);
 if(subtitle)txt(s,subtitle,60,144,1160,45,23,'#516773');
 txt(s,`${project.drawing_no} Rev.${project.revision}  |  待验证方案  |  ${p.slides.items.length}`,60,681,1160,23,16,'#617682');
 s.speakerNotes.text=`项目 ${project.title}\n源工站方案 ${r.id}\n资料签名 ${r.source_key}\n本地模型 ${r.model_identity.model_alias} / ${r.model_identity.inference_backend}\n导出说明也由本地模型生成，规格、归属、原文及数值直接取快照，不重新决策。\n来源文档 ${JSON.stringify(r.source_summary.documents)}\n导出时间 ${new Date().toISOString()}`;
 return s;
}
function units(s){return [...String(s)].reduce((n,c)=>n+(metrics[c]??(c.charCodeAt(0)>255?1:.7)),0);}
function wrap(value,width=1152,size=24){
 const max=(width-12)/size/1.025,lines=[];
 for(const line of val(value).split('\n')){
  let part='';const tokens=line.match(/[A-Za-z0-9_:+\-./]+|[^\n]/gu)||[''];
  for(const token of tokens){
   // Keep names, coordinate values and decimal signs intact where they fit.
   if(units(token)<=max){
    const punctuation=/^[。，；：！？、,.;:!?）)\]】]$/u.test(token);
    if(part&&units(part+token)>max&&!punctuation){lines.push(part.trimEnd());part='';}
    if(!part&&token===' ')continue;part+=token;
   }else for(const c of token){if(units(part+c)>max){lines.push(part.trimEnd());part='';}part+=c;}
  }
  lines.push(part.trimEnd());
 }
 return lines;
}
function paragraphs(title,items,chapter='工程评审',notes=null){let s=null,y=0,first=0;for(const item of items){const lines=wrap(describe(item));let offset=0;while(offset<lines.length){if(!s||y+34>645){s=page(title,chapter);if(!first)first=p.slides.items.length;y=174;if(notes)s.speakerNotes.append('\n'+JSON.stringify(notes));}const count=Math.min(Math.floor((645-y)/34),lines.length-offset);txt(s,lines.slice(offset,offset+count).join('\n'),60,y,1160,count*34,24);y+=count*34+17;offset+=count;}}if(!s)page(title,chapter);return first||p.slides.items.length;}
function tablePage(title,headers,rows,widths,chapter='数据附录',subtitle='',notes=null){
 let offset=0;const pages=[];
 do{
  let chosen=[],heights=[],height=52;
  while(offset+chosen.length<rows.length){const row=rows[offset+chosen.length];const lines=Math.max(...row.map((x,i)=>wrap(x,widths[i]-24,23).length));const h=Math.max(48,lines*31+20);if(h>420)throw Error(`Unpageable table row ${title}`);if(chosen.length&&height+h>440)break;chosen.push(row);heights.push(h);height+=h;if(height>=440)break;}
  const s=page(title,chapter,subtitle);pages.push(p.slides.items.length);const values=[headers,...chosen].map(row=>row.map((x,i)=>wrap(x,widths[i]-24,23).join('\n')));
  const t=s.tables.add({rows:values.length,columns:headers.length,left:60,top:198,width:1160,height,values,columnWidths:widths});tables.push(p.slides.items.length);t.borders.assign({fill:'#DCE3E6',width:1,style:'solid'});
  for(let i=0;i<values.length;i++)for(let j=0;j<headers.length;j++){const c=t.getCell(i,j);c.fill=i===0?'#173F50':i%2?'#FFFFFF':'#F0F5F6';c.text.style={typeface:font,fontSize:23,color:i===0?'#FFFFFF':'#243D49',bold:i===0,wrap:'none',autoFit:'none',insets:{left:12,right:12,top:10,bottom:10}};}
  // Explicit row sizes preserve wrapping and readable font size.
  try{t.rows[0].height=52;for(let i=0;i<heights.length;i++)t.rows[i+1].height=heights[i];}catch{}
  if(notes)s.speakerNotes.append('\n'+JSON.stringify(notes));offset+=chosen.length;
 }while(offset<rows.length);
 return pages;
}
async function framePage(title,key,lines,chapter='方案视图'){
 const s=page(title,chapter,'实际 CAD 渲染与已定位 FAI；设备安装位姿尚未确定。');const data=await fs.readFile(src.images[key]);
 s.images.add({blob:data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength),contentType:'image/png',alt:`实际Oracle Viewer渲染，${key}`,fit:'contain',position:{left:60,top:201,width:770,height:429}});
 let y=208;for(const line of lines){const wrapped=wrap(line,345,23);txt(s,wrapped.join('\n'),865,y,355,wrapped.length*31+8,23);y+=wrapped.length*31+27;}
 s.speakerNotes.append('\n渲染为显示几何，不模拟实际曝光/遮挡/MTF。针点是已保存CAD标记，不代表所有物理边均可见。全部FAI见后续附录。');
 return p.slides.items.length;
}
const pairLabels={eligible:'初选规格适配',reasons:'尚未满足的规格条件',sensor_roi_mm:'传感器ROI尺寸（mm）',sensor_diagonal_mm:'传感器ROI对角线（mm）',fov_mm:'目标视场（mm）',sampling_um_px:'物方采样间距（μm/px）'};
const pairValue=v=>Array.isArray(v)&&!v.length?'无（限当前已提供的规格项）':parameterValue(Array.isArray(v)?v.map(x=>typeof x==='number'?Number(x.toFixed(4)):x):typeof v==='number'?Number(v.toFixed(4)):v);
const symbols=t=>(t.symbols||[]).join(' · ')||'SEE MSOP';
const spec=t=>(t.specifications||[]).map(x=>`${x.spc?'SPC '+x.spc+'；':''}名义值 ${x.nominal}；上偏差 ${x.upper}；下偏差 ${x.lower}；${x.unit||'单位见原页'}`).join('\n')||'数值未形成结构化字段；保留原方法页标题及全文，不自行补值。';
let s=page('全项目测量与工站方案','ORACLE VIEWER / 工程评审');txt(s,`${project.drawing_no} Rev.${project.revision}`,60,194,1160,80,60,'#007C85',true);txt(s,project.title,60,296,1160,62,34);txt(s,`${result.coverage.source_total} 项 FAI  ·  ${result.station_count} 个工站  ·  ${result.setup_count} 组 Setup`,60,387,1160,64,37,'#173543',true);txt(s,'完整工作流、工站与子方案、设备规格、逐项测量要求及交付条件',60,481,1160,65,27);txt(s,'用于方案评审；安装可行性、方法等效性和计量验收仍需验证。',60,590,1160,52,24,'#936318');
sections.push({title:'总方案',slide:1});
paragraphs('如何阅读这份方案',['主汇报说明六个工作流节点、总方案及工站/Setup关系。每个节点逐条写明输入资料、处理动作、输出与交接条件。','工站章节列出全部 Setup、先后顺序、装夹、采集动作、候选设备、光学核算、共用实例及待验证条件。','FAI 附录逐项保留原要求、来源、通道、唯一结果归属、方法建议和验证条件。坐标附录分别列出 MSOP 源点和 CAD 标记。','最后的索引可按 FAI / Setup 查页。原 MSOP 方法全文、基准依赖及字段快照保存在相应 FAI 页的演讲者备注中。','未定位项仍在全项目方案内；未知的公差、设备位姿或动作时间均保留为未提供。']);
await framePage('总方案：测量对象与当前位置','overview',[`来源 FAI：${result.coverage.source_total} 项`,`已有归属：${result.coverage.assigned} 项`,`有标记位置：${result.coverage.located} 项`,`安装验证：${result.coverage.physically_validated} 项`,`计量验收：${result.coverage.metrology_accepted} 项`]);
paragraphs('总方案结论与适用边界',[result.review,`当前整线 CT：${val(result.line_ct_s)}。工站顺序、并行调度和转运时间需进一步定义。`,`未定位 FAI：${result.coverage.unlocated_fai.join(', ')||'无'}。这些项目保留结果归属，测量位置需补齐后才能编制可执行采集路径。`,'此轮规划使用本地 Strata。渲染及版式由工具生成；不会以缺少资料的默认值代替工程结论。']);
const nodeNames=['测量要求','测量区域与测量方式','相机、镜头与照明库','设备选型与拍摄安排','Station / Setup 与 BOM','验证与交付'];
for(const [i,node] of src.explanation.nodes.entries()){
 const start=p.slides.items.length+1;nodeIndex[node.id]=start;sections.push({title:`${String(i+1).padStart(2,'0')} ${nodeNames[i]}`,slide:start});
 for(const [key,name] of [['inputs','输入资料'],['actions','处理动作'],['outputs','输出结果'],['checks','交接与检查条件']])paragraphs(`${String(i+1).padStart(2,'0')} ${nodeNames[i]}：${name}`,node[key].map((v,j)=>`${j+1}. ${v}`),'工作流节点',node);
 if(i===0)paragraphs('01 当前资料与解析范围',[...r.source_summary.documents.map(d=>`${d.role==='cad'?'3D模型':d.role==='msop'?'MSOP':'2D图纸'}：${d.filename}`),`解析到 ${src.tasks.length} 项源 FAI；不是按编号连续范围猜测项目数量。`,`原始资料 SHA256、模型身份和方案签名均写入演讲者备注；改变源资料后需重新规划和导出。`]);
 if(i===1){const counts={};for(const t of r.routes)counts[t.channel]=(counts[t.channel]||0)+1;tablePage('02 当前全项目通道分工',['测量方式','FAI数','处理原则'],Object.entries(counts).map(([c,n])=>[describe(c),String(n),c==='method_review_required'?'原方法需进一步核对；不以二维替代深度/力学测试':'逐项遵循原方法构建特征，保留实际基准与跨通道配准']),[260,130,770],'工作流节点');}
 if(i===2)paragraphs('03 规格库的来源与使用范围',[`规格库版本：${src.catalog.version||src.catalog.release||'见快照'}；本方案引用 ${devices.size} 个不同候选型号。`,`厂家页面和规格书链接随每个型号列出；已下载附件状态见备注。规格值只来自当前库，不补价格、现货、未取得CAD或额外配套型号。`,'单个设备重复选用并不等于能够共用。共用需要同一物理实例、可达视场、合适光路、动作时序和安装条件，均需验证。']);
 if(i===3)paragraphs('04 当前输入与选型约束',[`图像 ROI：${val(r.constraints.roi_px)} px；最大采样间距：${val(r.constraints.max_sampling_um_px)} μm/px；最小安装间隙：${val(r.constraints.minimum_clearance_mm)} mm。以上是规划输入，不是客户计量验收标准。`,`已定义禁入区：${r.constraints.forbidden_zones?.length||0} 个；照明试拍资料：${r.constraints.lighting_evidence||'未提供'}。`,`设备型号的厂家外形、工作距、景深和像圈决定安装候选条件。必须结合实际目标面、背面可达性和治具验证光路，不预设所有项目均使用背光。`]);
 if(i===4)paragraphs('05 工站、Setup和结果归属',[`一个工站可含多组二维/三维 Setup；当前为 ${result.station_count} 站、${result.setup_count} 组。Setup 包含采集设备、装夹状态、动作及测量方法。`,`每项源 FAI 恰好由一个 Setup 输出最终结果。跨通道采集的原始数据必须先配准再合并，不能以重复结果归属代替协同测量。`,`本文件按模型给定 setup_order 列出候选执行顺序；工站并行和转运未验证。设备实例 BOM 按物理实例计数，配套部件仍需机械/电气设计。`]);
 if(i===5)paragraphs('06 当前验证状态与放行条件',[`已分配 ${result.coverage.assigned}/${result.coverage.source_total} 项；位置已标记 ${result.coverage.located} 项；安装已验证 ${result.coverage.physically_validated} 项；计量已验收 ${result.coverage.metrology_accepted} 项。`,`补齐夹具和设备 CAD 后检查遮挡、相机/镜头/光源及传感器碰撞、线缆弯曲与实际安装间隙。`,`逐项按原 MSOP 取得基准、构建特征并输出相同指标。以可追溯参考测量验证偏差、重复性、重复装夹与环境变化，计量指标由质量方确认。`,`无有效目标边/深度点/实际基准时输出测量无效，不使用默认值判合格。节拍只有在全部动作时长、转运和并行资源已定义后才能评估。`]);
}
sections.push({title:'全部工站与子方案',slide:p.slides.items.length+1});
for(const st of result.stations){
 await framePage(`${st.id} ${st.name}`,`station_${st.id}`,[`本站：${st.fai_numbers.length} 项 FAI`,`Setup：${st.setup_order.join(' → ')}`,`CT：${val(st.ct.total_s)} s`,'装夹及安装：待验证']);
 paragraphs(`${st.id}：执行顺序与工装边界`,[`顺序：${st.setup_order.join(' → ')}。源方案为候选顺序，并行执行尚未验证。`,`CT 分项：${st.ct.components.join('、')}。目标 ${val(st.ct.target_s)} s；当前总时长 ${val(st.ct.total_s)} s。`,`本站 FAI：${st.fai_numbers.join(', ')}。`,`每次姿态或装夹状态改变后，须验证是否重建实际工件基准。设备坐标不能直接替代 MSOP 的最终尺寸坐标系。`],'工站方案',st);
 for(const uid of st.setup_order){const u=st.setups.find(x=>x.id===uid);setupIndex[u.id]=p.slides.items.length+1;
  await framePage(`${st.id} / ${u.id}：${describe(u.channel)}`,`setup_${u.id}`,[`范围：${u.fai_numbers.length} 项`,`有位置：${u.location_covered_fai.length} 项`,`设备：${u.device_ids.join(' / ')||'待选型'}`,'图示：CAD标记区域'], 'Setup 子方案');
  paragraphs(`${u.id}：装夹、采集与计算`,[`FAI 清单：${u.fai_numbers.join(', ')}。`,`装夹状态（模型建议）：${u.fixture_state}`,`相对运动（模型建议）：${u.motion}`,`采集与计算顺序（模型建议）：${u.sequence}`],'Setup 子方案',u);
  if(u.optical_pair)paragraphs(`${u.id}：相机与镜头规格核算`,Object.entries(u.optical_pair).map(([k,v])=>`${pairLabels[k]||parameterLabel(k)}：${pairValue(v)}`).concat(['采样间距和瑞利分辨率均不能替代系统准确度。最终倍率、放大设置、目标边、景深与动作中的描述需按原MSOP及所选镜头核对。']),'Setup 子方案',u.optical_pair);
  paragraphs(`${u.id}：照明、路径与安装要求`,u.channel==='2d_vision'?[`照明候选：${u.device_ids.filter(id=>devices.get(id)?.kind==='light').join(' / ')||'待选型'}。模型的采集顺序保留在上一页，亮度、角度、触发和效果仍需打样。`,'背光必须具备通畅透射光路和治具开窗；环光/条光/同轴/漫射照明需确认提取的是指定物理边，避免螺纹、倒角或反光伪边。','原MSOP放大设置与镜头物理倍率分别核对；当前安装方向、位姿和观察可达性尚未验证。','先定义固定/运动主体、完整设备和夹具包络及禁入区，再求每个采集位姿与避障轨迹；当前动作文字不能下发控制器。']:['按原方法的测量面、矢量方向和采集深度选择实际观察/扫描方向；背面与遮挡区域不能通过正面二维投影替代。','确认传感器工作量程、视场、支架和扫描行程，以及治具、工件和设备的碰撞。','多通道原始点须配准到同一实际工件坐标系；按原方法构建特征后由本 Setup 输出最终结果。'],'Setup 子方案');
  paragraphs(`${u.id}：当前待补条件`,u.gaps.map((g,i)=>`${i+1}. ${g}`),'Setup 子方案');
 }
}
sections.push({title:'设备实例 BOM 与规格',slide:p.slides.items.length+1});
tablePage('设备实例 BOM：型号与共用关系',['型号 / 类别','工站 / Setup','物理实例','数量'],result.bom.map(b=>[b.id+'\n'+describe(b.kind),b.station_id+' / '+(b.setup_ids||[b.setup_id]).join(', '),b.instance_id,String(b.quantity)]),[295,210,515,140],'设备与规格','同型号分别采购计数；共用必须明确指向同一物理实例。',result.bom);
paragraphs('BOM 共用与配套部件的交接条件',['当前 BOM 为目录内设备候选，不是可采购放行清单；价格、库存、厂商CAD和未明确的配套组件不编造。','合并光源须验证照明形态、光轴方向、发光范围、均匀性和强度；模型未声明共享实例时，不能把同型号小光源直接合成一个大光源。','控制器与光源核对通道、电压、电流、功率、触发及线长。共焦/三维传感器须补控制器或计算单元、连接线、保护及安全要求。','机械电气交付还需治具、支架、运动轴、配准/标定件、工控机、IO、线缆、电源、接地和安全控制。规格与数量待系统设计确认。']);
for(const device of devices.values()){
 paragraphs(`${device.id}：厂家规格`,[`厂家：${device.vendor||'见资料'}；类别：${describe(device.kind)}。`,...Object.entries(device.spec||{}).map(([k,v])=>`${parameterLabel(k)}：${parameterValue(v)}`)],'设备与规格',device);
 paragraphs(`${device.id}：资料来源与缺失字段`,[`厂家页面：${device.source_url||'未提供'}`,`规格书：${device.datasheet_url||'未提供独立规格书链接'}`,`待核验字段：${(device.unknown_fields||[]).map(k=>({price:'价格',availability:'供货状态',vendor_CAD:'厂家机械CAD'})[k]||parameterLabel(k)).join('、')||'见厂家额定条件'}。`,`价格：${val(device.price)}；厂商机械CAD：${device.cad_status||'见快照'}。`],'资料溯源',device);
}
sections.push({title:'逐项 FAI 要求与方法',slide:p.slides.items.length+1});
for(const task of src.tasks){
 const owner=owners.get(task.number);if(!owner)throw Error('Missing owner '+task.number);const route=routes.get(task.number);faiIndex[task.number]=[];const start=p.slides.items.length+1;
 paragraphs(`FAI ${task.number}：${symbols(task)} 测量要求`,[`来源：MSOP P.${task.pages.join(', ')}；唯一结果归属 ${owner.station.id} / ${owner.setup.id}；通道 ${describe(route.channel)}。`,`原方法页标题：${[...new Set(task.methods.map(m=>m.text.split('\n')[0]))].join(' / ')}`,`特征：${task.features.join(', ')||'见原方法构建说明'}；测头 ${task.probes.join(', ')||'原文未明确'}。`,spec(task),`测量坐标系：${[...new Set(task.methods.map(m=>m.measured_system).filter(Boolean))].join(' / ')||'原文未明确'}；输出坐标系：${[...new Set(task.methods.map(m=>m.dimension_system).filter(Boolean))].join(' / ')||'原文未明确'}；程序：${[...new Set(task.methods.map(m=>m.program).filter(Boolean))].join(' / ')||'原文未明确'}。`,`位置：${describe(task.location_status)}；标记 ${task.anchors.length} 个；来源 ${task.location_origin||'未定位'}；置信度 ${({low:'低',medium:'中',high:'高'})[task.location_confidence]||task.location_confidence||'未给出'}（模型估计，非计量概率）。`],'FAI 完整附录',task);
 paragraphs(`FAI ${task.number}：采集与计算方案`,[route.proposal,...route.requirements.map((v,i)=>`条件 ${i+1}：${v}`),`原方法基准/依赖页：${[...new Set(task.dependencies.map(d=>d.page))].join(', ')||'未识别，请核对MSOP'}。详见对应演讲者备注的完整源方法及依赖内容。`],'FAI 完整附录',task);
 faiIndex[task.number]=Array.from({length:p.slides.items.length-start+1},(_,i)=>start+i);
}
sections.push({title:'MSOP 源坐标与 CAD 标记',slide:p.slides.items.length+1});
paragraphs('坐标附录：两种坐标分别解释',['下表源 XYZ/IJK 来自 MSOP，保留点名、程序、源坐标系及来源页。它们定义原测量方法的点与矢量方向。','CAD 标记是当前 Viewer 显示模型上的位置；位置尚未复核时不能代替实际工件坐标。源 Z 不会直接复制为 CAD Z。','同名点按程序/坐标系分开列出。出现冲突或语境歧义时保留原值并标记，不能默认合并。','原始坐标精度按提取值保留；完整 occurrences、源语境及坐标配准条件见备注。所有点保留，不以首尾点代替完整采集。'],'坐标与配准');
const pointGroups=new Map();
for(const task of src.tasks)for(const pt of task.source_points){const key=JSON.stringify([pt.name,pt.program,pt.coordinate_system,pt.xyz,pt.ijk]);if(!pointGroups.has(key))pointGroups.set(key,{...pt,tasks:[]});pointGroups.get(key).tasks.push(task.number);}
const frameGroups=new Map();for(const pt of pointGroups.values()){const key=`${pt.program||'程序未明确'} / ${pt.coordinate_system||'坐标系未明确'}`;if(!frameGroups.has(key))frameGroups.set(key,[]);frameGroups.get(key).push(pt);}
for(const [frame,points] of frameGroups){
 const rows=points.map(pt=>[pt.name,pt.xyz.join(', '),pt.ijk?.join(', ')||'未提供',`P.${pt.pages.join(', ')}\nFAI ${pt.tasks.join(', ')}${pt.coordinate_conflict||pt.scope_ambiguous?'\n冲突/语境待核对':''}`]);
 tablePage('MSOP 源点：'+frame,['点名','源 XYZ / mm','IJK','来源与关联 FAI'],rows,[290,310,240,320],'坐标与配准','源坐标未直接用作 CAD Z 或设备轴位置。',points);
}
const anchorRows=src.tasks.flatMap(t=>t.anchors.map((a,i)=>[`FAI ${t.number} / ${i+1}`,a.position.map((x,i)=>`${['X','Y','Z'][i]} ${Number(x.toFixed(6))}`).join('\n'),a.face_id.replace(/^[^:]+:instance(\d+):face(\d+)$/,'实例 $1 / 面 $2'),describe(t.location_status)]));
if(anchorRows.length)tablePage('CAD 标记点：当前 Viewer 位置',['FAI / 点','CAD XYZ / mm','关联 CAD 面','位置状态'],anchorRows,[180,360,425,195],'坐标与配准','显示坐标保留6位小数；完整值及面身份见备注。',src.tasks.map(t=>({number:t.number,anchors:t.anchors})));
sections.push({title:'验证与工程交付清单',slide:p.slides.items.length+1});
paragraphs('验证与交付：从方案到设备',[
 '方案人员：核对全部 FAI 的原物理特征、符号、公差、采点、基准及拟合方法。每项须形成 AOI 方法与海克斯康原方法的对应记录。',
 '机械人员：以实际 CAD 定义设备/镜头/光源/夹具/线缆位置与运动主体，检查工作距、观察光路、遮挡、可安装性及全行程避障。',
 '电气与软件人员：定义触发、照明切换、运动到位、稳定等待、曝光/扫描、数据配准和异常互锁。留存工件、标定版本、原始数据及计算记录。',
 '生产人员：按真实上下料、装夹、运动、稳定、曝光、扫描、计算与转运时间评估 CT；验证各 Setup 的资源共享和并行冲突。',
 '品质人员：认可误差预算与验收指标，评估可追溯参考对比、重复测量、重复装夹、不同位置和温度。方法等效及计量通过前不得生产放行。'
],'工程交付');
paragraphs('当前项目未完成项的处理顺序',[`位置缺失：${result.coverage.unlocated_fai.join(', ')||'无'}。先补三件套配准与指定物理边，保留模型建议的唯一结果归属。`,`方法需核对：${r.routes.filter(t=>t.channel==='method_review_required').map(t=>t.number).join(', ')||'无独立核对通道；其余通道仍需验证方法等效'}。`,'安装与动作：补夹具和设备几何，定义采集位姿、运动轴及路径，再检查视线和全程包络。','计量与节拍：定义质量验收条件、误差预算及动作时间，完成打样和参考验证，再定版BOM。'],'工程交付');
sections.push({title:'索引',slide:p.slides.items.length+1});
tablePage('章节索引',['章节','起始页'],sections.map(s=>[s.title,String(s.slide)]),[1000,160],'索引');
tablePage('Setup 索引',['Setup','工站 / 通道','起始页'],Object.entries(setupIndex).map(([id,n])=>{const owner=result.stations.find(s=>s.setups.some(u=>u.id===id));return[id,owner.id+' / '+describe(owner.setups.find(u=>u.id===id).channel),String(n)];}),[220,770,170],'索引');
tablePage('FAI 索引：逐项要求与方法',['FAI','唯一结果归属','通道','页码'],src.tasks.map(t=>{const o=owners.get(t.number),pages=faiIndex[t.number];return[String(t.number),o.station.id+' / '+o.setup.id,describe(o.setup.channel),pages[0]===pages.at(-1)?String(pages[0]):`${pages[0]}–${pages.at(-1)}`];}),[140,220,440,360],'索引');
await fs.writeFile(path.join(dir,'.build/layout-policy.json'),JSON.stringify({font,fontPath,manualWrap:true,bodySizePx:24,minTableSizePx:23,tableOwners:tables},null,2));
const candidate=path.join(dir,'.build/full-workflow.pptx');console.log('AUTHORING',p.slides.items.length);await(await PresentationFile.exportPptx(p)).save(candidate);
const resultCheck=[];
const ownerBatches=Array.from({length:Math.max(1,Math.ceil(tables.length/40))},(_,i)=>tables.slice(i*40,(i+1)*40));
for(const [pass,owners] of ownerBatches.entries()){const lastPass=pass===ownerBatches.length-1;resultCheck.push(await finalizePresentation({workspaceDir:dir,candidatePath:candidate,finalPath:lastPass?path.join(dir,'output/station-workflow.pptx'):path.join(dir,'.build/verified',`pass-${pass+1}.pptx`),pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...owners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:owners,requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(dir,'.build',`validation-${pass+1}.json`)}));console.log('VALIDATED',pass+1,ownerBatches.length);}
// Full visual review is required; each exported preview is retained privately.
for(const [i,slide] of p.slides.items.entries()){const png=await p.export({slide,format:'png',scale:1});await fs.writeFile(path.join(dir,'.build',`slide-${String(i+1).padStart(3,'0')}.png`),new Uint8Array(await png.arrayBuffer()));if(i%20===0)console.log('RENDERED',i+1);}
await fs.writeFile(path.join(dir,'deck-manifest.json'),JSON.stringify({slideCount:p.slides.items.length,sourcePlan:r.id,sourceFAICount:src.tasks.length,nodeIndex,setupIndex,faiIndex,sourcePointCount:pointGroups.size,cadAnchorCount:anchorRows.length,tableSlides:tables,modelIdentity:r.model_identity,modelExplanationAttempt:src.model_attempt,result:resultCheck},null,2));console.log('COMPLETE',p.slides.items.length);
