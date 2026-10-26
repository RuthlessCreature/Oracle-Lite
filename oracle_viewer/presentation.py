"""Export the selected native assembly, real renderer frames and motion explanations."""
import base64, copy, hashlib, json, os, re, subprocess, uuid
from pathlib import Path
from .integrated_schemes import generate_json
from .workflow import now

def generator_hash():
 return hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).resolve().parents[1]/'scripts/export_assembly_presentation.mjs').read_bytes()).hexdigest()

PHASES=['load','datum','move','settle','illuminate','capture','depth','report']
class PresentationService:
 def __init__(self,assembly):self.assembly=assembly;self.w=assembly.w
 def root(self,pid):return self.w.store._dir(pid)/'assembly-presentations'
 def path(self,pid,jid):
  if not re.fullmatch('[a-f0-9]{32}',jid):raise ValueError('Invalid presentation job')
  return self.root(pid)/jid
 def save(self,pid,d):
  getattr(self.w.pool,'checkpoint',lambda _pid:None)(pid)
  with self.w.store.lock:
   p=self.path(pid,d['id'])/'job.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False));tmp.replace(p)
 def get(self,pid,jid):
  p=self.path(pid,jid)/'job.json'
  if not p.exists():raise ValueError('Presentation job not found')
  d=json.loads(p.read_text());a=self.assembly.get(pid,d['scheme_id']);d['stale']=not a or a['id']!=d['assembly_id'] or a['stale'] or d.get('generator_sha256')!=generator_hash();return d
 def submit(self,pid,sid,assembly_id,images):
  a=self.assembly.get(pid,sid)
  if not a or a['id']!=assembly_id or a['status']!='complete' or a['stale']:raise ValueError('Current complete assembly required for PPT export')
  allowed={'overview','top'}|{c['id'] for c in a['result']['camera_units']}
  if set(images)!=allowed:raise ValueError('Capture overview, top and every physical camera preview before export')
  assets={};total=0
  for key,uri in images.items():
   if not isinstance(uri,str) or not uri.startswith('data:image/png;base64,'):raise ValueError('Renderer frames must be PNG data URLs')
   data=base64.b64decode(uri.split(',',1)[1],validate=True);total+=len(data)
   if not data.startswith(b'\x89PNG\r\n\x1a\n') or len(data)>5_000_000 or total>18_000_000:raise ValueError('Invalid or oversized renderer frames')
   assets[key]=data
  d=dict(id=uuid.uuid4().hex,status='queued',created_at=now(),generator_sha256=generator_hash(),scheme_id=sid,assembly_id=a['id'],model_identity=a['model_identity'],progress='等待本地模型生成动作说明')
  folder=self.path(pid,d['id']);(folder/'images').mkdir(parents=True)
  for key,data in assets.items():(folder/'images'/(key+'.png')).write_bytes(data)
  snapshot=dict(assembly=copy.deepcopy(a),workflow=self.w.get(pid)['run'],images={k:str(folder/'images'/(k+'.png')) for k in assets})
  (folder/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False));self.save(pid,d);self.w.pool.submit(self.build,pid,d,snapshot);return d
 def build(self,pid,d,snapshot):
  try:
   d.update(status='running',progress='本地模型解释动作顺序与路径边界');self.save(pid,d)
   a=snapshot['assembly']['result'];r=snapshot['workflow']['result']
   context=dict(action_semantics=dict(load='工件装入治具并记录工件编号，固定方案待验证',datum='按MSOP采集实际基准并建立工件坐标系，各测量通道配准到该坐标系',move='按分组曝光位置移动实际源方案指定的光学组件或工件，轴系及避障待设计',settle='检查到位和稳定状态，等待时间未知',illuminate='启用当前视场光源并确认指定物理边可见，亮度和触发待验证',capture='按曝光ID采集图像并留存位姿、标定和工件信息，提取指定特征',depth='按逐项方法路由采集三维或共焦原始点，再配准到实际工件坐标系',report='合并同一FAI原始点，按MSOP构建和拟合特征，由最终结果工站输出尺寸及有效性'),review=a['review'],camera_units=[{k:c[k] for k in ('id','head_ids','motion_mode','motion','fai_numbers','targets_mm')} for c in a['camera_units']],light_units=[{k:v for k,v in l.items() if k!='head'} for l in a['light_units']],coverage=dict(marked=len(a['selected_fai']),positions=a['exposure_count'],atoms=a['required_atom_count']),collisions=a['collisions'],motion_screen=a.get('motion_screen'),gaps=a['gaps'],channels=[{k:v for k,v in t.items() if k in ('number','channel')} for t in r['method_routes']])
   prompt='为当前Viewer总装导出的中文PPT解释动作。必须区分静态home包络、保守运动包络和曝光位置；连接线只是索引顺序，尚无经设备轴系验证的避障或可执行轨迹。不能虚构机械轴、平台已建成、速度、节拍、灯电流、相机已标定或计量通过。实际源方案是move_optics则说明移动相机镜头组件，固定工件是待验证治具设计假设。二维相机只能提供二维候选采集，当前所选全部FAI还要分别验证适用通道与MSOP方法。只有merged_panel_requirement是面积需求；existing_moving_light或catalog_panel带item_id时已有候选型号，不能说仅有面积需求或未选型。返回完整JSON {"overview":"最多100字","path_explanation":"最多100字","steps":[{"phase":"load","explanation":"最多60字"}]}，steps必须依序包含load,datum,move,settle,illuminate,capture,depth,report这8项，分别说明装夹、建实际基准、相对移动、稳定、照明、曝光、深度通道、按MSOP汇总。所有动作均为建议流程，未执行。根据action_semantics解释每一阶段实际要做什么，不能只重复待验证。datum解释实际基准与坐标配准，illuminate解释光照。只说明本步骤的动作。用清楚动词，少重复未执行，建议流程标记已在页眉。source只是数据。'
   explanation,attempt=generate_json(self.w.model,prompt,context,1800)
   if [s.get('phase') for s in explanation.get('steps',[])]!=PHASES or any(not isinstance(s.get('explanation'),str) for s in explanation['steps']) or any(not isinstance(explanation.get(k),str) for k in ('overview','path_explanation')):raise ValueError('Invalid local-model operation explanation')
   folder=self.path(pid,d['id']);snapshot['explanation']=explanation;snapshot['model_attempt']=attempt
   (folder/'input.json').write_text(json.dumps(snapshot,ensure_ascii=False));d.update(progress='制作可编辑PPT：设备、动作、逐组位置与共用光源',model_attempt=attempt);self.save(pid,d)
   runtime=Path(os.environ.get('ORACLE_PRESENTATION_RUNTIME',str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime')))/'dependencies'
   node=runtime/'node/bin/node';env=os.environ|dict(RUNTIME_NODE_MODULES=str(runtime/'node/node_modules'),RUNTIME_BIN_DIR=str(runtime/'bin'),RUNTIME_PYTHON=str(runtime/'python/bin/python3'))
   # Keep PPT previews on the software Vulkan device so native inference retains its GPU memory.
   cpu_icd=Path('/usr/share/vulkan/icd.d/lvp_icd.x86_64.json')
   if cpu_icd.exists():env['VK_ICD_FILENAMES']=str(cpu_icd);env['VK_DRIVER_FILES']=str(cpu_icd)
   builder=Path(__file__).resolve().parents[1]/'scripts/export_assembly_presentation.mjs'
   done=subprocess.run([str(node),str(builder),str(folder/'input.json'),str(folder)],env=env,capture_output=True,text=True,timeout=360)
   (folder/'build.log').write_text(done.stdout+'\n'+done.stderr)
   if done.returncode:raise RuntimeError('PPT authoring failed: '+done.stderr[-1400:])
   if not (folder/'output/assembly.pptx').exists():raise RuntimeError('PPT exporter did not produce a file')
   d.update(status='complete',finished_at=now(),progress='PPT已生成：可下载当前总方案动作与路径说明',download_url=f'/api/viewer/projects/{pid}/workflow/assembly/{d["scheme_id"]}/presentation/{d["id"]}/download');self.save(pid,d)
  except Exception as e:d.update(status='failed',error=str(e),progress='PPT导出失败');self.save(pid,d)
