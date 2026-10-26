"""Full source-wide station review deck; decisions remain the local model's."""
import base64
import copy
import hashlib
import json
import os
import re
import subprocess
import uuid
import time
from pathlib import Path
from .integrated_schemes import generate_json, DecisionError
from .workflow import library, now

BUILDER=Path(__file__).resolve().parents[1]/'scripts/export_station_presentation.mjs'
NODES=['requirements','regions','library','configuration','stations','validation']

def generator_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+BUILDER.read_bytes()+Path(__file__).with_name('static').joinpath('viewer.js').read_bytes()+Path(__file__).with_name('static').joinpath('copy.js').read_bytes()).hexdigest()

def frame_keys(record):
    return {'overview'}|{'station_'+s['id'] for s in record['result']['stations']}|{'setup_'+u['id'] for s in record['result']['stations'] for u in s['setups']}

def validated_frames(images,record):
    if set(images)!=frame_keys(record):
        raise ValueError('必须保存总览、全部工站及全部 Setup 的实际 Viewer 渲染图，不能漏项')
    total=0;frames={}
    for key,uri in images.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+',key):raise ValueError('Invalid renderer frame identifier')
        if not isinstance(uri,str) or not uri.startswith('data:image/png;base64,'):raise ValueError('Renderer frame must be PNG')
        data=base64.b64decode(uri.split(',',1)[1],validate=True);total+=len(data)
        if not data.startswith(b'\x89PNG\r\n\x1a\n') or len(data)>5_000_000 or total>80_000_000:
            raise ValueError('Invalid or oversized renderer frames')
        frames[key]=data
    return frames

def validate_node(node,node_id):
    if not isinstance(node,dict) or node.get('id')!=node_id:raise ValueError('Local node identity mismatch')
    for key in ('inputs','actions','outputs','checks'):
        values=node.get(key)
        if not isinstance(values,list) or not 2<=len(values)<=4 or any(not isinstance(v,str) or not v.strip() for v in values):
            raise ValueError('Incomplete local node '+node_id+' field '+key)
    return node


def explain_node(model,node_id,prompt,context,cache_dir,source_key,plan_id,progress):
    key=hashlib.sha256(json.dumps(dict(prompt=prompt,context=context,source_key=source_key,plan_id=plan_id),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cache_dir.mkdir(parents=True,exist_ok=True);cache=cache_dir/(key+'.json')
    if cache.exists():
        try:
            saved=json.loads(cache.read_text());validate_node(saved['node'],node_id)
            if saved.get('key')==key and not saved['attempt'].get('truncated',True):
                return saved['node'],[saved['attempt']|dict(cache_hit=True,cache_key=key)]
        except (ValueError,KeyError,TypeError):pass  # Regenerate with complete evidence; never accept a damaged cache.
    attempts=[];error=None
    for retry in range(3):
        if retry:progress(retry)
        try:
            node,attempt=generate_json(model,prompt+(f' 上次完整JSON校验失败：{error}。重新返回本节点的完整JSON，字符串内部换行使用JSON转义。' if error else ''),context,2200)
            attempts.append(attempt|dict(node_id=node_id,cache_hit=False));validate_node(node,node_id)
            temp=cache.with_suffix('.tmp');temp.write_text(json.dumps(dict(key=key,node=node,attempt=attempts[-1]),ensure_ascii=False));temp.replace(cache)
            return node,attempts
        except DecisionError as exc:
            attempts.append(exc.attempt|dict(node_id=node_id,validation_error=str(exc)));error=str(exc)
        except ValueError as exc:
            error=str(exc)
            if attempts:attempts[-1]['validation_error']=error
    raise DecisionError('Local workflow node '+node_id+' failed after complete JSON retries',dict(truncated=False,node_attempts=attempts))


class StationPresentationService:
    def __init__(self,stations):self.stations=stations;self.store=stations.store
    def path(self,pid,jid):
        if not re.fullmatch('[a-f0-9]{32}',jid):raise ValueError('Invalid presentation job')
        return self.store._dir(pid)/'station-presentations'/jid
    def save(self,pid,d):
        getattr(self.stations.pool,'checkpoint',lambda _pid:None)(pid)
        with self.store.lock:
            path=self.path(pid,d['id'])/'job.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False));tmp.replace(path)
    def get(self,pid,jid):
        d=json.loads((self.path(pid,jid)/'job.json').read_text());record=self.stations.get(pid)
        d['stale']=not record or record['id']!=d['plan_id'] or record['stale'] or d['generator_sha256']!=generator_hash()
        return d
    def latest(self,pid):
        root=self.store._dir(pid)/'station-presentations'
        candidates=sorted(root.glob('*/job.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for path in candidates:
            job=self.get(pid,path.parent.name)
            if not job['stale']:return job
        return None
    def submit(self,pid,plan_id,images):
        record=self.stations.get(pid)
        if not record or record['id']!=plan_id or record['status']!='complete' or record['stale']:
            raise ValueError('需要当前版本完整工站方案，资料变化后请重新规划')
        tasks=record['task_snapshot'];owners=[n for s in record['result']['stations'] for u in s['setups'] for n in u['fai_numbers']]
        if sorted(owners)!=sorted(t['number'] for t in tasks) or len(set(owners))!=len(tasks):raise ValueError('PPT source FAI scope or ownership is incomplete')
        frames=validated_frames(images,record)
        existing=self.latest(pid)
        if existing and existing['plan_id']==plan_id and existing['status'] in ('queued','running'):return existing
        d=dict(id=uuid.uuid4().hex,plan_id=plan_id,status='queued',created_at=now(),generator_sha256=generator_hash(),model_identity=record['model_identity'],progress='等待本地模型编写六个节点的工程说明')
        folder=self.path(pid,d['id']);(folder/'images').mkdir(parents=True)
        for key,data in frames.items():(folder/'images'/(key+'.png')).write_bytes(data)
        project=self.store.get(pid)
        # Parse explicit original coordinates without inventing a source-to-CAD transform.
        from .msop_coordinates import _coordinate_rows
        from .candidate_mining import extract_named_points
        pages=(self.store.active_msop(project) or {}).get('method_pages',[]);named=extract_named_points(pages)
        details=copy.deepcopy(tasks)
        for task in details:task['source_points']=_coordinate_rows(task.get('methods'),task.get('dependencies'),task.get('features',[]),pages,named)
        cat=library();ids={b['id'] for b in record['result']['bom']}
        snapshot=dict(record=copy.deepcopy(record),project={k:project.get(k) for k in ('id','title','drawing_no','revision')},tasks=details,catalog={k:v for k,v in cat.items() if k!='items'}|dict(items=[i for i in cat['items'] if i['id'] in ids]),images={k:str(folder/'images'/(k+'.png')) for k in frames})
        (folder/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False));self.save(pid,d);self.stations.pool.submit(self.build,pid,d,snapshot);return d
    def build(self,pid,d,snapshot):
        try:
            d.update(status='running',progress='本地模型逐节点说明输入、处理动作、输出与交接条件');self.save(pid,d)
            record=snapshot['record'];result=record['result']
            context=dict(project=snapshot['project'],coverage=result['coverage'],constraints=record['constraints'],review=result['review'],stations=[dict(id=s['id'],name=s['name'],setup_order=s['setup_order'],ct=s['ct'],setups=[{k:u.get(k) for k in ('id','channel','device_ids','fixture_state','motion','sequence','gaps')} for u in s['setups']]) for s in result['stations']],source_documents=record['source_summary']['documents'],model_identity=record['model_identity'],selected_catalog=snapshot['catalog']['items'],provenance_rules=['原始XYZ/IJK点表来自MSOP，而CAD仅提供几何/面/显示标记；传感器采集点和IJK测量方向不是同一个概念。','实际安装位姿、路径、光路可见性及测量区域边界尚未确定；输入ROI与采样仅支持规格初算，不代表区域覆盖或安装已满足。','动作中软件/程序放大设置不等于镜头物理倍率，需分别核对来源，不得声称3.5x是镜头已确认倍率。','模型提出的固定/运动与顶光/环光等描述均待核对；候选BOM未列出的器件不得描述为已有。'])
            nodes=[];attempts=[]
            node_names=['测量要求','测量区域与测量方式','厂家规格库','设备选型与采集安排','工站/Setup/BOM与节拍','安装/计量验证及交付']
            for index,node_id in enumerate(NODES):
                d.update(progress=f'本地模型编写节点 {index+1}/6：{node_names[index]}');self.save(pid,d)
                prompt=f'为Oracle Viewer工程评审PPT说明当前节点 {node_id}（{node_names[index]}）。根据已给定本地工站方案说明输入资料、处理动作、输出结果及交接检查条件，逐句写清方案人员、机械、电气、生产、品质实际要做什么。此轮仅返回该一个节点，不返回其它节点或整个流程。所有规划均待验证；安装位置、测量面边界、观察可达性、动作轨迹、计量及CT不能描述成已经完成。严格保留给定候选型号、工站/Setup数、FAI归属及状态。原XYZ/IJK来自MSOP，CAD提供几何和显示标记；IJK是原方法方向，不是传感器直接采集的数值。程序/软件放大设置与镜头物理倍率需分别核对。实际通道适用性须按指标和基准评估，不能把2D/3D分别机械限定为X/Y和Z。实物计量要与可追溯参考测量对比，不能只凭MSOP源数据验证。设备能力仅用selected_catalog已有规格；缺少的器件、照明试拍、误差预算和动作时间标记待补。遵守provenance_rules，但正文要写工程动作和交付条件，不写“严禁”“不声称”等AI自我约束。待生成资料要用待编制/待提供等词区分当前已有输出。返回完整纯JSON {{"id":"{node_id}","inputs":["..."],"actions":["..."],"outputs":["..."],"checks":["..."]}}，每组2至4条具体、完整、清楚的中文句子，避免原样堆内部字段名，不输出思考。'
                node,attempt=explain_node(self.stations.model,node_id,prompt,context,
                    self.store._dir(pid)/'station-ppt-node-cache',record['source_key'],record['id'],
                    lambda retry:d.update(progress=f'本地模型编写节点 {index+1}/6：{node_names[index]} · 重试 {retry}') or self.save(pid,d))
                nodes.append(node);attempts.extend(attempt)
            explanation=dict(nodes=nodes);attempt=dict(truncated=False,node_attempts=attempts,
                encoding='six_complete_local_node_decisions',cached_nodes=sum(bool(a.get('cache_hit')) for a in attempts))
            snapshot.update(explanation=explanation,model_attempt=attempt);folder=self.path(pid,d['id'])
            (folder/'input.json').write_text(json.dumps(snapshot,ensure_ascii=False));d.update(progress='制作完整工程PPT：六节点、全部工站/Setup、BOM及逐项FAI附录',model_attempt=attempt);self.save(pid,d)
            runtime=Path(os.environ.get('ORACLE_PRESENTATION_RUNTIME',str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime')))/'dependencies'
            env=os.environ|dict(RUNTIME_NODE_MODULES=str(runtime/'node/node_modules'),RUNTIME_BIN_DIR=str(runtime/'bin'),RUNTIME_PYTHON=str(runtime/'python/bin/python3'))
            cpu_icd=Path('/usr/share/vulkan/icd.d/lvp_icd.x86_64.json')
            if cpu_icd.exists():env.update(VK_ICD_FILENAMES=str(cpu_icd),VK_DRIVER_FILES=str(cpu_icd))
            logpath=folder/'build.log';started=time.monotonic();last_progress=''
            with logpath.open('w') as log:
                process=subprocess.Popen([str(runtime/'node/bin/node'),str(BUILDER),str(folder/'input.json'),str(folder)],env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
                while process.poll() is None:
                    if time.monotonic()-started>2400:
                        process.kill();process.wait();raise RuntimeError('PPT导出超时，未交付不完整文件')
                    lines=logpath.read_text(errors='replace').splitlines()
                    stage=next((line for line in reversed(lines) if line.startswith(('AUTHORING ','VALIDATED ','RENDERED '))),None)
                    if stage and stage!=last_progress:
                        parts=stage.split();last_progress=stage
                        message={'AUTHORING':f'已编排 {parts[1]} 页，正在校验完整文件','VALIDATED':f'表格与版式校验 {parts[1]} / {parts[2] if len(parts)>2 else "?"}','RENDERED':f'正在渲染检查第 {parts[1]} 页'}[parts[0]]
                        d.update(progress=message);self.save(pid,d)
                    time.sleep(1)
            if process.returncode:raise RuntimeError('PPT authoring failed: '+logpath.read_text(errors='replace')[-1800:])
            if not (folder/'output/station-workflow.pptx').exists():raise RuntimeError('PPT file missing')
            manifest=json.loads((folder/'deck-manifest.json').read_text())
            if sorted(manifest['faiIndex'])!=sorted(str(t['number']) for t in snapshot['tasks']):raise ValueError('PPT omitted source FAI')
            d.update(status='complete',finished_at=now(),slide_count=manifest['slideCount'],progress=f'完整方案PPT已生成，共 {manifest["slideCount"]} 页',download_url=f'/api/viewer/projects/{pid}/workflow/stations/presentation/{d["id"]}/download');self.save(pid,d)
        except Exception as e:
            if isinstance(e,DecisionError):d['failed_model_attempt']=e.attempt
            d.update(status='failed',error=str(e),progress='完整方案PPT导出失败');self.save(pid,d)
