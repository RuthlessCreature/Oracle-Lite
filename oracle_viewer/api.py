from __future__ import annotations
import json
import hashlib
import subprocess
import tempfile
import sys
import traceback
import csv
import io
import uuid
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from threading import RLock, local, Event
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, confloat, model_validator
from typing import Literal
from starlette.concurrency import run_in_threadpool
from .store import ViewerStore
from .assistant import answer_project
from .marking_status import marker_coverage, unmarked_diagnostics
from .annotations import MAX_ANNOTATION_POINTS


class ViewerTaskQueue:
    """Track work submitted to the viewer's shared single-worker executor."""
    def __init__(self, executor, store):
        self.executor, self.store = executor, store
        self.lock = RLock()
        self.rows = {}
        self.context = local()

    class TaskCancelled(BaseException):
        pass

    def checkpoint(self, project_id=None):
        task_id = getattr(self.context, 'task_id', None)
        if not task_id: return
        with self.lock: row = self.rows.get(task_id)
        if not row or (project_id and row.get('project_id') != project_id): return
        if row.get('cancel_requested'): raise self.TaskCancelled()
        if row.get('pause_requested'):
            with self.lock:
                row['status'] = 'paused'; row['progress'] = '已暂停；可继续'
            row['resume_event'].wait()
            if row.get('cancel_requested'): raise self.TaskCancelled()
            with self.lock:
                row['status'] = 'running'; row['pause_requested'] = False
                row['progress'] = '已继续执行'

    def _schedule(self, row):
        def tracked():
            self.context.task_id = row['id']
            with self.lock:
                if row.get('deleted'): return
                row['status'] = 'running'; row['started_at'] = __import__('time').time()
                row['progress'] = '正在执行'
            try:
                self.checkpoint()
                result = row['fn'](*row['args'], **row['kwargs'])
                self.checkpoint()
                task_record = next((arg for arg in row.get('args', ())
                                    if isinstance(arg, dict) and isinstance(arg.get('id'), str)
                                    and isinstance(arg.get('status'), str)), None)
                if task_record:
                    actual = task_record['status']
                    with self.lock:
                        if actual == 'failed':
                            row['status'] = 'failed'; row['error'] = task_record.get('error', '任务未通过结果校验')
                            row['progress'] = '执行结束但方案未通过校验'
                        elif actual == 'cancelled':
                            row['status'] = 'cancelled'; row['progress'] = '任务已停止'
                        elif actual == 'complete':
                            row['status'] = 'complete'; row['progress'] = '任务已完成'
                        else:
                            row['status'] = 'failed'; row['error'] = '执行函数已返回，但任务记录仍为 '+actual
                            row['progress'] = '任务未完成'
                elif row['kind']=='run_marking' and row.get('project_id'):
                    marking=self.store.get(row['project_id']).get('local_model_marking') or {}
                    actual=marking.get('status')
                    with self.lock:
                        row['status']={'completed':'complete','partial':'partial','blocked':'blocked','failed':'failed'}.get(actual,'failed')
                        row['progress']=marking.get('message') or '本轮定位已结束'
                        if marking.get('error'):row['error']=marking['error']
                else:
                    with self.lock:
                        row['status'] = 'complete'; row['progress'] = '任务已完成'
                return result
            except self.TaskCancelled:
                self._mark_cancelled_record(row)
                with self.lock: row['status'] = 'cancelled'
                return None
            except BaseException as exc:
                with self.lock:
                    row['status'] = 'failed'; row['error'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                self.context.task_id = None
                with self.lock:
                    if row['status'] == 'running': row['status'] = 'complete'
                    row['finished_at'] = __import__('time').time()
                self._trim()
        row['future'] = self.executor.submit(tracked)

    def _mark_cancelled_record(self, row):
        service = row.get('service'); project_id = row.get('project_id')
        if not service or not project_id or service.__class__.__name__ not in ('WorkflowService', 'StationService'):
            return
        record = next((arg for arg in row.get('args', ()) if isinstance(arg, dict) and arg.get('id')), None)
        if not record: return
        record['status'] = 'cancelled'; record['cancelled_at'] = __import__('time').time()
        record['progress'] = dict(record.get('progress') or {}, phase='cancelled', message='已删除；保留已完成的部分结果')
        try:
            path = service.path(project_id); temp = path.with_suffix('.tmp')
            with self.store.lock:
                temp.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False)); temp.replace(path)
        except Exception:
            pass

    def control(self, task_id, action):
        with self.lock:
            row = self.rows.get(task_id)
            if not row: raise KeyError('任务已不存在')
            if action == 'pause':
                if row['status'] == 'queued':
                    if row['future'].cancel(): row['status'] = 'paused'; row['progress'] = '已暂停排队任务'
                    else: row['pause_requested'] = True; row['resume_event'].clear()
                elif row['status'] == 'running':
                    row['pause_requested'] = True; row['resume_event'].clear()
                    row['progress'] = '当前模型调用结束后暂停'
                else: raise ValueError('当前任务状态不能暂停')
            elif action == 'resume':
                if row['status'] == 'paused':
                    row['resume_event'].set()
                    if row.get('future') and row['future'].cancelled():
                        row['status'] = 'queued'; row['progress'] = '等待工作线程'; self._schedule(row)
                    elif row.get('future') and row['future'].done():
                        raise ValueError('任务已结束，不能继续')
                else: raise ValueError('当前任务未暂停')
            elif action == 'delete':
                row['deleted'] = True; row['cancel_requested'] = True; row['resume_event'].set()
                if row['status'] == 'queued': row['future'].cancel(); row['status'] = 'cancelled'
                elif row['status'] == 'running': row['progress'] = '已请求停止；将在下一个任务检查点退出'
                else: row['status'] = 'cancelled'
            return {k:v for k,v in row.items() if k not in ('fn','args','kwargs','future','resume_event','service')}

    def __getattr__(self, name):
        return getattr(self.executor, name)

    def submit(self, fn, *args, **kwargs):
        task_id = uuid.uuid4().hex
        name = getattr(fn, '__name__', fn.__class__.__name__)
        owner = getattr(getattr(fn, '__self__', None), '__class__', type(None)).__name__
        project_id = next((x for x in args if isinstance(x, str) and len(x) == 32), None)
        labels = {
            'run_marking': 'FAI 自动标记', 'run': '本地模型任务',
            'work': '本地学习任务', 'export': '方案 PPT 导出',
        }
        owner_labels={'WorkflowService':'拍摄方案编制','StationService':'工站 / Setup 规划',
                      'AssemblyService':'整机布置计算','PresentationService':'方案 PPT 导出',
                      'StationPresentationService':'Station / Setup PPT 导出'}
        row = dict(id=task_id, project_id=project_id, kind=name,
                   label=owner_labels.get(owner,labels.get(name, name.replace('_', ' '))),
                   status='queued', queued_at=__import__('time').time(),
                   progress='等待工作线程', fn=fn, args=args, kwargs=kwargs,
                   resume_event=Event(), service=getattr(fn, '__self__', None),
                   record_id=next((arg['id'] for arg in args if isinstance(arg,dict)
                                   and isinstance(arg.get('id'),str) and isinstance(arg.get('status'),str)),None))
        with self.lock:
            self.rows[task_id] = row

        self._schedule(row)
        return row['future']

    def _trim(self):
        with self.lock:
            finished = sorted((r for r in self.rows.values() if r['status'] not in ('queued', 'running')),
                              key=lambda r: r.get('finished_at', 0))
            for row in finished[:-40]:
                self.rows.pop(row['id'], None)

    def snapshot(self):
        with self.lock:
            rows = [{k:v for k,v in row.items() if k not in ('fn','args','kwargs','future','resume_event','service')}
                    for row in self.rows.values() if not row.get('deleted')]
        for row in rows:
            try:
                project = self.store.get(row['project_id']) if row.get('project_id') else None
                row['project'] = (project or {}).get('drawing_no') or row.get('project_id') or '全局任务'
            except Exception:
                row['project'] = row.get('project_id') or '全局任务'
        rows.sort(key=lambda r: (0 if r['status'] == 'running' else 1 if r['status'] == 'queued' else 2,
                                 r.get('queued_at', 0)))
        for index, row in enumerate(rows):
            row['queue_position'] = index + 1 if row['status'] == 'queued' else None
        return dict(active=sum(r['status'] == 'running' for r in rows),
                    waiting=sum(r['status'] == 'queued' for r in rows),
                    paused=sum(r['status'] == 'paused' for r in rows), tasks=rows)


def automatic_marking_sources_ready(project, msop):
    """An active CAD plus indexed MSOP is enough to start local marking.

    XYZ registration can be empty; the per-FAI local vision route handles
    feature-only items. Missing coordinates must not prevent the job from
    reaching that model-controlled route.
    """
    model=project.get('model') or {}
    return (model.get('status') in ('ready','preview') and bool(model.get('source_sha256'))
            and bool(msop and msop.get('sha256')) and bool(msop.get('method_pages')))


class PresentationInput(BaseModel):
    assembly_id: str = Field(pattern='^[a-f0-9]{32}$')
    images: dict[str, str] = Field(min_length=2,max_length=10)


class StationPresentationInput(BaseModel):
    plan_id: str = Field(pattern='^[a-f0-9]{32}$')
    images: dict[str, str] = Field(min_length=1,max_length=128)


class ForbiddenZone(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    min_mm: list[float] = Field(min_length=3,max_length=3)
    max_mm: list[float] = Field(min_length=3,max_length=3)


    @model_validator(mode='after')
    def valid_box(self):
        import math
        if not all(math.isfinite(v) for v in self.min_mm+self.max_mm) or any(a>=b for a,b in zip(self.min_mm,self.max_mm)):
            raise ValueError('Forbidden zone must be finite with min_mm < max_mm on every axis')
        return self


class WorkflowInput(BaseModel):
    mode: Literal['automatic','manual'] = 'automatic'
    roi_mode: Literal['maximum','fixed'] = 'maximum'
    fai_numbers: list[int] = Field(default_factory=list, max_length=2000)
    minimum_clearance_mm: confloat(ge=0, le=100, allow_inf_nan=False) = 2.0
    roi_px: list[int] = Field(default_factory=lambda: [2400, 2000], min_length=2, max_length=2)
    feature_radius_mm: confloat(gt=0, le=100, allow_inf_nan=False) = 3.0
    max_sampling_um_px: confloat(gt=0, le=100, allow_inf_nan=False) = 10.0
    normal_axis: Literal['auto','+Z','-Z','other'] = Field(default='auto', description='Deprecated direction input; production planning derives views from CAD geometry')
    backlight_gap_mm: confloat(gt=0, le=200, allow_inf_nan=False) = 20.0
    include_ring_light_trials: bool = False
    forbidden_zones: list[ForbiddenZone] = Field(default_factory=list,max_length=100)
    lighting_evidence: str = Field(default="", max_length=3000)


class StationInput(BaseModel):
    roi_mode: Literal['maximum','fixed'] = 'maximum'
    roi_px: list[int] = Field(default_factory=lambda:[2400,2000],min_length=2,max_length=2)
    max_sampling_um_px: confloat(gt=0,le=100,allow_inf_nan=False) = 10.0
    minimum_clearance_mm: confloat(ge=0,le=100,allow_inf_nan=False) = 2.0
    forbidden_zones: list[ForbiddenZone] = Field(default_factory=list,max_length=100)
    lighting_evidence: str = Field(default='',max_length=3000)
    target_ct_s: confloat(gt=0,le=86400,allow_inf_nan=False) | None = None
    max_stations: int | None = Field(default=None,ge=1,le=32)
    station_overhead_s: confloat(ge=0,le=86400,allow_inf_nan=False) | None = None
    setup_cycle_s: dict[str,confloat(ge=0,le=86400,allow_inf_nan=False)] = Field(default_factory=dict,max_length=256)


class ProjectInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    drawing_no: str = Field(default='', max_length=100)
    revision: str = Field(default='', max_length=50)


class BindingInput(BaseModel):
    face_ids: list[str] = Field(min_length=1, max_length=100)
    note: str = Field(default='', max_length=2000)


class AskInput(BaseModel):
    question: str = Field(default='请出具遵循原测量方法的 AOI 初步方案，说明缺口与验证步骤。', min_length=1, max_length=3000)


Coordinate = confloat(strict=True, allow_inf_nan=False)


class AnnotationAnchor(BaseModel):
    face_id: str = Field(min_length=1, max_length=200)
    position: list[Coordinate] = Field(min_length=3, max_length=3)
    label_offset: list[Coordinate] = Field(default_factory=lambda:[24,-24], min_length=2, max_length=2)


class AnnotationInput(BaseModel):
    anchors: list[AnnotationAnchor] = Field(min_length=1, max_length=MAX_ANNOTATION_POINTS)
    candidate_face_ids: list[str] = Field(default_factory=list, max_length=100)
    note: str = Field(default='', max_length=2000)
    status: Literal['pending_review','auto_marked','reviewed'] = 'pending_review'
    cad_sha256: str = Field(min_length=1, max_length=64)
    document_id: str = Field(min_length=1, max_length=64)
    origin: Literal['manual','msop_visual_candidate','geometry_candidate','local_model_candidate'] = 'manual'
    confidence: Literal['low','medium','high'] | None = None


class GuardedRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        async def guarded(request):
            try:
                return await handler(request)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(408, 'CAD 处理超时') from exc
        return guarded


def attach_viewer(app, state):
    if not hasattr(state, 'viewer_model'):
        state.viewer_model = getattr(state, 'model', None)
    store = ViewerStore(state.cfg.state_dir / 'viewer')
    store.recover_tasks()
    store.bootstrap(state.cfg.training_output_dir.parent / '3suite')
    state.viewer = store
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='oracle-viewer')
    task_queue = ViewerTaskQueue(executor, store)
    pool = task_queue
    app.state.viewer_pool = pool
    app.state.viewer_task_queue = task_queue
    marking_jobs = {}
    marking_jobs_lock = RLock()
    static = Path(__file__).parent / 'static'
    app.mount('/viewer-static', StaticFiles(directory=static), name='viewer-static')
    router = APIRouter(prefix='/api/viewer', route_class=GuardedRoute)

    @router.get('/task-queue')
    async def viewer_task_queue():
        return task_queue.snapshot()

    @router.get('/cloud-training')
    async def cloud_training_status():
        receipt = state.cfg.state_dir / 'cloud-training-status.json'
        if not receipt.is_file():
            return {'pipeline': {'status': 'not_started'}, 'progress': {}, 'dataset': {}}
        return json.loads(receipt.read_text())

    @router.post('/task-queue/{task_id}/{action}')
    async def control_viewer_task(task_id: str, action: Literal['pause','resume','delete']):
        try: return task_queue.control(task_id, action)
        except KeyError as exc: raise HTTPException(404, str(exc)) from exc
        except ValueError as exc: raise HTTPException(409, str(exc)) from exc

    @app.get('/viewer', response_class=HTMLResponse)
    async def index():
        html = (static / 'index.html').read_text(encoding='utf-8')
        for name in ('viewer.css', 'viewer.js', 'workflow.js', 'engineering.js', 'learning.js'):
            version = hashlib.sha256((static / name).read_bytes()).hexdigest()[:12]
            html = html.replace(f'/viewer-static/{name}', f'/viewer-static/{name}?v={version}')
        return HTMLResponse(html, headers={'Cache-Control': 'no-store'})

    @router.get('/projects')
    async def projects():
        return store.listing()

    @router.post('/projects')
    async def create(data: ProjectInput):
        return store.summary(store.create(data.title, data.drawing_no, data.revision))

    @router.get('/projects/{project_id}')
    async def project(project_id: str):
        return store.public_project(project_id)

    @router.get('/projects/{project_id}/fai-strategies')
    async def fai_strategies(project_id: str):
        from .fai_strategy import strategy_state
        return await run_in_threadpool(strategy_state,store,project_id)

    @router.get('/projects/{project_id}/automatic-learning')
    async def automatic_learning_status(project_id: str):
        data=store.get(project_id);root=store._dir(project_id)/'automatic-learning'
        result=json.loads((root/'state.json').read_text()) if (root/'state.json').is_file() else {'status':'not_started'}
        if (root/'registration-progress.json').is_file():
            result['registration_progress']=json.loads((root/'registration-progress.json').read_text())
        if (root/'registration.json').is_file():
            receipt=json.loads((root/'registration.json').read_text());msop=store.active_msop(data)
            result['source_stale']=receipt.get('cad_sha256')!=(data.get('model') or {}).get('source_sha256') or not msop or receipt.get('msop_sha256')!=msop.get('sha256')
        if result.get('source_adapter_candidate'):
            candidate=Path(result['source_adapter_candidate']).resolve()
            allowed=Path(__file__).resolve().parents[1]/'talker/state/viewer/source-head-training'
            if candidate.is_relative_to(allowed.resolve()) and (candidate/'capture-progress.json').is_file():
                result['native_training_progress']=json.loads((candidate/'capture-progress.json').read_text())
        pilot=Path(__file__).resolve().parents[1]/'reports/engineering-checks/20261012/native-head-pilot'
        if (pilot/'training-report.json').is_file():
            report=json.loads((pilot/'training-report.json').read_text())
            report['scope']='source_xyz_output_head_pilot'
            report['is_project_geometry_training']=False
            result['native_head_pilot']=report
        expanded=pilot.parent/'native-head-full'
        if (expanded/'training-report.json').is_file():
            report=json.loads((expanded/'training-report.json').read_text())
            result['native_head_expanded']={'stage':'rejected_after_validation' if report.get('promotion_rejected') else 'trained_not_deployed','report':report}
        elif (expanded/'capture-manifest.json').is_file():
            rows=json.loads((expanded/'capture-manifest.json').read_text())
            result['native_head_expanded']={'stage':'feature_capture','samples_done':len(rows),'train_samples_done':sum(r['split']=='train' for r in rows),'test_samples_done':sum(r['split']=='test' for r in rows),'base_weights_changed':False}
        from .native_head_training import refinement_status
        refinement=refinement_status(Path(__file__).resolve().parents[1])
        if refinement:result['native_head_refinement']=refinement
        cloud_receipt=state.cfg.state_dir/'cloud-training-status.json'
        if getattr(state.viewer_model,'openai_backend_label',None)=='vllm-self-hosted' and cloud_receipt.is_file():
            result['cloud_training']=json.loads(cloud_receipt.read_text())
        return result

    @router.post('/projects/{project_id}/automatic-learning')
    async def automatic_learning_start(project_id: str):
        data=store.get(project_id)
        if not store.active_msop(data) or (data.get('model') or {}).get('status') not in ('ready','preview'):
            raise HTTPException(400,'自动配准需要已导入的MSOP和可读取的CAD模型')
        root=store._dir(project_id)/'automatic-learning';root.mkdir(exist_ok=True)
        current=json.loads((root/'state.json').read_text()) if (root/'state.json').is_file() else {}
        if current.get('status') in ('queued','running'):return current
        from .store import now
        from .automatic_learning import write_json
        queued=dict(status='queued',updated_at=now(),weight_training_started=False,weights_changed=False)
        write_json(root/'state.json',queued)
        def work():
            write_json(root/'state.json',dict(queued,status='running'))
            try:
                with (root/'job.log').open('w') as log:
                    subprocess.run([sys.executable,'scripts/run_automatic_learning.py','--project',project_id,
                        '--resume','--output',str(root.resolve())],cwd=Path(__file__).resolve().parent.parent,
                        stdout=log,stderr=subprocess.STDOUT,timeout=3600,check=True)
            except Exception as exc:
                write_json(root/'state.json',dict(queued,status='failed',error=f'{type(exc).__name__}: {exc}'))
        pool.submit(work)
        return queued

    @router.get('/projects/{project_id}/engineering-checks')
    async def engineering_checks(project_id: str, format: Literal['json','csv']='json'):
        from .engineering import build_report, csv_report
        from .workflow import compile_tasks, coordinate_tasks, library
        def check():
            data = store.get(project_id)
            geometry = store.mesh(project_id) if (data.get('model') or {}).get('status') in ('ready','preview') else None
            tasks = compile_tasks(store,data)
            if geometry: tasks = coordinate_tasks(store,data,tasks)
            return build_report(store,data,tasks,geometry,library(),workflow.get(project_id)['run'])
        report = await run_in_threadpool(check)
        if format == 'csv':
            return Response(csv_report(report),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="fai-engineering-checks.csv"'})
        return report

    @router.get('/projects/{project_id}/backup')
    async def project_backup(project_id: str):
        from .backup import archive_project
        import uuid
        target=store.root/'backups'/f'{project_id}-{uuid.uuid4().hex}.zip'
        path=await run_in_threadpool(archive_project,store,project_id,target)
        from starlette.background import BackgroundTask
        return FileResponse(path,filename=f'oracle-{project_id}.zip',background=BackgroundTask(path.unlink,missing_ok=True))

    @router.post('/projects/restore')
    async def project_restore(file: UploadFile = File(...)):
        from .backup import restore_project
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'project.zip'
            total=0
            with path.open('wb') as out:
                while chunk:=await file.read(1024*1024):
                    total+=len(chunk)
                    if total>2*1024**3:
                        raise HTTPException(413,'项目归档超过2GiB')
                    out.write(chunk)
            return await run_in_threadpool(restore_project,store,path)

    @router.get('/projects/{project_id}/fai/{number}')
    async def detail(project_id: str, number: int):
        return store.detail(project_id, number)

    @router.post('/projects/{project_id}/documents')
    async def upload(project_id: str, role: str = Form(...), file: UploadFile = File(...)):
        store.get(project_id)
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in ('.pdf','.step','.stp','.x_t','.x_b'):
            raise HTTPException(400, '支持 PDF、STEP/STP、X_T/X_B')
        with tempfile.TemporaryDirectory(prefix='oracle-viewer-') as tmp:
            path = Path(tmp) / ('source' + suffix)
            size = 0
            try:
                with path.open('wb') as out:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > 128 * 1024 * 1024:
                            raise HTTPException(413, 'Viewer 单文件上限 128 MiB')
                        out.write(chunk)
                try:
                    record = await run_in_threadpool(store.add_document, project_id, role, path, file.filename or path.name)
                except RuntimeError as exc:
                    raise HTTPException(400, '资料读取失败；请核对文件格式和完整性。') from exc
                return {k: v for k,v in record.items() if k not in ('path','method_pages')}
            finally:
                await file.close()

    @router.get('/projects/{project_id}/mesh')
    async def mesh(project_id: str):
        return await run_in_threadpool(store.mesh, project_id)

    @router.post('/projects/{project_id}/cad/rebuild')
    async def rebuild(project_id: str):
        model = await run_in_threadpool(store.rebuild_model, project_id)
        return {k: v for k, v in model.items() if k != 'mesh_file'}

    @router.get('/projects/{project_id}/documents/{document_id}/source')
    async def source(project_id: str, document_id: str):
        document = store.document(project_id, document_id)
        if not document:
            raise HTTPException(404, 'Document not found')
        return FileResponse(document['path'], filename=document['filename'])

    @router.get('/projects/{project_id}/reference-files/{file_id}/source')
    async def reference_source(project_id: str, file_id: str):
        record = store.reference_file(project_id, file_id)
        if not record:
            raise HTTPException(404, 'Reference file not found')
        project_root = store._dir(project_id).resolve()
        path = Path(record.get('path', '')).resolve()
        if not path.is_relative_to(project_root) or not path.is_file():
            raise HTTPException(404, 'Reference file is unavailable')
        return FileResponse(path, filename=record['filename'])

    @router.get('/projects/{project_id}/documents/{document_id}/pages/{page}.png')
    async def page_image(project_id: str, document_id: str, page: int):
        document = store.document(project_id, document_id)
        if not document or document['role'] == 'cad' or not 1 <= page <= document['page_count']:
            raise HTTPException(404, 'PDF page not found')
        def render():
            import pymupdf
            with pymupdf.open(document['path']) as pdf:
                return pdf[page-1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).tobytes('png')
        return Response(await run_in_threadpool(render), media_type='image/png')

    @router.post('/projects/{project_id}/fai/{number}/binding')
    async def bind(project_id: str, number: int, data: BindingInput):
        return store.bind(project_id, number, data.face_ids, data.note)

    @router.delete('/projects/{project_id}/fai/{number}/binding')
    async def unbind(project_id: str, number: int):
        store.unbind(project_id, number)
        return {'ok': True}

    @router.post('/projects/{project_id}/fai/{number}/annotation')
    async def annotate(project_id: str, number: int, data: AnnotationInput):
        return await run_in_threadpool(store.annotate, project_id, number,
            [a.model_dump() for a in data.anchors], data.note, data.status,
            data.cad_sha256, data.document_id, data.origin, data.confidence,
            data.candidate_face_ids)

    def run_marking(project_id, fingerprint):
        store.update_local_model_marking(project_id, status='running', fingerprint=fingerprint, error=None,
                                         items=[],counts=dict(accepted=0,abstain=0,skipped=0,processed=0))
        try:
            from .local_matching import mark_geometry_candidates
            from .marking_pipeline import prepare_marking_candidates
            store.update_local_model_marking(project_id,progress=dict(done=0,message='按 MSOP XYZ 与截面约束检索点位候选'))
            _,generated=prepare_marking_candidates(store,project_id,state.viewer_model,
                progress=lambda value:store.update_local_model_marking(project_id,progress=value))
            existing=sum(bool(i.get('annotation')) and (i['annotation'].get('origin') in ('geometry_candidate','msop_visual_candidate')
                         or i['annotation'].get('origin')=='local_model_candidate' and i['annotation'].get('status')=='pending_review')
                         for i in store.items(store.get(project_id)))
            store.update_local_model_marking(project_id,total=len(generated['candidates'])+existing,
                                             scope_total=len(store.items(store.get(project_id))),
                                             candidate_generation=generated['items'])
            result = mark_geometry_candidates(store, state.viewer_model, state.cfg.fai_matching_model_root,
                                              project_id,generated_candidates=generated['candidates'])
            coverage = marker_coverage(store.items(store.get(project_id)))
            reviewed_items={item['fai']:item for item in result.get('items',[])}
            for diagnostic in unmarked_diagnostics(store,store.get(project_id)):
                if diagnostic['fai'] in reviewed_items:
                    reviewed_items[diagnostic['fai']]['unmarked_diagnostic']=diagnostic
                else:
                    reviewed_items[diagnostic['fai']]=diagnostic
            store.update_local_model_marking(project_id, status='partial' if coverage['unmarked'] else 'completed', review_input_fingerprint=fingerprint,
                                             fingerprint=candidate_fingerprint(project_id,state.viewer_model),
                                             adapter=result['adapter'], counts=result['counts'],
                                             model=result.get('model'),
                                             inference_backend=result.get('inference_backend'),
                                             items=[reviewed_items[n] for n in sorted(reviewed_items)],
                                             coverage=coverage, error=None,
                                             candidate_generation=generated['items'],
                                             progress=dict(done=result['counts']['processed'],current_fai=None,
                                                           message='本地模型核对结束；覆盖数量按实际标记统计'),
                                             message=f"已标记 {coverage['marked']}/{coverage['total']} 项；未标记 {coverage['unmarked']} 项")
            # Coverage and pending candidates change during a successful run.
            # Cache the completed state as well as the original input receipt
            # so reopening a partial project does not repeat identical work.
            store.update_local_model_marking(project_id,
                fingerprint=candidate_fingerprint(project_id,state.viewer_model))
        except Exception as exc:
            store.update_local_model_marking(project_id, status='failed', fingerprint=fingerprint,
                                             error=(f'{type(exc).__name__}: {exc}\n'
                                                    f'{traceback.format_exc()[-3000:]}')[:4000])

    # Bind retry identity to the implementation loaded by this process.
    # Editing development files must not make an old running process repeatedly
    # retry identical failures as though it had loaded the new implementation.
    from . import local_matching as _loaded_matching_engine
    from . import fai_strategy as _loaded_strategy_engine
    from . import part_identity as _loaded_part_engine
    from . import point_names as _loaded_point_name_engine
    from . import scope_review as _loaded_scope_review_engine
    loaded_candidate_engine=hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in
        ('automatic_candidates.py','candidate_mining.py','msop_coordinates.py','msop_tables.py','point_context.py','feature_integrity.py','section_geometry.py',
         'surface_index.py','annotations.py','registration_pipeline.py','cad_view_atlas.py','feature_view_candidates.py',
         'spacing_candidates.py','fai_strategy.py','part_identity.py','landmark_transfer.py','point_names.py','scope_review.py'))).hexdigest()
    loaded_review_engine=hashlib.sha256(Path(_loaded_matching_engine.__file__).read_bytes()).hexdigest()

    def candidate_fingerprint(project_id, model):
        data = store.get(project_id)
        candidates = [(item['number'], item['annotation'].get('origin'), item['annotation'].get('status'), item['annotation'].get('updated_at'), item['annotation'].get('face_ids'))
                      for item in store.items(data)
                      if item.get('annotation') and (item['annotation'].get('origin') in ('geometry_candidate', 'msop_visual_candidate')
                         or item['annotation'].get('origin')=='local_model_candidate' and item['annotation'].get('status')=='pending_review')
                      ]
        active_msop = store.active_msop(data)
        registration=store._dir(project_id)/'automatic-learning'/'registration.json'
        payload = {'msop': active_msop.get('sha256') if active_msop else None,
                   'cad': (data.get('model') or {}).get('source_sha256'), 'candidates': candidates,
                   # Bump when candidate search/acceptance behavior changes so a
                   # completed run cannot suppress retries under a newer engine.
                   'automation_ruleset_revision': 53,
                   'coverage': marker_coverage(store.items(data)),
                   'registration_receipt':hashlib.sha256(registration.read_bytes()).hexdigest() if registration.is_file() else None,
                   'registration_alternatives':[(p.name,hashlib.sha256(p.read_bytes()).hexdigest()) for p in
                       sorted(registration.parent.joinpath('registration-alternatives').glob('*.json'))],
                   'candidate_engine':loaded_candidate_engine,
                   'review_engine':loaded_review_engine,
                   'inference_model': {'backend': model.info().get('inference_backend'),
                                       'model': model.info().get('model_alias')}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def queue_marking(project_id,fingerprint,**updates):
        with marking_jobs_lock:
            active=marking_jobs.get(project_id)
            if active is not None and not active.done():
                return store.get(project_id).get('local_model_marking') or {}
            from .fai_strategy import archive_marking_state
            archive_marking_state(store,project_id,store.get(project_id).get('local_model_marking') or {})
            updates={'items':[], 'counts':dict(accepted=0,abstain=0,skipped=0,processed=0),
                'message':'已排队，等待本地模型执行',
                'progress':dict(done=0,current_fai=None,message='等待本地模型执行'),**updates}
            queued=store.update_local_model_marking(project_id,status='queued',fingerprint=fingerprint,
                                                     error=None,**updates)
            marking_jobs[project_id]=pool.submit(run_marking,project_id,fingerprint)
            return queued

    @router.post('/projects/{project_id}/local-model-marking')
    async def start_local_model_marking(project_id: str, retry: bool = False):
        if state.cfg.fai_matching_model_root is None:
            raise HTTPException(503, '尚未配置本地 FAI 匹配适配器')
        if state.viewer_model_error or not state.viewer_model.loaded:
            raise HTTPException(503, state.viewer_model_error or 'Viewer 模型尚未就绪')
        store.get(project_id)
        with marking_jobs_lock:
            active=marking_jobs.get(project_id)
            if active is not None and not active.done():
                return store.get(project_id).get('local_model_marking') or {}
        store.require_current_feature_integrity(project_id)
        fingerprint = candidate_fingerprint(project_id, state.viewer_model)
        current = (store.get(project_id).get('local_model_marking') or {})
        if not retry and current.get('fingerprint') == fingerprint and current.get('status') in ('queued','running','completed','partial','blocked'):
            return current
        candidates = [item for item in store.items(store.get(project_id))
                      if item.get('annotation') and (item['annotation'].get('origin') in ('geometry_candidate', 'msop_visual_candidate')
                         or item['annotation'].get('origin')=='local_model_candidate' and item['annotation'].get('status')=='pending_review')
                         ]
        if not candidates:
            coverage = marker_coverage(store.items(store.get(project_id)))
            data=store.get(project_id)
            active=store.active_msop(data) or {}
            if coverage['unmarked'] and automatic_marking_sources_ready(data,active):
                return queue_marking(project_id,fingerprint,
                    total=coverage['unmarked'],coverage=coverage,
                    progress=dict(done=0,stage='registration',message='整理配准假设，交由本地模型选择并核验'))
            registration_path=store._dir(project_id)/'automatic-learning'/'registration.json'
            if coverage['unmarked'] and registration_path.is_file():
                receipt=json.loads(registration_path.read_text())
                active=store.active_msop(store.get(project_id)) or {}
                cad=(store.get(project_id).get('model') or {}).get('source_sha256')
                if (receipt.get('cad_sha256')==cad and receipt.get('msop_sha256')==active.get('sha256')
                        and any(f.get('triangle_refinement',{}).get('best') for f in receipt.get('frames',[]))):
                    return queue_marking(project_id,fingerprint,total=coverage['unmarked'],coverage=coverage)
            diagnostics = await run_in_threadpool(unmarked_diagnostics, store, store.get(project_id))
            return store.update_local_model_marking(project_id, status='blocked' if coverage['unmarked'] else 'completed', fingerprint=fingerprint,
                                                     counts={'accepted': 0, 'abstain': 0, 'skipped': coverage['unmarked'], 'processed': 0},
                                                     coverage=coverage, items=diagnostics, total=coverage['total'],
                                                     error=None, message=(f"已标记 {coverage['marked']}/{coverage['total']} 项；"
                                                       f"未标记 {coverage['unmarked']} 项，尚无可供本地模型核对的几何候选"
                                                       if coverage['unmarked'] else '所有 FAI 均有位置标记；没有待核对的新候选'))
        return queue_marking(project_id,fingerprint,total=len(candidates))

    @router.get('/projects/{project_id}/local-model-marking')
    async def local_model_marking_status(project_id: str):
        data = store.get(project_id)
        result = dict(data.get('local_model_marking') or {'status': 'not_started'})
        result['coverage'] = marker_coverage(store.items(data))
        if result['status'] == 'completed' and result['coverage']['unmarked']:
            result.update(status='partial', message=f"仍有 {result['coverage']['unmarked']} 项 FAI 未标记")
        return result

    @router.delete('/projects/{project_id}/fai/{number}/annotation')
    async def remove_annotation(project_id: str, number: int):
        await run_in_threadpool(store.remove_annotation, project_id, number)
        return {'ok': True}

    def work(project_id, task_id):
        try:
            task = store.update_task(project_id, task_id, status='running')
            question = f'针对 FAI {task["fai"]}，' + task['question']
            result = answer_project(state.viewer_model, question, task['input_context'], task['binding'])
            store.update_task(project_id, task_id, status='complete', result=result)
        except Exception as exc:
            store.update_task(project_id, task_id, status='failed', error=str(exc)[:1000])

    @router.post('/projects/{project_id}/fai/{number}/ask')
    async def ask(project_id: str, number: int, data: AskInput):
        project = store.get(project_id)
        if any(t['status'] in ('queued','running') for t in project['tasks']):
            raise HTTPException(409, '本项目已有方案任务正在运行')
        task = store.task(project_id, number, data.question)
        pool.submit(work, project_id, task['id'])
        return {'id': task['id'], 'status': task['status']}

    @router.get('/projects/{project_id}/tasks/{task_id}')
    async def task_status(project_id: str, task_id: str):
        task = next((t for t in store.public_project(project_id)['tasks'] if t['id'] == task_id), None)
        if not task:
            raise HTTPException(404, 'Task not found')
        return task

    @router.get('/projects/{project_id}/export')
    async def export(project_id: str):
        data = store.public_project(project_id)
        return Response(json.dumps(data, ensure_ascii=False, indent=2), media_type='application/json',
                        headers={'Content-Disposition': f'attachment; filename="oracle-project-{project_id}.json"'})

    from .workflow import WorkflowService, library, CATALOG, bom_csv
    workflow = WorkflowService(store, state.viewer_model, pool)
    # A server interruption must not leave a permanently locked planning run.
    for project_dir in store.root.iterdir():
        path = project_dir / 'aoi-workflow.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record.get('status') in ('queued', 'running'):
                record.update(status='failed', error='Server interrupted; rerun the workflow')
                workflow.save(project_dir.name, record)

    from .assembly import AssemblyService
    assembly = AssemblyService(workflow)
    for assembly_file in store.root.glob('*/aoi-assembly-*.json'):
        interrupted = json.loads(assembly_file.read_text())
        if interrupted.get('status') in ('queued', 'running'):
            interrupted.update(status='failed', error='Server interrupted; regenerate assembly')
            assembly.save(assembly_file.parent.name, interrupted['scheme_id'], interrupted)

    @router.get('/projects/{project_id}/workflow/assembly/{scheme_id}')
    async def assembly_status(project_id: str, scheme_id: Literal['shared','throughput','domestic'], format: Literal['json','csv']='json'):
        record = await run_in_threadpool(assembly.get, project_id, scheme_id)
        if format == 'json':
            return record
        if not record or not record.get('result'):
            raise HTTPException(409, 'Generate an assembly first')
        import csv, io
        output = io.StringIO(); writer = csv.writer(output)
        writer.writerow(['item','quantity','instances','kind','status','required_active_area_mm','source_url','stale'])
        for row in record['result']['bom']:
            writer.writerow([row['id'],row.get('quantity'),';'.join(row.get('instance_ids',[])),row.get('kind'),row.get('status'),' x '.join(map(str,row.get('active_requirement_mm',[]))),row.get('source_url',''),record['stale']])
        return Response('\ufeff'+output.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="aoi-whole-assembly-bom.csv"'})

    @router.post('/projects/{project_id}/workflow/assembly/{scheme_id}')
    async def assembly_run(project_id: str, scheme_id: Literal['shared','throughput','domestic']):
        try:
            return await run_in_threadpool(assembly.submit, project_id, scheme_id)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    from .presentation import PresentationService
    presentations = PresentationService(assembly)
    for job_file in store.root.glob('*/assembly-presentations/*/job.json'):
        job = json.loads(job_file.read_text())
        if job.get('status') in ('queued','running'):
            job.update(status='failed',error='Server interrupted; export the PPT again')
            presentations.save(job_file.parent.parent.parent.name,job)

    @router.post('/projects/{project_id}/workflow/assembly/{scheme_id}/presentation')
    async def presentation_run(project_id: str, scheme_id: Literal['shared','throughput','domestic'], data: PresentationInput):
        try:
            return await run_in_threadpool(presentations.submit, project_id, scheme_id, data.assembly_id, data.images)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @router.get('/projects/{project_id}/workflow/assembly/{scheme_id}/presentation/{job_id}')
    async def presentation_status(project_id: str, scheme_id: Literal['shared','throughput','domestic'], job_id: str):
        try:
            job = await run_in_threadpool(presentations.get, project_id, job_id)
        except ValueError as e:
            raise HTTPException(404, str(e)) from e
        if job['scheme_id'] != scheme_id:
            raise HTTPException(404, 'Presentation does not belong to this scheme')
        return job

    @router.get('/projects/{project_id}/workflow/assembly/{scheme_id}/presentation/{job_id}/download')
    async def presentation_download(project_id: str, scheme_id: Literal['shared','throughput','domestic'], job_id: str):
        job = await presentation_status(project_id, scheme_id, job_id)
        if job['status'] != 'complete':
            raise HTTPException(409, 'Presentation not complete')
        return FileResponse(presentations.path(project_id, job_id)/'output/assembly.pptx',filename=f'oracle-{scheme_id}-actions-paths.pptx',media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation')

    @router.get('/library')
    async def hardware_library():
        return library()

    @router.get('/library/assets/{filename}')
    async def library_asset(filename: str):
        if Path(filename).name != filename or Path(filename).suffix.lower() not in ('.pdf', '.html', '.txt', '.step', '.stp', '.igs', '.iges', '.x_t', '.x_b', '.sat', '.dxf', '.dwg', '.zip'):
            raise HTTPException(404, 'Invalid library asset')
        allowed = {a['file'] for i in library()['items'] for a in i['assets']}
        if filename not in allowed:
            raise HTTPException(404, 'Asset not downloaded')
        return FileResponse(CATALOG / 'assets' / filename, filename=filename)

    @router.get('/library/items/{item_id}/preview-mesh')
    async def library_preview_mesh(item_id: str):
        """Return a lightweight display mesh only from a downloaded catalog STEP model."""
        item = next((i for i in library()['items'] if i['id'] == item_id), None)
        if not item:
            raise HTTPException(404, '器件不在本地规格库')
        assets = {a['file'] for a in item.get('assets', [])}
        preferred = item.get('cad_body_model_file')
        candidates = [preferred] if preferred else []
        candidates += [name for name in item.get('cad_files', []) if Path(name).suffix.lower() in ('.step', '.stp')]
        filename = next((name for name in candidates if name in assets and (CATALOG/'assets'/name).is_file()), None)
        if not filename:
            raise HTTPException(404, '规格库没有该器件可用的 STEP/STP 实体模型；Viewer 不会用外形盒代替')
        source = CATALOG / 'assets' / filename
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        cache_root = Path(state.cfg.state_dir) / 'catalog-preview-meshes'
        cache_root.mkdir(parents=True, exist_ok=True)
        mesh = cache_root / f'{digest}-display.mesh.json'
        if not mesh.is_file():
            worker = Path(__file__).with_name('geometry_worker.py')
            process = await run_in_threadpool(subprocess.run,
                ['/usr/bin/python3', str(worker), str(source), str(mesh), '0.45'],
                capture_output=True, text=True, timeout=180)
            if process.returncode or not mesh.is_file():
                raise HTTPException(422, 'STEP 网格生成失败：'+(process.stderr or process.stdout or '未知错误')[-500:])
        return FileResponse(mesh, filename=f'{item_id}-catalog-preview.mesh.json', media_type='application/json')

    @router.get('/projects/{project_id}/workflow')
    async def workflow_status(project_id: str):
        return await run_in_threadpool(workflow.get, project_id)

    from .stations import StationService
    stations=StationService(store,state.viewer_model,pool)
    for station_file in store.root.glob('*/station-workflow.json'):
        interrupted=json.loads(station_file.read_text())
        if interrupted.get('status') in ('queued','running'):
            interrupted.update(status='failed',error='服务中断，请重新提交工站规划')
            stations.save(station_file.parent.name,interrupted)

    from .station_presentation import StationPresentationService
    station_presentations=StationPresentationService(stations)
    for job_file in store.root.glob('*/station-presentations/*/job.json'):
        interrupted=json.loads(job_file.read_text())
        if interrupted.get('status') in ('queued','running'):
            interrupted.update(status='failed',error='服务中断，请重新导出完整方案PPT')
            station_presentations.save(job_file.parent.parent.parent.name,interrupted)

    @router.post('/projects/{project_id}/workflow/stations/presentation')
    async def station_ppt_run(project_id: str,data: StationPresentationInput):
        try:
            return await run_in_threadpool(station_presentations.submit,project_id,data.plan_id,data.images)
        except (ValueError,KeyError) as exc:
            raise HTTPException(409,str(exc))

    @router.get('/projects/{project_id}/workflow/stations/presentation/current')
    async def station_ppt_latest(project_id: str):
        return await run_in_threadpool(station_presentations.latest,project_id)

    @router.get('/projects/{project_id}/workflow/stations/presentation/{job_id}')
    async def station_ppt_status(project_id: str,job_id: str):
        try:
            return await run_in_threadpool(station_presentations.get,project_id,job_id)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            raise HTTPException(404,str(exc))

    @router.get('/projects/{project_id}/workflow/stations/presentation/{job_id}/download')
    async def station_ppt_download(project_id: str,job_id: str):
        job=await station_ppt_status(project_id,job_id)
        if job['status']!='complete' or job['stale']:
            raise HTTPException(409,'PPT尚未完成或源方案已变化，请重新导出')
        project=store.get(project_id)
        return FileResponse(station_presentations.path(project_id,job_id)/'output/station-workflow.pptx',filename=f'oracle-{project["drawing_no"]}-rev{project["revision"]}-full-workflow.pptx',media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation')

    @router.get('/projects/{project_id}/workflow/stations')
    async def station_status(project_id: str):
        return await run_in_threadpool(stations.get,project_id)

    @router.post('/projects/{project_id}/workflow/stations')
    async def station_run(project_id: str,data: StationInput):
        if state.viewer_model_error or not state.viewer_model.loaded:
            raise HTTPException(503,'本地模型尚未就绪')
        return await run_in_threadpool(stations.submit,project_id,data.model_dump())

    @router.post('/projects/{project_id}/workflow')
    async def workflow_run(project_id: str, data: WorkflowInput):
        design = data.model_dump()
        design['fai_numbers'] = list(dict.fromkeys(design['fai_numbers']))
        return await run_in_threadpool(workflow.submit, project_id, design)

    @router.post('/projects/{project_id}/workflow/cancel')
    async def workflow_cancel(project_id: str):
        return await run_in_threadpool(workflow.cancel,project_id)

    @router.get('/projects/{project_id}/workflow/candidate-audit')
    async def workflow_candidate_audit(project_id: str):
        record=(await run_in_threadpool(workflow.get,project_id))['run'] or {}
        audit=(record.get('result') or {}).get('candidate_audit') or {}
        filename=audit.get('file')
        if filename!=f'aoi-candidates-{audit.get("source_run_id") or record.get("id")}.json.gz':
            raise HTTPException(404,'当前任务尚无完整候选审计记录')
        path=store._dir(project_id)/filename
        if not path.is_file():raise HTTPException(404,'候选审计文件不存在')
        return FileResponse(path,filename=filename,media_type='application/gzip')

    @router.get('/projects/{project_id}/workflow/export')
    async def workflow_export(project_id: str, format: Literal['json','csv']='json', scheme_id: str | None=None):
        record = (await run_in_threadpool(workflow.get, project_id))['run']
        if not record:
            raise HTTPException(404, 'Generate a workflow first')
        if scheme_id and scheme_id not in {s['id'] for s in (record.get('result') or {}).get('integrated_schemes',[])}:
            raise HTTPException(404, 'Unknown whole-project scheme')
        if scheme_id and format == 'json':
            scheme=next(s for s in record['result']['integrated_schemes'] if s['id']==scheme_id)
            return Response(json.dumps(dict(project=record['project'],model_identity=record.get('model_identity'),stale=record.get('stale'),scheme=scheme),ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename="aoi-whole-scheme.json"'})
        if format == 'csv':
            return Response(bom_csv(record,scheme_id), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="aoi-candidate-bom.csv"'})
        return Response(json.dumps(record, ensure_ascii=False, indent=2), media_type='application/json', headers={'Content-Disposition':'attachment; filename="aoi-workflow-draft.json"'})

    @router.post('/projects/{project_id}/workflow/virtual-shot-image/{head_id}')
    async def save_virtual_shot_image(project_id: str, head_id: str, workflow_id: str=Form(...), image: UploadFile=File(...)):
        record=(await run_in_threadpool(workflow.get,project_id))['run'] or {}
        if record.get('id')!=workflow_id or record.get('stale'):
            raise HTTPException(409,'拍摄方案已变化，请重新加载拍摄位置后导出')
        result=record.get('result') or {}
        head=next((h for h in result.get('heads',[])+result.get('alternate_heads',[]) if h.get('id')==head_id),None)
        if not head:raise HTTPException(404,'当前方案没有这个拍摄位置')
        raw=await image.read(32*1024*1024+1)
        if len(raw)>32*1024*1024 or len(raw)<24 or raw[:8]!=b'\x89PNG\r\n\x1a\n' or raw[12:16]!=b'IHDR':
            raise HTTPException(422,'虚拟拍图须为32MB以内的有效PNG')
        size=list(struct.unpack('>II',raw[16:24]));roi=(head.get('optics') or {}).get('roi_px')
        if size!=roi:raise HTTPException(422,'虚拟图像素尺寸与相机镜头计算的ROI不一致')
        digest=hashlib.sha256(raw).hexdigest();root=store._dir(project_id)/'virtual-shots'/workflow_id;root.mkdir(parents=True,exist_ok=True)
        (root/(digest+'.png')).write_bytes(raw)
        metadata=dict(workflow_id=workflow_id,head_id=head_id,camera=head['camera'],lens=head['lens'],roi_px=roi,
                      fov_mm=head['scene']['fov_xy_mm'],status='CAD_geometry_projection_not_real_capture',sha256=digest)
        (root/(digest+'.json')).write_text(json.dumps(metadata,ensure_ascii=False))
        return dict(metadata,download_url=f'/api/viewer/projects/{project_id}/workflow/virtual-shot-image/{digest}?workflow_id={workflow_id}')

    @router.get('/projects/{project_id}/workflow/virtual-shot-image/{image_id}')
    async def download_virtual_shot_image(project_id: str, image_id: str, workflow_id: str):
        if not re.fullmatch(r'[0-9a-f]{64}',image_id) or not re.fullmatch(r'[0-9a-f]{32}',workflow_id):
            raise HTTPException(404,'虚拟图记录不存在')
        path=store._dir(project_id)/'virtual-shots'/workflow_id/(image_id+'.png')
        if not path.is_file():raise HTTPException(404,'虚拟图文件不存在')
        return FileResponse(path,media_type='image/png',filename='oracle-virtual-shot-'+image_id[:12]+'.png')

    @router.get('/projects/{project_id}/workflow/virtual-shots')
    async def virtual_shot_coordinates(project_id: str, format: Literal['json','csv']='json', scheme_id: str | None=None):
        def build():
            record=workflow.get(project_id)['run']
            result=(record or {}).get('result') or {};heads=list(result.get('heads') or [])
            if scheme_id:
                scheme=next((s for s in result.get('integrated_schemes',[]) if s.get('id')==scheme_id),None)
                if not scheme:raise KeyError('Unknown whole-project scheme')
                allowed={h for st in scheme.get('stations',[]) for h in st.get('head_ids',[])}
                heads=[h for h in heads if h.get('id') in allowed]
            data=store.get(project_id)
            from .workflow import coordinate_tasks,compile_tasks
            tasks=coordinate_tasks(store,data,compile_tasks(store,data));by_number={t['number']:t for t in tasks};rows=[]
            if not heads:
                station_record=stations.get(project_id);station_result=(station_record or {}).get('result') or {}
                if not station_result:raise ValueError('尚无完成的拍摄布置或 Station / Setup 方案')
                for station in station_result.get('stations',[]):
                    for setup in station.get('setups',[]):
                        for region in setup.get('measurement_regions',[]):
                            number=region.get('number');task=by_number.get(number);mapping=(task or {}).get('msop_coordinate_map') or {}
                            mapped={p['name']:p for p in mapping.get('mapped_points',[])}
                            correspond={m.get('point_name'):m.get('anchor_index') for m in mapping.get('anchor_correspondence',[])}
                            linked={correspond.get(name):name for name in correspond}
                            anchors=region.get('anchors') or []
                            point_rows=[]
                            for i,anchor in enumerate(anchors):
                                name=linked.get(i);source=next((p for p in mapping.get('source_points',[]) if p.get('name')==name),None) if name else None
                                match=mapped.get(name) if name else None
                                point_rows.append((name or f"FAI{number}_anchor_{i+1}",source,match.get('cad_position_mm') if match else anchor.get('position'),bool(name and match)))
                            used_names={name for name,source,_cad,_registered in point_rows if source}
                            for source in mapping.get('source_points',[]):
                                if source.get('name') in used_names:continue
                                match=mapped.get(source.get('name'))
                                point_rows.append((source.get('name'),source,match.get('cad_position_mm') if match else None,bool(match)))
                            if not point_rows:
                                point_rows=[(p.get('name'),p,p.get('cad_position_mm'),True) for p in mapping.get('mapped_points',[])]
                            for name,source,cad,registered in point_rows:
                                rows.append(dict(shot_id=f"{station.get('id')}/{setup.get('id')} · setup region",fai=number,point_name=name,
                                    coordinate_status='registered_to_display_CAD' if registered else 'source_only_unregistered',
                                    setup_channel=setup.get('channel'),device_ids=setup.get('device_ids',[]),
                                    source_xyz_mm=(source or {}).get('xyz'),source_coordinate_system=(source or {}).get('coordinate_system'),
                                    cad_world_xyz_mm=cad,image_u_px=None,image_v_px=None,image_width_px=None,image_height_px=None,
                                    in_image=None,view_fov_mm=None,capture_image_status='no_camera_exposure_pose_or_intrinsics',
                                    coordinate_authority='CAD display coordinate; no sensor image projection was defined for this Station / Setup' if cad else 'MSOP source coordinate only; no CAD mapping'))
                if not rows:raise ValueError('Station / Setup 方案没有可导出的测量点坐标')
                project=store.get(project_id)
                return dict(project=dict(drawing_no=project.get('drawing_no'),revision=project.get('revision')),
                    workflow_id=(station_record or {}).get('id'),scheme_id=None,coordinate_convention=dict(image='Station / Setup 输出尚无相机曝光位姿、内参或标定，因此图像像素坐标留空，不以屏幕预览坐标冒充拍图坐标',CAD='CAD显示坐标，不代表已验收计量坐标系',MSOP='MSOP XYZ保留原坐标系；仅存在受支持映射时才填写CAD位置'),rows=rows)
            for head in heads:
                scene=head.get('scene') or {};fov=scene.get('fov_xy_mm') or [None,None]
                width,height=(head.get('optics') or {}).get('roi_px') or (record.get('design') or {}).get('roi_px') or [None,None]
                bx,by,target=scene.get('basis_x'),scene.get('basis_y'),scene.get('target_mm')
                for number in head.get('fai_numbers',[]):
                    task=by_number.get(number)
                    if not task:continue
                    mapping=task.get('msop_coordinate_map') or {};mapped={p['name']:p for p in mapping.get('mapped_points',[])}
                    correspond={m.get('point_name'):m.get('anchor_index') for m in mapping.get('anchor_correspondence',[])}
                    points=[]
                    for source in mapping.get('source_points',[]):
                        p=mapped.get(source.get('name'));points.append(dict(name=source.get('name'),source=source.get('xyz'),frame=source.get('coordinate_system'),cad=p.get('cad_position_mm') if p else None,status='registered_to_display_CAD' if p else 'source_only_unregistered',anchor=correspond.get(source.get('name'))))
                    if not points:
                        points=[dict(name=f"FAI{number}_anchor_{i+1}",source=None,frame=None,cad=a.get('position'),status='CAD_marker_only_no_MSOP_point_link',anchor=i) for i,a in enumerate(task.get('anchors',[]))]
                    for p in points:
                        u=v=None;inside=False;xyz=p.get('cad')
                        if xyz and target and bx and by and all(fov) and all((width,height)):
                            delta=[float(x)-float(t) for x,t in zip(xyz,target)];x=sum(delta[i]*float(bx[i]) for i in range(3));y=sum(delta[i]*float(by[i]) for i in range(3))
                            u=(x/float(fov[0])+.5)*float(width);v=(.5-y/float(fov[1]))*float(height);inside=0<=u<float(width) and 0<=v<float(height)
                        rows.append(dict(shot_id=head.get('id'),fai=number,point_name=p['name'],coordinate_status=p['status'],camera=head.get('camera'),lens=head.get('lens'),light=head.get('light'),lighting_strategy=head.get('lighting_strategy'),source_xyz_mm=p.get('source'),source_coordinate_system=p.get('frame'),cad_world_xyz_mm=xyz,image_u_px=round(u,3) if u is not None else None,image_v_px=round(v,3) if v is not None else None,image_width_px=width,image_height_px=height,in_image=inside,view_fov_mm=fov,coordinate_authority='CAD_display_mm; display-geometry registration only' if xyz and p.get('source') else 'unregistered; image/world coordinates unavailable'))
            return dict(project=(record or {}).get('project'),workflow_id=(record or {}).get('id'),scheme_id=scheme_id,coordinate_convention=dict(image_origin='top-left, zero-based continuous pixel coordinates',CAD='cad_world_xyz_mm is viewer display CAD frame, not metrology-certified unless source geometry is validated',MSOP='source_xyz_mm remains in its source coordinate system and is never silently transformed',projection='orthographic CAD virtual view using saved camera basis and FOV; not a real sensor capture'),rows=rows)
        try:report=await run_in_threadpool(build)
        except ValueError as exc:raise HTTPException(409,str(exc))
        except KeyError as exc:raise HTTPException(404,str(exc))
        if format=='json':return Response(json.dumps(report,ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename="oracle-virtual-shot-coordinates.json"'})
        output=io.StringIO(newline='');fields=['shot_id','fai','point_name','camera','lens','light','lighting_strategy','setup_channel','device_ids','coordinate_status','capture_image_status','source_coordinate_system','source_x_mm','source_y_mm','source_z_mm','cad_x_mm','cad_y_mm','cad_z_mm','image_u_px','image_v_px','image_width_px','image_height_px','in_image','coordinate_authority'];writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
        for row in report['rows']:
            src=row.get('source_xyz_mm') or [];cad=row.get('cad_world_xyz_mm') or []
            writer.writerow(dict(shot_id=row['shot_id'],fai=row['fai'],point_name=row['point_name'],camera=row.get('camera'),lens=row.get('lens'),light=row.get('light'),lighting_strategy=row.get('lighting_strategy'),setup_channel=row.get('setup_channel'),device_ids=';'.join(row.get('device_ids',[])),coordinate_status=row['coordinate_status'],capture_image_status=row.get('capture_image_status','virtual_projection_available' if row.get('image_u_px') is not None else 'no_valid_projection'),source_coordinate_system=row.get('source_coordinate_system'),source_x_mm=src[0] if len(src)>0 else '',source_y_mm=src[1] if len(src)>1 else '',source_z_mm=src[2] if len(src)>2 else '',cad_x_mm=cad[0] if len(cad)>0 else '',cad_y_mm=cad[1] if len(cad)>1 else '',cad_z_mm=cad[2] if len(cad)>2 else '',image_u_px=row.get('image_u_px'),image_v_px=row.get('image_v_px'),image_width_px=row.get('image_width_px'),image_height_px=row.get('image_height_px'),in_image=row.get('in_image'),coordinate_authority=row.get('coordinate_authority')))
        return Response('\ufeff'+output.getvalue(),media_type='text/csv; charset=utf-8',headers={'Content-Disposition':'attachment; filename="oracle-virtual-shot-coordinates.csv"'})

    app.include_router(router)
