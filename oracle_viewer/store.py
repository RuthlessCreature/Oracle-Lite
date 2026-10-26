from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from . import msop

MSOP_PARSER_VERSION = 2
from .annotations import validate_anchors


def now():
    return datetime.now(timezone.utc).isoformat()


def annotation_input(annotation):
    if annotation is None:
        return None
    return {k: annotation.get(k) for k in ('status', 'note', 'origin', 'confidence',
                                           'registration_sha256')} | {
        'anchors': [{k: a.get(k) for k in ('face_id', 'position')} for a in annotation['anchors']]}


class ViewerStore:
    """Atomic, local project records shared by both interfaces (one server process)."""
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def _dir(self, project_id):
        if not isinstance(project_id, str) or not project_id.isalnum():
            raise ValueError('Invalid project ID')
        return self.root / project_id

    def _write(self, data):
        directory = self._dir(data['id'])
        directory.mkdir(exist_ok=True)
        temp = directory / 'project.json.tmp'
        temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temp.replace(directory / 'project.json')

    def get(self, project_id):
        path = self._dir(project_id) / 'project.json'
        if not path.exists():
            raise KeyError('Project not found')
        return json.loads(path.read_text(encoding='utf-8'))

    def create(self, title, drawing_no='', revision=''):
        with self.lock:
            data = {'id': uuid.uuid4().hex, 'title': title, 'drawing_no': drawing_no,
                    'revision': revision, 'created_at': now(), 'documents': [],
                    'model': None, 'bindings': {}, 'annotations': {}, 'tasks': [], 'events': [],
                    'local_model_marking': None}
            self._write(data)
            return data

    def listing(self):
        return [self.summary(self.get(p.parent.name)) for p in sorted(self.root.glob('*/project.json'))]

    def summary(self, data):
        items = self.items(data)
        return {k: data[k] for k in ('id', 'title', 'drawing_no', 'revision', 'created_at')} | {
            'fai_count': len(items), 'located_count': sum(bool(i['binding']) for i in items),
            'marked_count': sum(bool(i['annotation']
                and i['annotation'].get('origin') == 'local_model_candidate'
                and i['annotation'].get('status') == 'auto_marked'
                and i['annotation'].get('anchors')) for i in items),
            'reviewed_count': sum(i['annotation'] is not None and i['annotation']['status']=='reviewed' for i in items),
            'geometry_status': (data['model'] or {}).get('status', 'missing')}

    def active_msop(self, data):
        return next((d for d in reversed(data['documents']) if d['role'] == 'msop'), None)

    def items(self, data):
        document = self.active_msop(data)
        if not document:
            return []
        items = msop.inventory(document['method_pages'], document.get('supplemental_fai_data'))
        for item in items:
            item['key'] = f'{document["id"]}:{item["number"]}'
            item['document_id'] = document['id']
            item['binding'] = data['bindings'].get(item['key'])
            item['annotation'] = data.get('annotations', {}).get(item['key'])
        return items

    def set_supplemental_fai_data(self, project_id, document_id, rows, source='local_vision_review'):
        """Store page-image transcriptions separately from native MSOP text extraction."""
        if not isinstance(rows, dict) or len(rows) > 1000:
            raise ValueError('Supplemental FAI data must be a bounded object')
        with self.lock:
            data = self.get(project_id)
            document = next((d for d in data['documents'] if d['id'] == document_id and d['role'] == 'msop'), None)
            if not document or document is not self.active_msop(data):
                raise ValueError('Supplemental data must target the active MSOP document')
            known = {str(item['number']) for item in msop.inventory(document['method_pages'])}
            if set(rows) - known:
                raise ValueError('Supplemental data references an FAI absent from the active MSOP')
            checked = {}
            for number, row in rows.items():
                if not isinstance(row, dict):
                    raise ValueError('Each supplemental FAI row must be an object')
                pages = row.get('source_pages', [])
                if not isinstance(pages, list) or any(not isinstance(v, int) for v in pages):
                    raise ValueError('Supplemental source_pages must be integer page numbers')
                if any(v < 1 or v > document['page_count'] for v in pages):
                    raise ValueError('Supplemental source page is out of range')
                checked[number] = dict(row)
            document['supplemental_fai_data'] = checked
            data['events'].append({'at': now(), 'action': 'supplemental_fai_data',
                                   'document_id': document_id, 'source': source,
                                   'fai_numbers': sorted(map(int, checked))})
            self._write(data)
            return checked

    def detail(self, project_id, number):
        data = self.get(project_id)
        item = next((i for i in self.items(data) if i['number'] == number), None)
        if not item:
            raise KeyError('FAI not found in explicit MSOP headings')
        doc = self.active_msop(data)
        selected = [p | {'measurement_indicators': msop.measurement_indicators(p.get('text', ''))}
                    for p in doc['method_pages'] if number in p['fai_ids']]
        return {'item': item, 'methods': selected,
                'dependencies': msop.dependencies(doc['method_pages'], selected),
                'context': msop.context(doc['method_pages'], selected, doc['filename']),
                'project': self.summary(data)}

    def public_project(self, project_id):
        data = self.get(project_id)
        active = self.active_msop(data)
        for task in data['tasks']:
            task['stale'] = (task['document_id'] != (active or {}).get('id')
                             or task.get('cad_sha256') != (data['model'] or {}).get('source_sha256')
                             or annotation_input(task.get('annotation')) != annotation_input(
                                 data.get('annotations', {}).get(f"{(active or {}).get('id')}:{task['fai']}")))
        return self.summary(data) | {
            'documents': [{k: d[k] for k in ('id','role','filename','sha256','page_count',
                                              'conversion_status','conversion_error') if k in d}
                          for d in data['documents']],
            'reference_files': [{k: f[k] for k in ('id', 'filename', 'category', 'sha256',
                                                    'bytes', 'description', 'metadata') if k in f}
                                for f in data.get('reference_files', [])],
            'source_warnings': data.get('source_warnings', []),
            'measurement_programs': data.get('measurement_programs', []),
            'items': self.items(data),
            'model': ({k: v for k,v in data['model'].items() if k != 'mesh_file'} if data['model'] else None),
            'local_model_marking': data.get('local_model_marking'),
            'tasks': data['tasks'],
            'coverage': '按 MSOP 明确标题收录；范围标题已展开，重复页保留。不代表子测量项全部核对完成。',
            'events': data['events'][-30:]}

    def add_document(self, project_id, role, source: Path, filename: str):
        if role not in ('msop', 'drawing', 'cad'):
            raise ValueError('Unsupported document role')
        suffix = source.suffix.lower()
        if role != 'cad' and suffix != '.pdf':
            raise ValueError('MSOP and drawing inputs must be PDF in this version')
        if role == 'cad' and suffix not in ('.step','.stp','.x_t','.x_b'):
            raise ValueError('CAD input must be STEP/STP/X_T/X_B')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with self.lock:
            data = self.get(project_id)
            duplicate = next((d for d in data['documents'] if d['sha256'] == digest and d['role'] == role), None)
            if duplicate:
                if role == 'cad' and (data['model'] or {}).get('status') not in ('ready', 'preview'):
                    self.rebuild_model(project_id, duplicate['id'])
                return duplicate
            document_id = uuid.uuid4().hex
            directory = self._dir(project_id) / 'assets'
            directory.mkdir(exist_ok=True)
            stored = directory / (document_id + suffix)
            shutil.copyfile(source, stored)
            record = {'id': document_id, 'role': role, 'filename': Path(filename).name,
                      'path': str(stored), 'sha256': digest, 'page_count': 0}
            try:
                if role == 'msop':
                    record['method_pages'] = msop.read_msop(stored)
                    record['page_count'] = len(record['method_pages'])
                    record['method_parser_version'] = MSOP_PARSER_VERSION
                elif role == 'drawing':
                    import pymupdf
                    with pymupdf.open(stored) as pdf:
                        record['page_count'] = len(pdf)
                if role == 'cad':
                    data['model'] = self._convert_model(record)
                if role in ('msop', 'cad'):
                    # An anchor is valid only against the source document and exact CAD hash.
                    if data['bindings']:
                        data['events'].append({'at': now(), 'action': 'bindings_invalidated',
                                               'reason': 'MSOP 或 CAD 更换；原定位记录需重新核对。',
                                               'previous': data['bindings']})
                    data['bindings'] = {}
                    self._invalidate_annotations(data, 'MSOP 或 CAD 更换；需重新核对标记位置。')
                data['documents'].append(record)
                self._write(data)
                return record
            except Exception as exc:
                # Keep the original CAD source when this machine cannot decode
                # its Parasolid schema, so a later decoder/schema retry remains
                # possible and the Viewer can report why geometry is unavailable.
                if role == 'cad' and isinstance(exc, ValueError):
                    record['conversion_status'] = 'unavailable'
                    record['conversion_error'] = str(exc)[:1000]
                    data['documents'].append(record)
                    data['model'] = {
                        'status': 'unavailable', 'document_id': document_id,
                        'filename': record['filename'], 'source_sha256': digest,
                        'error': record['conversion_error'],
                    }
                    if data['bindings']:
                        data['events'].append({'at': now(), 'action': 'bindings_invalidated',
                                               'reason': 'CAD 源文件已更新；几何不可用，原定位记录需重新核对。',
                                               'previous': data['bindings']})
                    data['bindings'] = {}
                    self._invalidate_annotations(data, 'CAD 源文件已更新；几何不可用，需重新核对标记位置。')
                    self._write(data)
                    return record
                stored.unlink(missing_ok=True)
                stored.with_suffix('.mesh.json').unlink(missing_ok=True)
                raise

    def refresh_msop_index(self, project_id):
        """Reparse an existing MSOP when the coordinate-scope parser changes."""
        with self.lock:
            data = self.get(project_id)
            document = self.active_msop(data)
            if not document or document.get('method_parser_version') == MSOP_PARSER_VERSION:
                return False
            source = Path(document.get('path', ''))
            if not source.is_file():
                raise FileNotFoundError(f"Active MSOP source is unavailable: {source}")
            pages = msop.read_msop(source)
            if len(pages) != document.get('page_count'):
                raise ValueError('Reparsed MSOP page count differs from the stored source index')
            document['method_pages'] = pages
            document['method_parser_version'] = MSOP_PARSER_VERSION
            data['events'].append({'at': now(), 'action': 'msop_index_refreshed',
                'document_id': document['id'], 'parser_version': MSOP_PARSER_VERSION,
                'reason': 'coordinate_scope_parser_upgrade'})
            self._write(data)
            return True

    def _convert_model(self, document):
        source = Path(document['path'])
        mesh = source.with_suffix('.mesh.json')
        parasolid = source.suffix.lower() in ('.x_t', '.x_b')
        interpreter = (Path(__file__).resolve().parents[1] / '.venv-cad/bin/python'
                       if parasolid else Path('/usr/bin/python3'))
        if not interpreter.exists():
            raise ValueError('本地 CAD 解码环境未安装；请运行 scripts/setup_viewer_cad.sh')
        worker = Path(__file__).with_name('parasolid_worker.py' if parasolid else 'geometry_worker.py')
        process = subprocess.run([str(interpreter), str(worker), str(source), str(mesh)],
                                 capture_output=True, text=True, timeout=600 if parasolid else 180)
        result = next((line[len('ORACLE_RESULT='):] for line in process.stdout.splitlines()
                       if line.startswith('ORACLE_RESULT=')), None)
        if process.returncode or not result:
            error = next((line[len('ORACLE_ERROR='):] for line in process.stdout.splitlines()
                          if line.startswith('ORACLE_ERROR=')), '本地 CAD 解码或网格生成失败')
            raise ValueError(error[:600])
        metadata = json.loads(result)
        return metadata | {'status': metadata.get('status', 'ready'),
                           'document_id': document['id'], 'filename': document['filename'],
                           'mesh_file': str(mesh)}

    def rebuild_model(self, project_id, document_id=None):
        """Re-run an importer after decoder updates without changing the CAD source."""
        with self.lock:
            data = self.get(project_id)
            documents = [d for d in data['documents'] if d['role'] == 'cad'
                         and (document_id is None or d['id'] == document_id)]
            if not documents:
                raise KeyError('CAD document not found')
            document = documents[-1]
            model = self._convert_model(document)
            document['conversion_status'] = model['status']
            document.pop('conversion_error', None)
            if data['bindings']:
                data['events'].append({'at': now(), 'action': 'bindings_invalidated',
                                       'reason': '几何解码器更新；需重新核对特征面关联。',
                                       'previous': data['bindings']})
            data['bindings'] = {}
            self._invalidate_annotations(data, '几何解码器更新；需重新核对标记位置。')
            data['model'] = model
            data['events'].append({'at': now(), 'action': 'cad_rebuilt',
                                   'document_id': document['id'],
                                   'source_sha256': document['sha256'], 'status': model['status']})
            self._write(data)
            return model

    def document(self, project_id, document_id):
        return next((d for d in self.get(project_id)['documents'] if d['id'] == document_id), None)

    def reference_file(self, project_id, file_id):
        return next((f for f in self.get(project_id).get('reference_files', [])
                     if f.get('id') == file_id), None)

    def mesh(self, project_id):
        model = self.get(project_id)['model']
        if not model or model['status'] not in ('ready', 'preview'):
            raise ValueError('Real CAD geometry not available')
        return json.loads(Path(model['mesh_file']).read_text())

    def bind(self, project_id, number, face_ids, note):
        with self.lock:
            detail = self.detail(project_id, number)
            mesh = self.mesh(project_id)
            if mesh.get('metrology_ready') is False:
                raise ValueError('当前为显示预览；严格几何转换未通过，FAI 关联暂未启用')
            available = {f['id']: f for f in mesh['faces']}
            if not face_ids or len(face_ids) > 100 or len(set(face_ids)) != len(face_ids) or any(f not in available for f in face_ids):
                raise ValueError('Select valid, unique faces from the active CAD')
            data = self.get(project_id)
            binding = {'face_ids': face_ids, 'source_sha256': mesh['source_sha256'],
                       'document_id': detail['item']['document_id'], 'confirmed_at': now(),
                       'confirmation': 'manual', 'scope': '特征位置关联；不代表基准配准或 AOI 方法等效已验证',
                       'anchors': [available[f]['center'] for f in face_ids], 'note': note}
            data['bindings'][detail['item']['key']] = binding
            data['events'].append({'at': now(), 'action': 'manual_binding', 'fai': number, 'binding': binding})
            self._write(data)
            return binding

    def unbind(self, project_id, number):
        with self.lock:
            detail = self.detail(project_id, number)
            data = self.get(project_id)
            data['bindings'].pop(detail['item']['key'], None)
            data['events'].append({'at': now(), 'action': 'unbind', 'fai': number})
            self._write(data)

    def _invalidate_annotations(self, data, reason):
        if data.get('annotations'):
            data['events'].append({'at': now(), 'action': 'annotations_invalidated',
                                   'reason': reason, 'previous': data['annotations']})
        data['annotations'] = {}

    def annotate(self, project_id, number, anchors, note, status, cad_sha256, document_id,
                 origin='manual', confidence=None, candidate_face_ids=None, feature_verification=None):
        """Save source-scoped visual pins; preview geometry is allowed for this purpose."""
        if status not in ('pending_review', 'auto_marked', 'reviewed'):
            raise ValueError('Invalid annotation review status')
        if origin not in ('manual', 'msop_visual_candidate', 'geometry_candidate', 'local_model_candidate'):
            raise ValueError('Invalid annotation origin')
        with self.lock:
            detail = self.detail(project_id, number)
            mesh = self.mesh(project_id)
            if cad_sha256 != mesh['source_sha256'] or document_id != detail['item']['document_id']:
                raise ValueError('CAD 或 MSOP 已更新，请重新加载后标记')
            checked = validate_anchors(mesh, anchors)
            available_ids = {face['id'] for face in mesh['faces']}
            candidate_face_ids = list(dict.fromkeys(candidate_face_ids or []))
            if (len(candidate_face_ids) > 100
                    or any(face_id not in available_ids for face_id in candidate_face_ids)):
                raise ValueError('Candidate face set must reference valid faces in the active CAD')
            candidate_face_ids = candidate_face_ids or list(dict.fromkeys(a['face_id'] for a in checked))
            data = self.get(project_id)
            annotation = {
                'anchors': checked, 'face_ids': list(dict.fromkeys(a['face_id'] for a in checked)),
                'candidate_face_ids': candidate_face_ids,
                'source_sha256': cad_sha256, 'document_id': document_id,
                'status': status, 'origin': origin, 'method': 'visual_location', 'updated_at': now(),
                'document_sha256': self.active_msop(data)['sha256'],
                'note': note, 'units': mesh.get('units', 'mm'), 'metrology_ready': False,
                'confidence': confidence,
                'confidence_basis': 'model self-assessment; uncalibrated' if origin == 'local_model_candidate' else None,
                'scope': 'CAD 显示位置标记；未建立 MSOP 测量坐标系，不代表 AOI 方法等效或计量验收',
            }
            if feature_verification is not None:
                annotation['feature_verification'] = feature_verification
            registration_path = self._dir(project_id) / 'automatic-learning' / 'registration.json'
            if origin == 'local_model_candidate' and registration_path.is_file():
                annotation['registration_sha256'] = hashlib.sha256(registration_path.read_bytes()).hexdigest()
            data.setdefault('annotations', {})[detail['item']['key']] = annotation
            data['events'].append({'at': now(), 'action': 'visual_annotation', 'fai': number,
                                   'annotation': annotation})
            self._write(data)
            return annotation

    def quarantine_feature_conflicts(self, project_id, conflicts):
        """Neither conflicting peer is preferred; preserve original positions."""
        with self.lock:
            data=self.get(project_id)
            keys={row['peer_fai_key'] for row in conflicts}
            for key in keys:
                annotation=data.get('annotations',{}).get(key)
                if annotation:
                    annotation.update(status='stale',stale_reason='shared_feature_location_conflict',
                                      feature_conflicts=conflicts)
            data['events'].append({'at':now(),'action':'shared_feature_conflicts_quarantined',
                                   'fai_keys':sorted(keys),'anchor_positions_changed':False})
            self._write(data)

    def require_current_feature_integrity(self, project_id):
        from .feature_integrity import VERSION
        with self.lock:
            data=self.get(project_id);changed=[]
            for key,annotation in data.get('annotations',{}).items():
                verification=annotation.get('feature_verification') or {}
                if (annotation.get('origin')=='local_model_candidate' and annotation.get('status')=='auto_marked'
                        and (verification.get('integrity_version')!=VERSION or verification.get('status')!='confirmed')):
                    annotation.update(status='stale',stale_reason='feature_identity_review_required',
                                      stale_previous_status='auto_marked')
                    changed.append(key)
            if changed:
                data['events'].append({'at':now(),'action':'feature_integrity_upgrade',
                    'fai_keys':changed,'anchor_positions_changed':False,'integrity_version':VERSION})
                self._write(data)
            return changed

    def invalidate_stale_local_model_annotations(self, project_id, registration_sha256):
        """Hide automatic marks tied to an older/missing coordinate registration.

        The original annotation is retained as history. The next automatic pass
        may replace it using the active registration and local model.
        """
        with self.lock:
            data = self.get(project_id)
            changed = []
            independent = []
            active = self.active_msop(data) or {}
            cad = (data.get('model') or {}).get('source_sha256')
            receipts = [data.get('local_model_marking') or {}]
            for path in (self._dir(project_id)/'automatic-learning'/'marking-runs').glob('*.json'):
                saved=json.loads(path.read_text())
                if (saved.get('cad_sha256')==cad and saved.get('msop_sha256')==active.get('sha256')
                        and saved.get('historical_source_hash_independently_recorded')):
                    receipts.append(saved.get('state') or {})
            audits = [row for receipt in receipts for row in receipt.get('items',[])
                      if row.get('result')=='accepted' and row.get('visual_fallback')]
            for key, annotation in data.get('annotations', {}).items():
                # Independent view-ray marks must not disappear merely because
                # another FAI's coordinate-registration receipt changed. The
                # retained local-model audit must reproduce every current anchor.
                if (annotation.get('origin')=='local_model_candidate' and annotation.get('status')=='auto_marked'
                        and annotation.get('source_sha256')==cad
                        and annotation.get('document_sha256')==active.get('sha256')
                        and not annotation.get('registration_independent')):
                    for audit in audits:
                        locations=audit.get('locations') or [];anchors=annotation.get('anchors') or []
                        if (str(audit.get('fai'))!=key.rsplit(':',1)[-1] or not anchors
                                or len(locations)!=len(anchors)):continue
                        try:
                            matches=all(location.get('placement_method') in ('model_view_ray','local_model_selected_spacing_hypothesis')
                                and location.get('face_id')==anchor['face_id']
                                and len(location['position'])==len(anchor['position'])==3
                                and all(abs(float(a)-float(b))<=1e-8 for a,b in zip(location['position'],anchor['position']))
                                for location,anchor in zip(locations,anchors))
                        except (KeyError,TypeError,ValueError):matches=False
                        if matches:
                            annotation['registration_independent']=True
                            annotation.pop('registration_sha256',None)
                            independent.append(key);break
                if (annotation.get('origin') != 'local_model_candidate'
                        or annotation.get('status') == 'stale'
                        or (annotation.get('registration_independent') is True
                            and annotation.get('source_sha256')==cad
                            and annotation.get('document_sha256')==active.get('sha256'))
                        or annotation.get('registration_sha256') == registration_sha256):
                    continue
                previous = annotation.get('status')
                annotation['status'] = 'stale'
                annotation['stale_reason'] = 'automatic_registration_changed'
                annotation['stale_previous_status'] = previous
                annotation['stale_registration_sha256'] = registration_sha256
                changed.append(key)
            if independent:
                data['events'].append({'at':now(),'action':'independent_visual_placement_audits_verified',
                                       'fai_keys':independent,'anchor_positions_changed':False})
            if changed:
                data['events'].append({'at': now(), 'action': 'automatic_annotations_staled',
                    'registration_sha256': registration_sha256,
                    'fai_keys': changed})
            if changed or independent:self._write(data)
            return changed

    def update_local_model_marking(self, project_id, **changes):
        with self.lock:
            data = self.get(project_id)
            current = data.get('local_model_marking') or {}
            current.update(changes, updated_at=now())
            data['local_model_marking'] = current
            self._write(data)
            return current

    def remove_annotation(self, project_id, number):
        with self.lock:
            detail = self.detail(project_id, number)
            data = self.get(project_id)
            data.setdefault('annotations', {}).pop(detail['item']['key'], None)
            data['events'].append({'at': now(), 'action': 'annotation_removed', 'fai': number})
            self._write(data)

    def task(self, project_id, number, question):
        with self.lock:
            detail = self.detail(project_id, number)
            data = self.get(project_id)
            task = {'id': uuid.uuid4().hex, 'fai': number, 'question': question,
                    'status': 'queued', 'created_at': now(), 'result': None, 'error': None,
                    'document_id': detail['item']['document_id'],
                    'cad_sha256': (data['model'] or {}).get('source_sha256'),
                    'input_context': detail['context'], 'binding': detail['item']['binding'],
                    'annotation': detail['item']['annotation']}
            if task['annotation']:
                task['input_context'] += '\n\n视觉标记（显示坐标，不是测量结果；坐标系尚未配准）：\n' + json.dumps(task['annotation'], ensure_ascii=False)
            data['tasks'].append(task)
            self._write(data)
            return task

    def update_task(self, project_id, task_id, **changes):
        with self.lock:
            data = self.get(project_id)
            task = next(t for t in data['tasks'] if t['id'] == task_id)
            task.update(changes, updated_at=now())
            self._write(data)
            return task

    def recover_tasks(self):
        for project in self.listing():
            for path in list(self._dir(project['id']).glob('aoi-*.json'))+list(self._dir(project['id']).glob('assembly-presentations/*/job.json'))+list(self._dir(project['id']).glob('automatic-learning/state.json')):
                record=json.loads(path.read_text())
                if record.get('status') in ('queued','running'):
                    record.update(status='failed',error='服务重启，任务未完成；已完成步骤保留，可重新提交。')
                    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False));temp.replace(path)
            marking=self.get(project['id']).get('local_model_marking') or {}
            if marking.get('status') in ('queued','running'):
                self.update_local_model_marking(project['id'],status='failed',error='服务重启，自动标记未完成；已有建议保留，可重新提交。')
            for task in self.get(project['id'])['tasks']:
                if task['status'] in ('queued','running'):
                    self.update_task(project['id'], task['id'], status='failed', error='服务重启，任务未完成；请重新提交。')

    def bootstrap(self, sources: Path):
        if self.listing() or not sources.exists():
            return
        candidates = list(sources.glob('*MSOP.pdf'))
        if not candidates:
            return
        data = self.create('D52 Frame Welding Assy', '613-13599', '06')
        self.add_document(data['id'], 'msop', candidates[0], candidates[0].name)
        for path in sources.glob('*CYIA.pdf'):
            self.add_document(data['id'], 'drawing', path, path.name)
        for path in sources.glob('*CYIA.x_t'):
            self.add_document(data['id'], 'cad', path, path.name)
