"""Checksummed local project archives; restore forks a project, never overwrites it."""
from __future__ import annotations
import hashlib
import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path,PurePosixPath
from .store import now


def archive_project(store,pid,target):
    root=store._dir(pid).resolve();target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    manifest={};tmp=target.with_suffix('.tmp')
    # Freeze under the store lock so annotations and their plan references are
    # from one version. Assets are immutable source-hash-addressed records.
    with store.lock,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.is_symlink() or path.suffix=='.tmp':continue
            name=path.relative_to(root).as_posix();blob=path.read_bytes()
            manifest[name]=dict(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest());archive.writestr(name,blob)
        archive.writestr('oracle-backup-manifest.json',json.dumps(dict(version=1,project_id=pid,created_at=now(),files=manifest),ensure_ascii=False))
    tmp.replace(target);return target


def restore_project(store,source):
    try: archive=zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc: raise ValueError('无效的项目ZIP归档') from exc
    with archive:
        names=archive.namelist()
        if len(names)>10000 or len(names)!=len(set(names)):raise ValueError('Invalid archive entry count or duplicate paths')
        for name in names:
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:raise ValueError('Unsafe archive path')
        manifest=json.loads(archive.read('oracle-backup-manifest.json'))
        if set(names)-{'oracle-backup-manifest.json'}!=set(manifest['files']):raise ValueError('Archive manifest differs from contents')
        if sum(archive.getinfo(n).file_size for n in names)>2*1024**3:raise ValueError('Project archive exceeds 2GiB')
        pid=uuid.uuid4().hex
        with tempfile.TemporaryDirectory(dir=store.root,prefix='restore-') as temp:
            directory=Path(temp)
            for name,meta in manifest['files'].items():
                blob=archive.read(name)
                if len(blob)!=meta['bytes'] or hashlib.sha256(blob).hexdigest()!=meta['sha256']:raise ValueError('Archive checksum mismatch: '+name)
                path=directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
            project=json.loads((directory/'project.json').read_text());project['id']=pid
            project['restored_from']=dict(project_id=manifest['project_id'],created_at=now());project['local_model_marking']=None
            destination=store._dir(pid)
            for document in project['documents']:
                document['path']=str(destination/'assets'/Path(document['path']).name)
            if project.get('model') and project['model'].get('mesh_file'):
                project['model']['mesh_file']=str(destination/'assets'/Path(project['model']['mesh_file']).name)
            for task in project.get('tasks',[]):
                if task.get('status') in ('queued','running'):task.update(status='failed',error='归档恢复后需重新提交未完成任务')
            (directory/'project.json').write_text(json.dumps(project,ensure_ascii=False,allow_nan=False))
            # Cached plans belong to the original project. Keep them as history
            # and prevent a restored project from exporting them as current.
            for path in directory.glob('aoi-*.json'):
                data=json.loads(path.read_text());data['project_id']=pid;data['input_fingerprint']='restored_requires_replanning';data['evidence_key']='restored_requires_replanning'
                if data.get('status') in ('queued','running'):data['status']='failed'
                path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False))
            with store.lock:shutil.move(str(directory),str(destination))
    return store.summary(store.get(pid))
