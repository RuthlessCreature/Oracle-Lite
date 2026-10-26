from __future__ import annotations
import json
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest
import pymupdf
from fastapi import FastAPI
from fastapi.testclient import TestClient
from oracle_viewer.msop import parse_page, inventory, dependencies
from oracle_viewer.store import ViewerStore
from oracle_viewer.assistant import answer_project
from oracle_viewer.api import attach_viewer


def pdf(path, text):
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((20,30), text, fontsize=10)
        doc.save(path)
    return path


def test_signed_nominal_is_not_fai_range():
    p = parse_page('MSOP: FAI 17 -33.75 +0.10/-0.10\nMeasure FAI38_CIR_L\nFAI0037_Z',27)
    assert p['fai_ids'] == [17]
    assert p['note_fai_ids'] == []
    assert p['specification']['nominal'] == '-33.75'


def test_composite_coordinate_system_is_preserved_instead_of_truncated():
    p=parse_page('MSOP: FAI 43\nMeasured System ALI_ABC_A& ALI_ABD\nProgram UP',57)
    assert p['measured_system']=='ALI_ABC_A&ALI_ABD'


def test_marking_parser_upgrade_reindexes_existing_msop_from_source(store):
    viewer, project_id = store
    project=viewer.get(project_id)
    doc=viewer.active_msop(project)
    doc['method_parser_version']=1
    viewer._write(project)
    assert viewer.refresh_msop_index(project_id) is True
    refreshed=viewer.active_msop(viewer.get(project_id))
    assert refreshed['method_parser_version']==2
    assert viewer.get(project_id)['events'][-1]['action']=='msop_index_refreshed'
    assert viewer.refresh_msop_index(project_id) is False


def test_registration_update_stales_only_automatic_marks_from_the_old_frame(tmp_path):
    store=ViewerStore(tmp_path)
    project=store.create('p','drawing','A')
    project['annotations']={
        'msop:1':dict(origin='local_model_candidate',status='auto_marked',
                      registration_sha256='old',anchors=[dict(face_id='f',position=[0,0,0])]),
        'msop:2':dict(origin='manual',status='reviewed',anchors=[dict(face_id='f',position=[0,0,0])]),
        'msop:3':dict(origin='local_model_candidate',status='auto_marked',
                      registration_sha256='current',anchors=[dict(face_id='f',position=[0,0,0])]),
    }
    store._write(project)
    assert store.invalidate_stale_local_model_annotations(project['id'],'current')==['msop:1']
    annotations=store.get(project['id'])['annotations']
    assert annotations['msop:1']['status']=='stale'
    assert annotations['msop:1']['stale_previous_status']=='auto_marked'
    assert annotations['msop:2']['status']=='reviewed'
    assert annotations['msop:3']['status']=='auto_marked'


def test_range_conflict_and_duplicate_measurement_pages():
    pages = [parse_page('MSOP: FAI 7 0 +0.15/0.00',22),
             parse_page('MSOP: FAI 7 0.00 +0.15/0.00',23),
             parse_page('MSOP: FAI 54 to FAI 60\nThe FAI61-FAI79 are weld spots height',103),
             parse_page('MSOP: FAI 61 to FAI 79\nThe FAI61-FAI79 are weld tab pull test',104)]
    items = {i['number']:i for i in inventory(pages)}
    assert items[7]['pages'] == [22,23]
    assert len(items[7]['specifications']) == 2
    assert pages[2]['fai_ids'] == list(range(54,61))
    assert pages[2]['note_fai_ids'] == list(range(61,80))
    assert items[54]['warnings'] and items[61]['warnings']
    assert 53 not in items


def test_dependencies_preserve_program_and_final_alignment():
    pages = [parse_page('MSOP: FAI 23 62.58 +0.12/-0.12\nDimension System ALI_ABD\nProgram UP',33),
             parse_page('MSOP: ALI_ABD\nProgram UP\nDATUMB_LIN DATUMD_LIN BD_ORIGN',18),
             parse_page('MSOP: ALI_ABD\nProgram DOWN\nWRONG_ORIGIN',78),
             parse_page('MSOP: DATUM B\nProgram UP\nMeasured System ALI_ABC_A',11),
             parse_page('MSOP: DATUM D\nProgram UP\nMeasured System ALI_ABC_A',13),
             parse_page('MSOP: ALI_ABC_A\nProgram UP\nDATUMA_PLN',16),
             parse_page('MSOP: DATUM A\nProgram UP',9)]
    result = dependencies(pages,[pages[0]])
    assert {p['page'] for p in result} == {9,11,13,16,18,33}


def test_measurement_indicators_use_explicit_msop_dimension_terms():
    flat = parse_page('MSOP: FAI 7 0 +0.15/0.00\nDimension Flatness. Probe MASTER_P5', 22)
    delta_z = parse_page('MSOP: FAI 51 1.28 +0.07/-0.04\nDimension 2D distance point to point Parallel to Z axis', 99)
    y_value = parse_page('MSOP: FAI 23 62.58 +0.12/-0.12\nDimension Y Value', 33)
    x_distance = parse_page('MSOP: FAI 47 4.25 +0.15/-0.15\nDimension 2D distance from POINT to LINE parallel to X axis', 95)
    assert [x['symbol'] for x in flat['measurement_indicators']] == ['⏥']
    assert [x['symbol'] for x in delta_z['measurement_indicators']] == ['ΔZ']
    assert [x['symbol'] for x in y_value['measurement_indicators']] == ['Y']
    assert [x['symbol'] for x in x_distance['measurement_indicators']] == ['ΔX']


def test_supplemental_image_only_measurement_data_is_source_scoped(store):
    s, project_id = store
    doc = s.active_msop(s.get(project_id))
    s.set_supplemental_fai_data(project_id, doc['id'], {
        '23': {'source_pages': [1], 'measurement_features': ['F1 · RIGHT FANG'],
               'measurement_indicators': [{'symbol': 'F', 'label': 'PULL FORCE', 'source': 'table'}],
               'measurement_metrics': [{'location': 'F1 · RIGHT FANG', 'mean': '4.0 MIN',
                                        'minimum': '3.0', 'allowed_range': '2.0-2.5', 'unit': 'kgf'}]},
    })
    item = s.detail(project_id, 23)['item']
    assert item['measurement_indicators'][0]['symbol'] == 'F'
    assert item['measurement_metrics'][0]['mean'] == '4.0 MIN'
    assert item['features'] == ['F1 · RIGHT FANG']


@pytest.fixture
def store(tmp_path):
    s = ViewerStore(tmp_path/'state')
    p = s.create('test','test-001','01')
    source = pdf(tmp_path/'MSOP.pdf','MSOP: FAI 23 62.58 +0.12/-0.12\nProgram UP\nProbe VISION')
    s.add_document(p['id'],'msop',source,source.name)
    return s,p['id']


def test_persistence_input_isolation_and_bad_project_id(store):
    s,p = store
    assert ViewerStore(s.root).detail(p,23)['item']['pages'] == [1]
    with pytest.raises(ValueError):s.get('../../outside')
    with pytest.raises(KeyError):s.detail(p,24)
    with pytest.raises(ValueError):s.bind(p,23,['made-up-face'],'')


def test_unavailable_parasolid_is_preserved_for_later_retry(store,tmp_path,monkeypatch):
    s,p=store
    source=tmp_path/'newer-schema.x_t'
    source.write_bytes(b'X_T source remains intact')
    monkeypatch.setattr(s,'_convert_model',lambda doc: (_ for _ in ()).throw(
        ValueError('required exact schema catalog does not exist: sch_15003.sch_txt')))
    record=s.add_document(p,'cad',source,source.name)
    project=s.public_project(p)
    assert Path(record['path']).read_bytes()==source.read_bytes()
    assert record['conversion_status']=='unavailable'
    assert 'sch_15003.sch_txt' in record['conversion_error']
    assert project['geometry_status']=='unavailable'
    assert project['documents'][-1]['filename']=='newer-schema.x_t'
    assert project['model']['error'].endswith('sch_15003.sch_txt')
    monkeypatch.setattr(s, '_convert_model', lambda doc: {
        'status': 'preview', 'document_id': doc['id'],
        'filename': doc['filename'], 'source_sha256': doc['sha256']})
    s.rebuild_model(p)
    recovered = s.public_project(p)
    assert recovered['geometry_status'] == 'preview'
    assert recovered['documents'][-1]['conversion_status'] == 'preview'
    assert 'conversion_error' not in recovered['documents'][-1]
    assert 'error' not in recovered['model']
    assert recovered['model']['source_sha256'] == record['sha256']
    assert Path(record['path']).read_bytes() == source.read_bytes()


def put_mesh(s,p):
    data=s.get(p)
    path=s._dir(p)/'test.mesh.json'
    mesh={'source_sha256':'source-a','faces':[{'id':'face-a','center':[1,2,3],
          'positions':[0,0,0,1,0,0,0,1,0],'indices':[0,1,2]}]}
    path.write_text(json.dumps(mesh))
    data['model']={'status':'ready','source_sha256':'source-a','mesh_file':str(path)}
    s._write(data)


def test_annotation_preserves_joint_candidate_face_pool(store):
    s,p=store
    put_mesh(s,p)
    result=s.annotate(p,23,[{'face_id':'face-a','position':[.25,.25,0]}],
                      'joint surface fit','pending_review','source-a',
                      s.active_msop(s.get(p))['id'],'geometry_candidate',
                      candidate_face_ids=['face-a'])
    assert result['candidate_face_ids']==['face-a']


def test_binding_validates_source_faces_and_invalidation(store,tmp_path):
    s,p=store
    put_mesh(s,p)
    with pytest.raises(ValueError):s.bind(p,23,['face-b'],'')
    b=s.bind(p,23,['face-a'],'孔口位置，尚未配准')
    assert b['anchors']==[[1,2,3]]
    assert s.public_project(p)['located_count']==1
    assert 'mesh_file' not in s.public_project(p)['model']
    source=pdf(tmp_path/'new-MSOP.pdf','MSOP: FAI 23 62.60 +0.12/-0.12')
    s.add_document(p,'msop',source,source.name)
    assert s.public_project(p)['located_count']==0
    assert s.get(p)['events'][-1]['action']=='bindings_invalidated'
    assert s.get(p)['events'][-1]['previous']


def test_task_snapshot_survives_revisions_and_restart(store,tmp_path):
    s,p=store
    task=s.task(p,23,'出 AOI 方案')
    s.update_task(p,task['id'],status='running')
    source=pdf(tmp_path/'next-MSOP.pdf','MSOP: FAI 24 53.40 +0.12/-0.12')
    s.add_document(p,'msop',source,source.name)
    s.recover_tasks()
    t=s.get(p)['tasks'][0]
    assert t['status']=='failed'
    assert 'FAI 23' in t['input_context']
    assert s.items(s.get(p))[0]['number']==24


def test_project_model_call_never_uses_global_retriever():
    class Model:
        loaded=True
        _lock=threading.Lock()
        def _generate_once(self,messages,**kwargs):
            assert 'ONLY_CURRENT_SOURCE' in messages[0]['attachments'][0]['parsed_text']
            return 'draft',True,1024
    result=answer_project(Model(),'解释 FAI 23 的方法','ONLY_CURRENT_SOURCE')
    assert result['truncated'] is True
    assert result['review_status']=='draft'


def test_model_unavailable_reports_explicit_error():
    with pytest.raises(RuntimeError,match='尚未就绪'):
        answer_project(SimpleNamespace(loaded=False),'解释方法','source')


def test_api_document_pages_errors_and_binding(tmp_path):
    app=FastAPI()
    cfg=SimpleNamespace(state_dir=tmp_path/'state',training_output_dir=tmp_path/'output')
    state=SimpleNamespace(cfg=cfg,model=SimpleNamespace(loaded=False))
    attach_viewer(app,state)
    with TestClient(app) as c:
        p=c.post('/api/viewer/projects',json={'title':'integration','revision':'01'}).json()['id']
        assert c.get(f'/api/viewer/projects/{p}/automatic-learning').json()['status']=='not_started'
        assert c.post(f'/api/viewer/projects/{p}/automatic-learning').status_code==400
        source=pdf(tmp_path/'source.pdf','MSOP: FAI 23 62.58 +0.12/-0.12\nProgram UP')
        with source.open('rb') as f:
            r=c.post(f'/api/viewer/projects/{p}/documents',data={'role':'msop'},files={'file':('source.pdf',f,'application/pdf')})
        assert r.status_code==200
        doc=r.json()['id']
        assert c.get(f'/api/viewer/projects/{p}/fai/23').status_code==200
        assert c.get(f'/api/viewer/projects/{p}/fai/24').status_code==404
        image=c.get(f'/api/viewer/projects/{p}/documents/{doc}/pages/1.png')
        assert image.content.startswith(b'\x89PNG')
        assert c.get(f'/api/viewer/projects/{p}/documents/{doc}/pages/0.png').status_code==404
        assert c.post(f'/api/viewer/projects/{p}/fai/23/binding',json={'face_ids':['invented']}).status_code==400
        assert c.get('/viewer').status_code==200
        assert c.get('/viewer-static/vendor/three.module.js').status_code==200
        state.viewer_pool = app.state.viewer_pool
        state.viewer_pool.shutdown(wait=True)


def test_actual_msop_index_and_circle_draft():
    path=Path(__file__).resolve().parents[2]/'3suite'/'HEX D52 P2 Frame Welding Assy_613-13599-06_V02.01 MSOP.pdf'
    if not path.exists():pytest.skip('local project fixture not present')
    from oracle_viewer.msop import read_msop,context
    from talker.engineering_plan import source_checked_circle_plan
    pages=read_msop(path)
    assert len(pages)==104
    items={i['number']:i for i in inventory(pages)}
    assert set(items)==set(range(1,83))
    assert items[23]['pages']==[33]
    assert items[23]['features']==['CIR_N1_VISION']
    assert not items[37]['warnings']  # FAI38_CIR_L is a feature, not a conflicting FAI range.
    selected=[p for p in pages if 23 in p['fai_ids']]
    result=source_checked_circle_plan('FAI 23 AOI方案',context(pages,selected,path.name))
    assert result and 'BD_ORIGN' in result and '0.4223' in result


def test_real_step_worker_dimensions_topology_and_validity(tmp_path):
    freecad=Path('/usr/lib/freecad-python3/lib/FreeCAD.so')
    if not freecad.exists():pytest.skip('FreeCAD runtime not installed')
    source=tmp_path/'part.step'
    maker="import sys;sys.path.append('/usr/lib/freecad-python3/lib');import FreeCAD;import Part;Part.makeBox(100,60,2).cut(Part.makeCylinder(4,5,FreeCAD.Vector(20,20,-1))).exportStep(sys.argv[1])"
    subprocess.run(['/usr/bin/python3','-c',maker,str(source)],check=True,capture_output=True)
    target=tmp_path/'mesh.json'
    from oracle_viewer import geometry_worker
    subprocess.run(['/usr/bin/python3',geometry_worker.__file__,str(source),str(target)],check=True,capture_output=True)
    mesh=json.loads(target.read_text())
    assert mesh['brep_valid']
    assert mesh['bbox']==[0,0,0,100,60,2]
    assert mesh['face_count']==7
    assert any(f['surface']=='Cylinder' and f['radius']==4 for f in mesh['faces'])
    assert mesh['volume_mm3']==pytest.approx(12000-3.141592653589793*4**2*2)
    assert len(set(f['id'] for f in mesh['faces']))==7
    assert all(f['positions'] and f['indices'] for f in mesh['faces'])


def test_talker_and_viewer_use_same_project_state_and_model(tmp_path,monkeypatch):
    import talker.app as module
    from talker.config import TalkerConfig
    cfg=TalkerConfig(tmp_path/'output',tmp_path/'talker')
    cfg.ensure_dirs()
    class FakeModel:
        loaded=True
        def __init__(self,*args):self._lock=threading.Lock()
        def load(self):pass
        def info(self):return {'loaded':True}
        def answer(self,*args):raise AssertionError('Global retrieval must not serve a project-scoped query')
        def _generate_once(self,messages,**kwargs):
            source=messages[0]['attachments'][0]['parsed_text']
            assert 'FAI 24 53.40' in source
            assert 'FAI 23 62.58' not in source
            return 'FAI 24 current-project method',False,20
    monkeypatch.setattr(module,'load_config',lambda *args:cfg)
    monkeypatch.setattr(module,'TalkerModel',FakeModel)
    app=module.create_app()
    with TestClient(app) as c:
        p=c.post('/api/viewer/projects',json={'title':'shared'}).json()['id']
        with pymupdf.open() as doc:
            doc.new_page().insert_text((20,30),'MSOP: FAI 23 62.58 +0.12/-0.12\nProgram UP')
            doc.new_page().insert_text((20,30),'MSOP: FAI 24 53.40 +0.12/-0.12\nProgram UP')
            data=doc.tobytes()
        c.post(f'/api/viewer/projects/{p}/documents',data={'role':'msop'},files={'file':('MSOP.pdf',data,'application/pdf')})
        # An explicit new FAI overrides the originating Viewer selection even after Chinese text.
        r=c.post('/api/chat',data={'message':'解释一下FAI24的方法','project_id':p,'fai':23})
        assert r.status_code==200,r.text
        assert r.json()['answer']=='FAI 24 current-project method'
        tasks=c.get(f'/api/viewer/projects/{p}').json()['tasks']
        assert tasks[0]['fai']==24 and tasks[0]['status']=='complete'
        assert tasks[0]['conversation_id']==r.json()['conversation_id']
        assert app.state.talker.model is not None  # No Viewer model instance is constructed.


def test_project_model_flags_known_method_field_errors():
    class Model:
        loaded=True
        _lock=threading.Lock()
        def _generate_once(self,*args,**kwargs):
            return 'ALI_ABD 程序。所有基准采点使用 MASTER_P5。',False,20
    result=answer_project(Model(),'只解释原方法，不做方案','Probe MASTER_P5\nProbe MASTER_P3_SHANK\nProgram UP')
    assert result['engine']=='shared-local-model'
    assert len(result['validation_warnings'])==2


def test_display_preview_is_renderable_but_cannot_bind_fai(store):
    s,p=store
    put_mesh(s,p)
    data=s.get(p);data['model']['status']='preview';s._write(data)
    path=Path(data['model']['mesh_file'])
    mesh=json.loads(path.read_text());mesh['metrology_ready']=False;path.write_text(json.dumps(mesh))
    assert s.mesh(p)['faces'][0]['id']=='face-a'
    with pytest.raises(ValueError,match='FAI'):
        s.bind(p,23,['face-a'],'')


def test_rebuild_preserves_source_and_audits_decoder_change(store,tmp_path,monkeypatch):
    s,p=store
    put_mesh(s,p)
    s.bind(p,23,['face-a'],'old association')
    source=tmp_path/'original.x_t';source.write_bytes(b'original CAD bytes')
    data=s.get(p);record={'id':'cad1','role':'cad','filename':'model.x_t','path':str(source),
                         'sha256':'original-hash','page_count':0}
    data['documents'].append(record);s._write(data)
    monkeypatch.setattr(s,'_convert_model',lambda doc:{'status':'preview','document_id':doc['id'],
                                                    'source_sha256':doc['sha256']})
    s.rebuild_model(p)
    after=s.get(p)
    assert source.read_bytes()==b'original CAD bytes'
    assert after['documents'][-1]==record | {'conversion_status': 'preview'}
    assert after['bindings']=={}
    assert after['events'][-2]['previous']
    assert after['events'][-1]['action']=='cad_rebuilt'
    assert after['model']['status']=='preview'


def test_parasolid_assembly_repeated_instances_and_transform():
    # Geometry kernel is intentionally absent: test source scene-graph placement separately.
    from oracle_viewer.parasolid_worker import instantiate
    from types import SimpleNamespace as N
    def node(index,kind,**fields):
        return {'index':index,'node_type':kind,'fields':[
            {'definition':{'name':name},'values':[{'value':v} for v in values]}
            for name,values in fields.items()]}
    brep=N(bodies=[N(id=0,source=N(node_index=20),regions=[0])],
           regions=[N(shells=[0])],shells=[N(back_faces=[0],front_faces=[])],
           faces=[N(source=N(node_index=30),surface=0)],surfaces=[N(kind=N(value='plane'))])
    prototypes=[{'source_face_id':0,'positions':[0,0,0,1,0,0,0,1,0],'indices':[0,1,2]}]
    nodes=[node(1,10,sub_instance=[2]),node(2,11,part=[20],transform=[0],next_in_part=[3]),
           node(3,11,part=[20],transform=[4],next_in_part=[0]),
           node(4,100,rotation_matrix=[1,0,0,0,1,0,0,0,1],translation_vector=[[.01,.02,.03]],scale=[2])]
    faces,count=instantiate(prototypes,brep,nodes,'a'*64)
    assert count==2 and len(faces)==2
    assert faces[0]['positions']==prototypes[0]['positions']
    assert faces[1]['positions']==[20,40,60,22,40,60,20,42,60]
    assert faces[0]['id']!=faces[1]['id']
    assert faces[1]['instance_path']==[1,3]
    assert faces[1]['source_node_index']==30


def test_api_backup_restore_and_invalid_archive(tmp_path):
    app=FastAPI();cfg=SimpleNamespace(state_dir=tmp_path/'state',training_output_dir=tmp_path/'output')
    state=SimpleNamespace(cfg=cfg,model=SimpleNamespace(loaded=False));attach_viewer(app,state)
    with TestClient(app) as client:
        pid=client.post('/api/viewer/projects',json={'title':'Archive example','drawing_no':'1','revision':'A'}).json()['id']
        archived=client.get(f'/api/viewer/projects/{pid}/backup');assert archived.status_code==200
        restored=client.post('/api/viewer/projects/restore',files={'file':('example.zip',archived.content,'application/zip')})
        assert restored.status_code==200 and restored.json()['id']!=pid
        assert len(client.get('/api/viewer/projects').json())==2
        invalid=client.post('/api/viewer/projects/restore',files={'file':('bad.zip',b'not a zip','application/zip')})
        assert invalid.status_code==400


@pytest.mark.parametrize('method,position,preserved',[
    ('model_view_ray',[1,2,3],True),('rigid_xyz_transform',[1,2,3],False),
    ('model_view_ray',[1,2,4],False)])
def test_only_proven_independent_model_marks_survive_registration_updates(tmp_path,method,position,preserved):
    viewer=ViewerStore(tmp_path);p=viewer.create('p','drawing','A')
    p['model']={'source_sha256':'cad'}
    p['documents']=[{'id':'msop','role':'msop','sha256':'msop'}]
    p['annotations']={'msop:1':dict(origin='local_model_candidate',status='auto_marked',
        source_sha256='cad',document_sha256='msop',registration_sha256='old',
        anchors=[dict(face_id='f',position=[1,2,3])])}
    p['local_model_marking']={'items':[dict(fai=1,result='accepted',visual_fallback=True,
        locations=[dict(face_id='f',position=position,placement_method=method)])]}
    viewer._write(p)
    changed=viewer.invalidate_stale_local_model_annotations(p['id'],'new')
    annotation=viewer.get(p['id'])['annotations']['msop:1']
    assert (changed==[]) is preserved
    assert annotation['anchors']==[dict(face_id='f',position=[1,2,3])]
    assert (annotation.get('registration_independent') is True) is preserved
