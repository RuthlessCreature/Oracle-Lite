import json
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image

from oracle_viewer.cad_view_atlas import (ray_hit_view, render_cad_view_atlas,
    render_focused_view_retry, nearby_surface_view_candidates,
    distributed_surface_view_candidates,render_surface_candidate_atlas,
    project_point_to_view_uv)
from oracle_viewer import local_matching


def triangle_mesh():
    return dict(source_sha256='cad-hash',faces=[dict(id='face-1',occt_face_valid=True,
        positions=[0,0,0,10,0,0,0,10,0],indices=[0,1,2])])


def test_orthographic_atlas_renders_all_six_views_and_ray_hits_active_surface(tmp_path):
    mesh=triangle_mesh();target=tmp_path/'atlas.png'
    render_cad_view_atlas(mesh,target,panel_size=120)
    with Image.open(target) as image:
        assert image.size==(360,240)
    hit=ray_hit_view(mesh,'+Z',1/3,2/3,panel_size=480)
    assert hit and hit['face_id']=='face-1'
    assert max(abs(a-b) for a,b in zip(hit['position'],[10/3,10/3,0]))<1e-6
    assert ray_hit_view(mesh,'+Z',.98,.02) is None


def test_focused_retry_view_upscales_one_panel_and_marks_previous_model_picks(tmp_path):
    atlas=tmp_path/'atlas.png'
    render_cad_view_atlas(triangle_mesh(),atlas,panel_size=120)
    output=tmp_path/'focused.png'
    result=render_focused_view_retry(atlas,output,
        [dict(view_id='+Z',uv=[.5,.5],feature='FAI point')],
        [dict(view_id='+Z',uv=[.4,.4],source_point_name='P1')],panel_size=120)
    assert result==output
    with Image.open(output) as image:
        assert image.size==(240,240)
        assert image.getpixel((120,120))==(212,35,99)
    assert render_focused_view_retry(atlas,tmp_path/'mixed.png',
        [dict(view_id='+Z',uv=[.5,.5]),dict(view_id='-Z',uv=[.5,.5])],[],
        panel_size=120) is None


def test_failed_uv_retry_offers_only_nearby_cad_ray_hits_to_local_model(tmp_path):
    mesh=triangle_mesh()
    candidates=nearby_surface_view_candidates(mesh,'+Z',[.90,.90],panel_size=480)
    assert candidates
    assert len(candidates)<=8
    assert all(ray_hit_view(mesh,row['view_id'],*row['uv']) is not None for row in candidates)
    assert all(row['face_id']=='face-1' for row in candidates)
    atlas=tmp_path/'atlas.png';render_cad_view_atlas(mesh,atlas,panel_size=120)
    output=tmp_path/'retry.png'
    result=render_focused_view_retry(atlas,output,
        [dict(view_id='+Z',uv=[.90,.90],feature='failed edge')],[],panel_size=120,
        surface_candidates=candidates)
    assert result==output
    with Image.open(output) as image:
        assert image.size==(240,240)
        # Candidate hits are drawn in the distinct green proposal color.
        assert (22,131,74) in image.get_flattened_data()


def test_failed_uv_search_checks_pixel_scale_before_coarse_steps(monkeypatch):
    mesh=triangle_mesh()
    attempted=[]
    def narrow_surface(_triangles,_face_ids,view_id,u,v,*,panel_size=480,projection=None):
        attempted.append((u,v))
        if .005<abs(u-.5)<.007 and abs(v-.5)<1e-9:
            return dict(face_id='narrow-edge',position=[u*100,v*100,0],view_id=view_id,uv=[u,v])
        return None
    monkeypatch.setattr('oracle_viewer.cad_view_atlas._ray_hit_view_triangles',narrow_surface)
    candidates=nearby_surface_view_candidates(mesh,'+Z',[.5,.5],panel_size=480)
    assert candidates
    assert max(max(abs(row['uv'][i]-.5) for i in range(2)) for row in candidates)<.01
    assert len(attempted)>=16
    assert max(max(abs(uv[i]-.5) for i in range(2)) for uv in attempted[:16])<.01


def test_whole_view_recovery_candidates_are_exact_and_overlaid(tmp_path):
    mesh=triangle_mesh()
    candidates=distributed_surface_view_candidates(mesh,view_ids=['+Z'],grid_size=16,
        max_candidates_per_view=6)
    assert 1<=len(candidates)<=6
    assert all(row['candidate_role']=='global_surface_search' for row in candidates)
    assert all(ray_hit_view(mesh,row['view_id'],*row['uv']) is not None for row in candidates)
    atlas=tmp_path/'atlas.png';render_cad_view_atlas(mesh,atlas,panel_size=120)
    target=tmp_path/'candidate-atlas.png'
    render_surface_candidate_atlas(atlas,target,candidates,panel_size=120)
    with Image.open(target) as image:
        assert image.size==(360,240)
        assert (22,131,74) in image.get_flattened_data()
    focused=tmp_path/'focused-recovery.png'
    render_focused_view_retry(atlas,focused,[dict(view_id='+Z',uv=[.5,.5])],[],panel_size=120,scale=2,
        surface_candidates=candidates)
    with Image.open(focused) as image:
        assert image.size==(540,240)


def test_ray_projection_orientation_is_consistent_for_all_six_views():
    faces=[]
    quads={
        '+X':([1,0,0],[1,1,0],[1,1,1],[1,0,1]),
        '-X':([0,0,0],[0,0,1],[0,1,1],[0,1,0]),
        '+Y':([0,1,0],[0,1,1],[1,1,1],[1,1,0]),
        '-Y':([0,0,0],[1,0,0],[1,0,1],[0,0,1]),
        '+Z':([0,0,1],[1,0,1],[1,1,1],[0,1,1]),
        '-Z':([0,0,0],[0,1,0],[1,1,0],[1,0,0]),
    }
    for name,points in quads.items():
        faces.append(dict(id=name,positions=[v for point in points for v in point],indices=[0,1,2,0,2,3]))
    mesh=dict(faces=faces)
    expected={'+X':(1,.5,.5),'-X':(0,.5,.5),'+Y':(.5,1,.5),'-Y':(.5,0,.5),
              '+Z':(.5,.5,1),'-Z':(.5,.5,0)}
    for view,point in expected.items():
        hit=ray_hit_view(mesh,view,.5,.5)
        assert hit and hit['face_id']==view
        assert max(abs(a-b) for a,b in zip(hit['position'],point))<1e-6


def test_distinct_depth_points_can_share_the_same_orthographic_uv():
    quads={
        '+X':([1,0,0],[1,1,0],[1,1,1],[1,0,1]),
        '-X':([0,0,0],[0,0,1],[0,1,1],[0,1,0]),
        '+Y':([0,1,0],[0,1,1],[1,1,1],[1,1,0]),
        '-Y':([0,0,0],[1,0,0],[1,0,1],[0,0,1]),
        '+Z':([0,0,1],[1,0,1],[1,1,1],[0,1,1]),
        '-Z':([0,0,0],[0,1,0],[1,1,0],[1,0,0]),}
    mesh=dict(faces=[dict(id=name,positions=[v for point in pts for v in point],
        indices=[0,1,2,0,2,3]) for name,pts in quads.items()])
    low,_scale=project_point_to_view_uv(mesh,'+Z',[.25,.75,0])
    high,_scale=project_point_to_view_uv(mesh,'+Z',[.25,.75,1])
    assert np.allclose(low,high)


def test_distinct_xyz_source_markers_collapsed_to_one_uv_are_detected():
    markers=[dict(view_id='+Z',uv=[.4,.6],source_point_names=[name])
             for name in ('P1','P2','P3')]
    sources={name:dict(xyz=xyz) for name,xyz in zip(('P1','P2','P3'),
        ([0,0,0],[1,0,0],[0,1,0]))}
    groups=local_matching._collapsed_source_marker_groups(markers,sources)
    assert len(groups)==1 and len(groups[0][3])==3
    assert local_matching._collapsed_source_marker_groups(
        [dict(view_id='+Z',uv=[.4,.6],source_point_names=['P1'])],sources)==[]


def test_no_xyz_fai_is_placed_only_from_local_vision_decision(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=7,document_id='msop',pages=[12],headings=['FAI 7 ⌀4'],features=['CIR_N1'])
    detail=dict(methods=[dict(page=12,text='FAI 7 Measure visual circle CIR_N1.')],dependencies=[])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p12.png','/tmp/drawing-fai7.png'])

    class Model:
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,messages,**kwargs):
            attachment=messages[0]['attachments'][0]
            assert len(attachment['images'])==3 and attachment['images'][-1].endswith('cad-view-atlas-uv-grid-v2-cad-hash.png')
            return json.dumps(dict(decision='AUTO_MARK',fai=7,markers=[
                dict(view_id='+Z',uv=[1/3,2/3],source_pages=[12],feature='CIR_N1 center'),
                dict(view_id='+Z',uv=[.2,.8],source_pages=[12],feature='CIR_N1 lower extent')],
                source_pages=[12],evidence='Both requested circle locations are visible on the top view.',confidence='low')),False,80

    result=local_matching._mark_visual_fallback(store,project,item,detail,Model(),triangle_mesh())
    assert result['result']=='accepted' and result['visual_fallback'] and result['anchor_count']==2
    assert len(store.annotations)==1
    annotation=store.annotations[0]
    assert annotation[1]==7 and len(annotation[2])==2 and annotation[4]=='auto_marked' and annotation[7]=='local_model_candidate'
    assert all(anchor['face_id']=='face-1' for anchor in annotation[2])
    assert max(abs(a-b) for a,b in zip(annotation[2][0]['position'],[10/3,10/3,0]))<1e-6
    assert max(abs(a-b) for a,b in zip(annotation[2][1]['position'],[2,2,0]))<1e-6


def test_rejected_xyz_projection_falls_through_to_fai_scoped_local_registration(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=12,document_id='msop',pages=[14],headings=['FAI 12 profile points'])
    detail=dict(methods=[dict(page=14,text='FAI 12 Measure 3 vector points A,B,C.')],dependencies=[])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p14.png'])
    rejected_atlas=tmp_path/'rejected-group.png'
    Image.new('RGB',(30,30),'white').save(rejected_atlas)
    monkeypatch.setattr(local_matching,'_mark_model_selected_xyz_group',lambda *args,**kwargs:{
        'fai':12,'result':'abstain_or_invalid','reason':'model_xyz_group_geometry_gate_failed',
        'geometry_diagnostic':{'reason':'source_section_coplanarity_not_preserved'},
        'model_candidate_atlas_path':str(rejected_atlas)})
    points=[dict(name=name,xyz=xyz,pages=[14],program='UP',coordinate_system='ALI_A',
                 requires_variant_choice=False,coordinate_variants=[])
            for name,xyz in zip(('A','B','C'),([2,2,0],[8,2,0],[2,8,0]))]

    class Model:
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,messages,**kwargs):
            context=messages[0]['attachments'][0]['parsed_text']
            assert messages[0]['attachments'][0]['images'][-1]==str(rejected_atlas)
            assert 'rejected model-selected XYZ overlay' in context
            assert 'source_section_coplanarity_not_preserved' in context
            assert '自行提出并授权一个新的 FAI 级放置' in messages[0]['content']
            return json.dumps(dict(decision='AUTO_MARK',fai=12,markers=[
                dict(view_id='+Z',uv=uv,feature=name,source_pages=[14],source_point_names=[name])
                for name,uv in [('A',[.2,.8]),('B',[.8,.8]),('C',[.2,.2])]],
                registration_points=[
                    dict(name='A',view_id='+Z',uv=[.2,.8]),
                    dict(name='B',view_id='+Z',uv=[.8,.8]),
                    dict(name='C',view_id='+Z',uv=[.2,.2])],
                source_pages=[14],evidence='The source method and profile view identify this location.',
                confidence='low')),False,60

    candidate=dict(known_source_points=points,required_marker_count=3,
                   evidence={'known_source_points':points})
    result=local_matching._mark_visual_fallback(store,project,item,detail,Model(),triangle_mesh(),candidate)
    assert result['result']=='accepted'
    assert result['visual_fallback'] is True
    assert len(store.annotations)==1 and store.annotations[0][1]==12


def test_no_xyz_visual_fallback_rejects_partial_multi_location_hit_group(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=7,document_id='msop',pages=[12],headings=['FAI 7 weld spots'])
    detail=dict(methods=[dict(page=12,text='FAI 7 inspect all weld spots.')],dependencies=[])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p12.png'])
    class Model:
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,*args,**kwargs):
            return json.dumps(dict(decision='AUTO_MARK',fai=7,markers=[
                dict(view_id='+Z',uv=[1/3,2/3],feature='weld 1',source_pages=[12]),
                dict(view_id='+Z',uv=[.98,.02],feature='weld 2',source_pages=[12])],
                evidence='two welds',confidence='medium')),False,40
    result=local_matching._mark_visual_fallback(store,project,item,detail,Model(),triangle_mesh())
    assert result['result']=='abstain_or_invalid'
    assert result['reason']=='model_selected_pixel_group_did_not_fully_hit_cad_surface'
    assert store.annotations==[]


def test_no_xyz_visual_fallback_abstains_without_local_vision(tmp_path):
    result=local_matching._mark_visual_fallback(
        SimpleNamespace(_dir=lambda _pid:tmp_path),dict(id='p'),dict(number=3),dict(),
        SimpleNamespace(info=lambda:{'image_inference':'unsupported_by_native_text_model'}),triangle_mesh())
    assert result['result']=='abstain_or_invalid'
    assert result['reason']=='local_multimodal_vision_model_unavailable'


def test_no_xyz_visual_fallback_isolates_local_inference_failure(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=8,document_id='msop',pages=[12],headings=['FAI 8 circle'])
    detail=dict(methods=[dict(page=12,text='FAI 8 inspect circle.')],dependencies=[])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p12.png'])

    class Model:
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,*args,**kwargs):raise ConnectionError('local vision endpoint unavailable')

    result=local_matching._mark_visual_fallback(store,project,item,detail,Model(),triangle_mesh())
    assert result['result']=='abstain_or_invalid'
    assert result['reason']=='local_vision_inference_failed'
    assert 'ConnectionError' in result['error']
    assert store.annotations==[]


def test_visual_completion_requires_full_declared_point_group_and_supplies_known_xyz(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=9,document_id='msop',pages=[12],headings=['FAI 9 two points'])
    detail=dict(methods=[dict(page=12,text='FAI 9 Measure 2 vector points A,B.')],dependencies=[])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p12.png'])
    markers=[dict(view_id='+Z',uv=[1/3,2/3],source_pages=[12],feature='A'),
             dict(view_id='+Z',uv=[.2,.8],source_pages=[12],feature='B')]
    candidate=dict(required_marker_count=2,known_source_points=[dict(name='A',xyz=[1,2,3])],
        evidence={'registration_diagnostic':dict(geometry_result='item_point_residual_rejected',
            no_geometric_anchor_authorized=True)})

    class Model:
        def __init__(self,count):self.count=count
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,messages,**kwargs):
            evidence=messages[0]['attachments'][0]['parsed_text']
            assert '"name":"A"' in evidence and '[1,2,3]' in evidence
            assert 'item_point_residual_rejected' in evidence
            assert 'markers 必须恰好包含 2 项' in messages[0]['content']
            assert '不可沿用其中任何预测位置' in messages[0]['content']
            return json.dumps(dict(decision='AUTO_MARK',fai=9,markers=markers[:self.count],
                source_pages=[12],evidence='The complete point group is visible.',confidence='medium')),False,45

    incomplete=local_matching._mark_visual_fallback(
        store,project,item,detail,Model(1),triangle_mesh(),candidate)
    assert incomplete['result']=='abstain_or_invalid'
    assert incomplete['reason']=='model_visual_decision_invalid'
    assert store.annotations==[]

    complete=local_matching._mark_visual_fallback(
        store,project,item,detail,Model(2),triangle_mesh(),candidate)
    assert complete['result']=='accepted' and complete['anchor_count']==2
    assert len(store.annotations)==1 and len(store.annotations[0][2])==2


def test_visual_fallback_requires_local_model_choice_for_conflicting_xyz_variants(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=10,document_id='msop',pages=[97],headings=['FAI 10 points'])
    detail=dict(methods=[dict(page=97,text='FAI 10 Measure 1 vector point DATUMC_PNT_P04.')],
                dependencies=[dict(page=13,text='point row'),dict(page=75,text='alternate point row')])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.active_msop=lambda _project:{'method_pages':detail['dependencies']}
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p97.png'])
    choices=[
        dict(page=13,program='UP',coordinate_system='ALI_ABC_A',xyz=[-18.57,129.22,-.35]),
        dict(page=75,program='DOWN',coordinate_system='ALI_ABC_A',xyz=[-18.57,64.52,-.35]),
    ]
    selected=dict(name='DATUMC_PNT_P04',**choices[1])
    markers=[dict(view_id='+Z',uv=[1/3,2/3],source_pages=[97],feature='DATUMC_PNT_P04')]
    candidate=dict(required_marker_count=1,known_source_points=[dict(
        name='DATUMC_PNT_P04',xyz=None,coordinate_variants=choices)])

    class Model:
        calls=0
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,messages,**kwargs):
            self.calls+=1
            if 'Competing source rows:' in messages[0]['content']:
                assert '坐标消歧模型' in messages[0]['content']
                return json.dumps(dict(decision='SELECT_VARIANTS',fai=10,
                    coordinate_choices=[selected],evidence='The DOWN method selects the DOWN row.',
                    confidence='low')),False,60
            context=messages[0]['attachments'][0]['parsed_text']
            question=messages[0]['content']
            assert 'coordinate_variants' in context
            assert '独立坐标消歧步骤选定' in question
            return json.dumps(dict(decision='AUTO_MARK',fai=10,markers=markers,
                source_pages=[97],coordinate_choices=[selected],
                evidence='The DOWN method agrees with the selected source row.',
                confidence='low')),False,60

    model=Model()
    result=local_matching._mark_visual_fallback(
        store,project,item,detail,model,triangle_mesh(),candidate)
    assert result['result']=='accepted'
    assert model.calls==2
    assert result['coordinate_choices']==[selected]
    assert '本地模型选定的原文坐标版本' in store.annotations[0][3]


def test_complete_xyz_visual_group_uses_one_model_selected_rigid_placement(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=11,document_id='msop',pages=[20],headings=['FAI 11 four-point plane'])
    detail=dict(methods=[dict(page=20,text='FAI 11 Measure 4 vector points A,B,C,D.')],dependencies=[])
    face=dict(id='plane',occt_face_valid=True,positions=[0,0,2,10,0,2,10,10,2,0,10,2],
              indices=[0,1,2,0,2,3])
    mesh=dict(source_sha256='rigid-cad',faces=[face])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p20.png'])
    sources=[dict(name=name,xyz=xyz,pages=[20],program='UP',coordinate_system='ALI_A',requires_variant_choice=False,coordinate_variants=[])
             for name,xyz in zip(('A','B','C','D'),([-4,-4,0],[4,-4,0],[4,4,0],[-4,4,0]))]
    cad_points=[(1,1,2),(9,1,2),(9,9,2),(1,9,2)]
    markers=[dict(source_point_names=[name],view_id='+Z',uv=uv,feature=name,source_pages=[20])
             for name,uv in zip(('A','B','C','D'),([.2,.2],[.8,.2],[.8,.8],[.2,.8]))]
    registration_points=[dict(source_point_name=name,view_id='+Z',uv=[.5,.5])
                         for name in ('A','B','C','D')]
    hits=iter(cad_points+cad_points)
    monkeypatch.setattr(local_matching,'ray_hit_view',lambda *_args,**_kwargs:
        dict(face_id='plane',position=list(next(hits))))

    class Model:
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,messages,**kwargs):
            question=messages[0]['content']
            assert 'registration_points' in question and 'source_point_names' in question
            return json.dumps(dict(decision='AUTO_MARK',fai=11,markers=markers,
                registration_points=registration_points,
                source_pages=[20],coordinate_choices=[],evidence='All four point names match the CAD plane.',
                confidence='medium')),False,90

    candidate=dict(required_marker_count=4,known_source_points=sources,
                   evidence={'required_marker_count':4,'known_source_points':sources})
    result=local_matching._mark_visual_fallback(store,project,item,detail,Model(),mesh,candidate)
    assert result['result']=='accepted' and result.get('anchor_count')==4, result
    positions=[anchor['position'] for anchor in store.annotations[0][2]]
    assert all(abs(position[2]-2)<1e-8 for position in positions)
    assert '独立刚体变换约束' in store.annotations[0][3]


def test_registered_depth_points_may_share_uv_and_keep_distinct_xyz_anchors(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=12,document_id='msop',pages=[22],headings=['FAI 12 depth-separated points'])
    detail=dict(methods=[dict(page=22,text='FAI 12 Measure 4 vector points A,B,C,D.')],dependencies=[])
    quads={
        '+X':([1,0,0],[1,1,0],[1,1,1],[1,0,1]),
        '-X':([0,0,0],[0,0,1],[0,1,1],[0,1,0]),
        '+Y':([0,1,0],[0,1,1],[1,1,1],[1,1,0]),
        '-Y':([0,0,0],[1,0,0],[1,0,1],[0,0,1]),
        '+Z':([0,0,1],[1,0,1],[1,1,1],[0,1,1]),
        '-Z':([0,0,0],[0,1,0],[1,1,0],[1,0,0]),}
    mesh=dict(source_sha256='cube-cad',faces=[dict(id=name,
        positions=[v for point in pts for v in point],indices=[0,1,2,0,2,3])
        for name,pts in quads.items()])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p22.png'])
    xyz={'A':[.25,.25,0.],'B':[.25,.25,1.],'C':[.75,.25,0.],'D':[.25,.75,0.]}
    sources=[dict(name=name,xyz=point,pages=[22],program='UP',coordinate_system='ALI_A',
        requires_variant_choice=False,coordinate_variants=[]) for name,point in xyz.items()]
    markers=[dict(source_point_names=[name],view_id='+Z',
        uv=project_point_to_view_uv(mesh,'+Z',xyz[name])[0],feature=name,source_pages=[22])
        for name in xyz]
    mapped=[xyz[name] for name in xyz]
    hits=iter(mapped+mapped)
    monkeypatch.setattr(local_matching,'ray_hit_view',lambda *_args,**_kwargs:
        dict(face_id='+Z',position=list(next(hits))))

    class Model:
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,messages,**kwargs):
            return json.dumps(dict(decision='AUTO_MARK',fai=12,markers=markers,
                registration_points=[dict(source_point_name=name,view_id='+Z',
                    uv=project_point_to_view_uv(mesh,'+Z',xyz[name])[0]) for name in xyz],
                source_pages=[22],coordinate_choices=[],
                evidence='The registered XYZ frame identifies separate depth surfaces.',
                confidence='medium')),False,80

    candidate=dict(required_marker_count=4,known_source_points=sources,
        evidence={'required_marker_count':4,'known_source_points':sources})
    result=local_matching._mark_visual_fallback(store,project,item,detail,Model(),mesh,candidate)
    assert result['result']=='accepted' and result['anchor_count']==4,result
    positions=[row['position'] for row in store.annotations[0][2]]
    assert np.allclose(positions[0],xyz['A'])
    assert np.allclose(positions[1],xyz['B'])
    assert np.allclose(positions[2],xyz['C'])


def test_visual_registration_keeps_program_coordinate_systems_separate(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=12,document_id='msop',pages=[21],headings=['FAI 12 mixed frames'])
    detail=dict(methods=[dict(page=21,text='FAI 12 inspect both features.')],dependencies=[])
    face=dict(id='plane',occt_face_valid=True,positions=[0,0,2,10,0,2,10,10,2,0,10,2],
              indices=[0,1,2,0,2,3])
    mesh=dict(source_sha256='rigid-cad',faces=[face])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p21.png'])
    points=[]
    for name,xyz in zip(('A1','A2','A3','A4'),([-4,-4,0],[4,-4,0],[4,4,0],[-4,4,0])):
        points.append(dict(name=name,xyz=xyz,pages=[21],program='UP',coordinate_system='ALI_A',
                           requires_variant_choice=False,coordinate_variants=[]))
    for name,xyz in zip(('B1','B2'),([100,100,10],[110,100,10])):
        points.append(dict(name=name,xyz=xyz,pages=[21],program='DOWN',coordinate_system='ALI_B',
                           requires_variant_choice=False,coordinate_variants=[]))
    registration_points=[dict(source_point_name=point['name'],view_id='+Z',uv=[.5,.5]) for point in points]
    markers=[dict(source_point_names=['A1','A2','A3','A4'],view_id='+Z',uv=[.15,.15],
                  feature='registered A',source_pages=[21]),
             dict(source_point_names=['B1','B2'],view_id='+Z',uv=[.5,.5],
                  feature='visual B',source_pages=[21])]
    cad_points=[(1,1,2),(9,1,2),(9,9,2),(1,9,2),(2,2,2),(8,2,2),(5,5,2),(8,8,2)]
    hits=iter(cad_points)
    monkeypatch.setattr(local_matching,'ray_hit_view',lambda *_args,**_kwargs:
        dict(face_id='plane',position=list(next(hits))))

    class Model:
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,messages,**kwargs):
            return json.dumps(dict(decision='AUTO_MARK',fai=12,markers=markers,
                registration_points=registration_points,source_pages=[21],coordinate_choices=[],
                evidence='The UP frame maps independently; the DOWN point uses its selected view.',
                confidence='low')),False,100

    candidate=dict(required_marker_count=2,known_source_points=points)
    result=local_matching._mark_visual_fallback(store,project,item,detail,Model(),mesh,candidate)
    assert result['result']=='accepted'
    anchors=store.annotations[0][2]
    assert anchors[0]['position']==[5.0,5.0,2.0]
    assert anchors[1]['position']==[8.0,8.0,2.0]


def test_viewer_marking_entry_writes_visual_fai_only_after_local_model_response(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A',model={'source_sha256':'cad-hash'})
    item=dict(number=7,document_id='msop',pages=[12],headings=['FAI 7 welds'],features=[],binding=None,annotation=None)
    method=dict(page=12,program='UP',text='FAI 7 inspect weld spots.')
    detail=dict(methods=[method],dependencies=[])
    class Store:
        annotations=[]
        def get(self,_pid):return project
        def mesh(self,_pid):return triangle_mesh()
        def active_msop(self,_project):return {'method_pages':[method]}
        def items(self,_project):return [item]
        def detail(self,_pid,_number):return {'methods':[method],'dependencies':[]}
        def _dir(self,_pid):return tmp_path
        def annotate(self,*args):self.annotations.append(args)
        def update_local_model_marking(self,*args,**kwargs):pass
    store=Store()
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p12.png','/tmp/drawing-fai7.png'])
    class Model:
        loaded=True;bundle=object();openai_base_url='http://127.0.0.1:8081';openai_model='local-vlm';_lock=threading.RLock()
        def info(self):return {'inference_backend':'strata-native','model_alias':'local-vlm',
                               'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,*args,**kwargs):
            return json.dumps(dict(decision='AUTO_MARK',fai=7,markers=[
                dict(view_id='+Z',uv=[1/3,2/3],feature='weld A',source_pages=[12]),
                dict(view_id='+Z',uv=[.2,.8],feature='weld B',source_pages=[12])],
                evidence='Both weld locations are visible.',confidence='medium')),False,65
    generated=[dict(fai=7,anchors=[],note='no XYZ',origin='msop_visual_candidate',
                    candidate_kind='model_visual_localization')]
    result=local_matching.mark_geometry_candidates(store,Model(),None,'p',generated_candidates=generated)
    assert result['counts']['accepted']==1
    assert len(store.annotations)==1 and len(store.annotations[0][2])==2


def test_long_msop_location_table_is_compacted_without_dropping_xyz_rows(tmp_path,monkeypatch):
    project=dict(id='p',drawing_no='generic',revision='A')
    item=dict(number=14,document_id='msop',pages=[14],headings=['FAI 14 datum points'])
    points=[dict(name=f'P{i}',xyz=[i,2*i,0.5],pages=[14],program='UP',
                 coordinate_system='ALI_A') for i in range(1,5)]
    detail=dict(methods=[dict(page=14,text='FAI 14 measure these four points.\\nLocation Definitions\\n'+
        'redundant table row '*12000)],dependencies=[])
    store=SimpleNamespace(_dir=lambda _pid:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-p14.png'])

    class Model:
        def __init__(self):self.contexts=[]
        def info(self):return {'image_inference':'local_auxiliary_vision'}
        def _generate_once(self,messages,**kwargs):
            self.contexts.append(messages[0]['attachments'][0]['parsed_text'])
            return json.dumps(dict(decision='ABSTAIN',fai=14)),False,20

    model=Model()
    result=local_matching._mark_visual_fallback(store,project,item,detail,model,triangle_mesh(),
        dict(required_marker_count=4,known_source_points=points,evidence={}))
    assert result['reason']=='model_visual_decision_invalid'
    assert model.contexts
    compact=model.contexts[0]
    assert len(compact)<58000
    assert all(f'"name":"P{i}"' in compact and f'[{i},{2*i},0.5]' in compact for i in range(1,5))
    assert 'redundant table row' not in compact


def test_dense_registration_overlay_keeps_every_marker_id_above_32(tmp_path,monkeypatch):
    from oracle_viewer.cad_view_atlas import render_registration_group_atlas
    from PIL import ImageDraw
    recorded=[];original=ImageDraw.ImageDraw.text
    def recording(self,xy,text,*args,**kwargs):
        if str(text).isdigit():recorded.append(int(text))
        return original(self,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',recording)
    points=[[1+i*.1,1+(i%3)*.1,0] for i in range(33)]
    render_registration_group_atlas(triangle_mesh(),[dict(source_xyz=points,rotation=np.eye(3).tolist(),translation_mm=[0,0,0])],tmp_path/'dense.png',panel_size=240)
    assert set(range(1,34))<=set(recorded)
