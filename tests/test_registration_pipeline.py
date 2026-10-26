import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from oracle_viewer import registration_pipeline


class LocalModel:
    loaded=True;bundle=object();openai_base_url='http://127.0.0.1:8081'
    def info(self):return dict(inference_backend='strata-native',model_alias='local-test')
    def _generate_once(self,*args,**kwargs):
        raise AssertionError('empty frame receipts should not invoke inference')


class Store:
    def __init__(self,root):self.root=root;self.updates=[]
    def get(self,pid):return dict(model=dict(source_sha256='cad'))
    def active_msop(self,project):return dict(sha256='msop')
    def _dir(self,pid):return self.root/pid
    def update_local_model_marking(self,pid,**data):self.updates.append(data)


def test_matching_receipt_is_reused_without_running_decoder(tmp_path,monkeypatch):
    store=Store(tmp_path);output=store._dir('p')/'automatic-learning/registration.json'
    output.parent.mkdir(parents=True);output.write_text(json.dumps(dict(cad_sha256='cad',msop_sha256='msop',
        algorithm_sha256=registration_pipeline.registration_algorithm_sha256())))
    def forbidden(*args,**kwargs):raise AssertionError('matching receipt must be reused')
    monkeypatch.setattr(registration_pipeline.subprocess,'Popen',forbidden)
    assert registration_pipeline.ensure_registration(store,'p',LocalModel())==output


@pytest.mark.parametrize('generated_sha',['cad','stale'])
def test_missing_receipt_runs_configured_state_root_and_rejects_stale_output(tmp_path,monkeypatch,generated_sha):
    store=Store(tmp_path)
    class Worker:
        returncode=0
        def __init__(self,args,**kwargs):
            assert args[args.index('--state-root')+1]==str(tmp_path.resolve())
            output=Path(args[args.index('--output')+1])
            output.write_text(json.dumps(dict(cad_sha256=generated_sha,msop_sha256='msop',
                algorithm_sha256=registration_pipeline.registration_algorithm_sha256())))
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def poll(self):return 0
    monkeypatch.setattr(registration_pipeline.subprocess,'Popen',Worker)
    if generated_sha=='stale':
        with pytest.raises(ValueError,match='stale'):registration_pipeline.ensure_registration(store,'p',LocalModel())
    else:
        assert registration_pipeline.ensure_registration(store,'p',LocalModel()).is_file()


def test_algorithm_mismatch_invalidates_old_registration_receipt(tmp_path,monkeypatch):
    store=Store(tmp_path);output=store._dir('p')/'automatic-learning/registration.json'
    output.parent.mkdir(parents=True);output.write_text(json.dumps(dict(cad_sha256='cad',msop_sha256='msop',
        algorithm_sha256='older-algorithm')))
    runs=[]
    class Worker:
        returncode=0
        def __init__(self,args,**kwargs):
            runs.append(args)
            Path(args[args.index('--output')+1]).write_text(json.dumps(dict(cad_sha256='cad',msop_sha256='msop',
                algorithm_sha256=registration_pipeline.registration_algorithm_sha256())))
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def poll(self):return 0
    monkeypatch.setattr(registration_pipeline.subprocess,'Popen',Worker)
    assert registration_pipeline.ensure_registration(store,'p',LocalModel())==output
    assert len(runs)==1


def test_local_model_selects_a_registration_hypothesis_without_algorithmic_default(tmp_path):
    options=[
        dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0],max_residual_mm=.08,
             cad_points_mm=[[1,2,3]],residuals_mm=[.08]),
        dict(rotation=[[-1,0,0],[0,-1,0],[0,0,1]],translation_mm=[5,5,0],max_residual_mm=.06,
             cad_points_mm=[[4,3,3]],residuals_mm=[.06]),
    ]
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=options[0],alternatives=[options[1]]))])
    class Model(LocalModel):
        def _generate_once(self,messages,**kwargs):
            text=messages[0]['attachments'][0]['parsed_text']
            assert 'P1' in text and 'H01' in text
            prompt=messages[0]['content']
            assert '单个坐标轴的符号反向' in prompt
            assert '两两距离' in prompt
            assert '不能单独作为弃答理由' in prompt
            assert kwargs['max_new_tokens']==768
            return (json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id='H01',program='UP',
                                    coordinate_system='ALI_A',source_point_count=1,
                                    evidence='MSOP frame and all points are consistent',confidence='medium')),False,42)
    store=SimpleNamespace()
    selected=registration_pipeline._local_model_registration_decisions(store,'p',Model(),receipt)
    decision=selected['local_model_registration']['frames'][0]
    assert decision['status']=='selected' and decision['hypothesis_index']==1
    assert decision['hypothesis_id']=='H01'
    assert selected['local_model_registration']['authority']=='local_model_decision'
    assert selected['local_model_registration']['model_identity']['decision_policy_sha256']


def test_local_model_can_select_robust_feature_family_transform():
    rotation=[[1,0,0],[0,1,0],[0,0,1]]
    base=dict(rotation=rotation,translation_mm=[20,20,2],max_residual_mm=24,
              p95_residual_mm=22,cad_points_mm=[[20,20,2]]*5,residuals_mm=[22]*5)
    family=dict(rotation=rotation,translation_mm=[0,0,2],max_residual_mm=.6,
                p95_residual_mm=.5,cad_points_mm=[[x,y,2] for x,y in [(0,0),(1,0),(1,1),(0,1),(2,2)]],
                residuals_mm=[.1,.2,.3,.4,.6])
    points=[dict(name=f'FRAME_P{i}',xyz=[float(i),float(i%2),0],pages=[12]) for i in range(5)]
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=points,
        triangle_refinement=dict(best=base,alternatives=[]),
        feature_family_hypotheses=[dict(feature_group='FRAME',
            support_point_names=[point['name'] for point in points],support_point_count=5,
            triangle_refinement=dict(best=family,alternatives=[]))])])
    class Model(LocalModel):
        def _generate_once(self,messages,**kwargs):
            context=messages[0]['attachments'][0]['parsed_text']
            assert 'FRAME' in context and 'support_point_mappings' in context
            assert '0.1 mm 是严格测量/常规几何候选门槛' in messages[0]['content']
            return (json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id='G00_00',
                program='UP',coordinate_system='ALI_A',source_point_count=5,
                evidence='The full frame point-family overlay is coherent with the source view.',
                confidence='medium')),False,48)

    result=registration_pipeline._local_model_registration_decisions(
        SimpleNamespace(), 'p', Model(), receipt)
    decision=result['local_model_registration']['frames'][0]
    assert decision['status']=='selected'
    assert decision['hypothesis_id']=='G00_00'
    assert decision['hypothesis_index'] is None
    assert decision['selected_transform']['support_feature_group']=='FRAME'
    assert decision['selected_transform']['translation_mm']==[0,0,2]


def test_restore_selected_candidate_uses_full_precision_without_changing_model_choice():
    exact_rotation=[[0.99999992,-0.00031,-0.00018],
                    [-0.00031,-0.99999994,0.00011],
                    [-0.00018,0.00011,-0.99999998]]
    family_option=dict(rotation=exact_rotation,translation_mm=[-0.01321,-64.37931,2.14729],
                       p95_residual_mm=.04570879,max_residual_mm=.04785382,
                       held_out_max_mm=.04655705,inlier_ratio=1.0,initialization='sample_ICP',
                       support_feature_group='FRAME',support_point_names=['FRAME_R03'],
                       support_point_count=36)
    receipt=dict(frames=[dict(program='UP',coordinate_system='ALI_ABC',
        feature_family_hypotheses=[dict(feature_group='FRAME',triangle_refinement={
            'best':family_option,'alternatives':[]})])],
        local_model_registration=dict(frames=[dict(program='UP',coordinate_system='ALI_ABC',
            status='selected',hypothesis_id='G00_00',model_confidence='high',
            selected_transform=dict(rotation=[[round(v,4) for v in row] for row in exact_rotation],
                                    translation_mm=[round(v,4) for v in family_option['translation_mm']]))]))

    result=registration_pipeline._restore_selected_candidate_precision(receipt)
    decision=result['local_model_registration']['frames'][0]
    assert decision['hypothesis_id']=='G00_00'
    assert decision['model_confidence']=='high'
    assert decision['selected_transform']['rotation']==exact_rotation
    assert decision['selected_transform']['translation_mm']==family_option['translation_mm']
    assert decision['selected_transform']['support_feature_group']=='FRAME'


def test_registration_inference_uses_shared_local_model_lock():
    option=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0],
                cad_points_mm=[[1,2,3]],residuals_mm=[.02],max_residual_mm=.02)
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=option,alternatives=[]))])
    class LockedModel(LocalModel):
        def __init__(self):self._lock=threading.Lock()
        def _generate_once(self,*args,**kwargs):
            assert self._lock.locked(), 'registration inference must serialize on the shared model lock'
            return (json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id='H00',program='UP',
                                    coordinate_system='ALI_A',source_point_count=1,
                                    evidence='Complete point group is consistent',confidence='low')),False,30)
    decision=registration_pipeline._local_model_registration_decisions(
        None,'p',LockedModel(),receipt)['local_model_registration']['frames'][0]
    assert decision['status']=='selected'


def test_multimodal_registration_attaches_msop_page_and_each_candidate_overlay(tmp_path,monkeypatch):
    option_a=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0],
        cad_points_mm=[[1,2,3]],residuals_mm=[.02],max_residual_mm=.02)
    option_b=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[1,0,0],
        cad_points_mm=[[2,2,3]],residuals_mm=[.03],max_residual_mm=.03)
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=option_a,alternatives=[option_b]))])
    class VisualModel(LocalModel):
        def info(self):return dict(inference_backend='strata-native',model_alias='local-test',
                                   image_inference='local_auxiliary_vision')
        def _generate_once(self,messages,**kwargs):
            attachment=messages[0]['attachments'][0]
            assert attachment['images']==['/tmp/msop-p4.png','/tmp/candidate-H00.png','/tmp/candidate-H01.png']
            assert kwargs['max_images']==3
            assert 'MSOP 点位页' in messages[0]['content']
            return (json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id='H01',program='UP',
                coordinate_system='ALI_A',source_point_count=1,evidence='MSOP and CAD projection agree',
                confidence='medium')),False,42)
    store=SimpleNamespace(get=lambda _pid:dict(model=dict(source_sha256='cad-hash')),
        mesh=lambda _pid:dict(faces=[]),_dir=lambda _pid:tmp_path)
    monkeypatch.setattr(registration_pipeline,'_registration_visual_evidence',
        lambda *args:(['/tmp/msop-p4.png','/tmp/candidate-H00.png','/tmp/candidate-H01.png'],
                      {'status':'attached','source_pages':[4],'cad_overlay_count':2}))
    result=registration_pipeline._local_model_registration_decisions(store,'p',VisualModel(),receipt)
    decision=result['local_model_registration']['frames'][0]
    assert decision['status']=='selected' and decision['hypothesis_id']=='H01'
    assert decision['visual_evidence']['cad_overlay_count']==2


def test_registration_visual_evidence_builds_candidate_overlays_and_keeps_msop_pages(tmp_path,monkeypatch):
    from oracle_viewer import local_matching
    from oracle_viewer import cad_view_atlas
    project=dict(id='p',model=dict(source_sha256='cad-hash'))
    store=SimpleNamespace(get=lambda _pid:project,mesh=lambda _pid:{'faces':[dict(
        id='face',occt_face_valid=True,positions=[0,0,0,1,0,0,0,1,0],indices=[0,1,2])]},
                          _dir=lambda _pid:tmp_path)
    monkeypatch.setattr(local_matching,'_method_page_images',
        lambda _store,_project,detail,_model:['/tmp/msop-p4.png'])
    generated=[]
    def render(_mesh,_xyz,_rotation,_translation,path,hypothesis_id,**_kwargs):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_bytes(b'png')
        generated.append(hypothesis_id)
        return Path(path)
    monkeypatch.setattr(cad_view_atlas,'render_registration_hypothesis_atlas',render)
    model=SimpleNamespace(info=lambda:{'image_inference':'local_auxiliary_vision'})
    frame=dict(program='UP',coordinate_system='ALI_A',source_points=[
        dict(name='P1',xyz=[1,2,3],pages=[4])])
    hypotheses=[dict(hypothesis_id='H00',rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0]),
                dict(hypothesis_id='H01',rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[1,0,0])]
    images,metadata=registration_pipeline._registration_visual_evidence(
        store,'p',model,frame,hypotheses)
    assert images[0]=='/tmp/msop-p4.png' and len(images)==3
    assert generated==['H00','H01']
    assert metadata['status']=='attached' and metadata['source_pages']==[4]


def test_visual_allocation_failure_retries_text_only_with_local_model_authority(monkeypatch):
    option=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0],
                cad_points_mm=[[1,2,3]],residuals_mm=[.02],max_residual_mm=.02)
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=option,alternatives=[]))])
    class VisualModel(LocalModel):
        def __init__(self):self.images=[]
        def info(self):return dict(inference_backend='strata-native',model_alias='local-test')
        def _generate_once(self,messages,**kwargs):
            images=messages[0]['attachments'][0]['images'];self.images.append(list(images))
            if images:raise RuntimeError('local vision could not allocate GPU memory')
            return (json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id='H00',program='UP',
                coordinate_system='ALI_A',source_point_count=1,evidence='Numeric source and CAD mappings agree',
                confidence='low')),False,34)
    model=VisualModel()
    monkeypatch.setattr(registration_pipeline,'_registration_visual_evidence',
        lambda *args:(['/tmp/source.png','/tmp/overlay.png'],{'status':'attached'}))
    result=registration_pipeline._local_model_registration_decisions(None,'p',model,receipt)
    decision=result['local_model_registration']['frames'][0]
    assert decision['status']=='selected'
    assert model.images==[['/tmp/source.png','/tmp/overlay.png'],[]]
    assert decision['visual_evidence']['status']=='multimodal_inference_failed'
    assert 'RuntimeError' in decision['visual_evidence']['error']


def test_single_registration_hypothesis_still_requires_local_model_authorization():
    option=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0],
                cad_points_mm=[[1,2,3]],residuals_mm=[.02],max_residual_mm=.02)
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=option,alternatives=[]))])
    class Model(LocalModel):
        def _generate_once(self,messages,**kwargs):
            return (json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id='H00',program='UP',
                                    coordinate_system='ALI_A',source_point_count=1,
                                    evidence='The complete point group is consistent',confidence='low')),False,30)
    selected=registration_pipeline._local_model_registration_decisions(None,'p',Model(),receipt)
    assert selected['local_model_registration']['frames'][0]['status']=='selected'


def test_loaded_loopback_strata_model_needs_no_transformers_bundle():
    option=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0],
                cad_points_mm=[[1,2,3]],residuals_mm=[.02],max_residual_mm=.02)
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=option,alternatives=[]))])
    class StrataModel:
        loaded=True
        bundle=None
        openai_base_url='http://127.0.0.1:8081'
        _lock=threading.Lock()
        def info(self):return dict(inference_backend='strata-native',model_alias='qwen3.8-local')
        def _generate_once(self,*args,**kwargs):
            assert self._lock.locked()
            return (json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id='H00',program='UP',
                                    coordinate_system='ALI_A',source_point_count=1,
                                    evidence='All point mappings agree',confidence='medium')),False,24)
    decisions=registration_pipeline._local_model_registration_decisions(None,'p',StrataModel(),receipt)
    assert decisions['local_model_registration']['frames'][0]['status']=='selected'


def test_model_can_select_frame_transform_while_point_variants_remain_separately_unresolved():
    option=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0],
                cad_points_mm=[[1,2,3]],residuals_mm=[.02],max_residual_mm=.02)
    conflict=dict(name='P1',occurrences=[
        dict(program='UP',coordinate_system='ALI_A',page=4,xyz=[1,2,3]),
        dict(program='UP',coordinate_system='ALI_A',page=9,xyz=[1.2,2,3]),
    ])
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=[dict(name='P0',xyz=[1,2,3],pages=[4])],
        conflicts=[conflict],triangle_refinement=dict(best=option,alternatives=[]))])
    class Model(LocalModel):
        def _generate_once(self,messages,**kwargs):
            context=messages[0]['attachments'][0]['parsed_text']
            question=messages[0]['content']
            assert 'source_conflicts' in context and '后续独立的本地模型步骤' in question
            return (json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id='H00',program='UP',
                                    coordinate_system='ALI_A',source_point_count=1,
                                    evidence='Clean points support this transform',confidence='low')),False,30)
    selected=registration_pipeline._local_model_registration_decisions(None,'p',Model(),receipt)
    frame=selected['local_model_registration']['frames'][0]
    assert frame['status']=='selected'
    assert frame['unresolved_source_conflict_count']==1


def test_direct_transformers_registration_still_requires_a_loaded_bundle():
    option=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0],
                cad_points_mm=[[1,2,3]],residuals_mm=[.02],max_residual_mm=.02)
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=option,alternatives=[]))])
    class UnloadedTransformers:
        loaded=True
        bundle=None
        openai_base_url=''
        def info(self):return dict(inference_backend='transformers-peft',model_alias='local-test')
    with pytest.raises(RuntimeError,match='本地模型未就绪'):
        registration_pipeline._local_model_registration_decisions(None,'p',UnloadedTransformers(),receipt)


def test_registration_decisions_checkpoint_and_resume_after_local_inference_interruption():
    option=dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[0,0,0],
                cad_points_mm=[[1,2,3]],residuals_mm=[.02],max_residual_mm=.02)
    frames=[dict(program='UP',coordinate_system=f'ALI_{name}',
        source_points=[dict(name=f'P_{name}',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=dict(option),alternatives=[])) for name in ('A','B')]
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=frames)
    persisted=[]
    def checkpoint(value):persisted.append(json.loads(json.dumps(value)))

    def response(frame):
        return json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id='H00',program='UP',
            coordinate_system=frame['coordinate_system'],source_point_count=1,
            evidence='All source points agree with this transform',confidence='medium'))

    class InterruptedModel(LocalModel):
        calls=0
        def _generate_once(self,messages,**kwargs):
            self.calls+=1
            if self.calls==2:raise ConnectionError('temporary local endpoint interruption')
            frame=frames[self.calls-1]
            return response(frame),False,30

    interrupted=InterruptedModel()
    with pytest.raises(ConnectionError):
        registration_pipeline._local_model_registration_decisions(
            None,'p',interrupted,receipt,checkpoint=checkpoint)
    partial=persisted[-1]
    assert partial['local_model_registration']['status']=='running'
    assert len(partial['local_model_registration']['frames'])==1
    assert partial['local_model_registration']['frames'][0]['status']=='selected'

    class ResumedModel(LocalModel):
        calls=[]
        def _generate_once(self,messages,**kwargs):
            text=messages[0]['attachments'][0]['parsed_text']
            scope='ALI_B' if 'ALI_B' in text else 'ALI_A'
            self.calls.append(scope)
            frame=next(frame for frame in frames if frame['coordinate_system']==scope)
            return response(frame),False,30

    resumed=ResumedModel()
    completed=registration_pipeline._local_model_registration_decisions(
        None,'p',resumed,partial,checkpoint=checkpoint)
    assert resumed.calls==['ALI_B']
    assert completed['local_model_registration']['status']=='complete'
    assert [row['status'] for row in completed['local_model_registration']['frames']]==['selected','selected']
    assert [row['frame_index'] for row in completed['local_model_registration']['frames']]==[0,1]


def test_registration_tournament_resumes_from_last_local_model_comparison():
    options=[]
    for offset in (0,1,2):
        options.append(dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[offset,0,0],
            cad_points_mm=[[1+offset,2,3]],residuals_mm=[.02+offset*.01],max_residual_mm=.02+offset*.01))
    frame=dict(program='UP',coordinate_system='ALI_A',
        source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=options[0],alternatives=options[1:]))
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[frame])
    persisted=[]
    def checkpoint(value):persisted.append(json.loads(json.dumps(value)))

    class Model(LocalModel):
        calls=0
        fail_after_first=False
        def _generate_once(self,messages,**kwargs):
            self.calls+=1
            if self.fail_after_first and self.calls==2:
                raise ConnectionError('interrupted between registration comparisons')
            text=messages[0]['attachments'][0]['parsed_text']
            payload=json.loads(text[text.index('\n')+1:])
            selected=payload['hypotheses'][-1]['hypothesis_id']
            return json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id=selected,
                program='UP',coordinate_system='ALI_A',source_point_count=1,
                evidence='This candidate matches the cited source frame',confidence='medium')),False,30

    first=Model();first.fail_after_first=True
    with pytest.raises(ConnectionError):
        registration_pipeline._local_model_registration_decisions(None,'p',first,receipt,checkpoint=checkpoint)
    partial=persisted[-1]['local_model_registration']['frames'][0]
    assert partial['status']=='comparing'
    assert partial['completed_comparisons']==1
    assert partial['next_comparison_index']==2

    resumed=Model()
    completed=registration_pipeline._local_model_registration_decisions(
        None,'p',resumed,persisted[-1],checkpoint=checkpoint)
    assert resumed.calls==1
    decision=completed['local_model_registration']['frames'][0]
    assert decision['status']=='selected'
    assert decision['compared_hypothesis_count']==3 and len(decision['evidence'])==2


def test_pairwise_model_abstention_does_not_hide_remaining_registration_hypotheses():
    options=[]
    for offset in range(4):
        options.append(dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[offset,0,0],
            cad_points_mm=[[1+offset,2,3]],residuals_mm=[.02+offset*.01],max_residual_mm=.02+offset*.01))
    frame=dict(program='UP',coordinate_system='ALI_A',
        source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=options[0],alternatives=options[1:]))
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[frame])
    class Model(LocalModel):
        calls=0
        def _generate_once(self,messages,**kwargs):
            self.calls+=1
            payload=json.loads(messages[0]['attachments'][0]['parsed_text'].split('\n',1)[1])
            pair=payload['hypotheses']
            if self.calls==1:
                return (json.dumps(dict(decision='ABSTAIN',hypothesis_id=None,program='UP',
                    coordinate_system='ALI_A',source_point_count=1,
                    evidence='These two alternatives are both inconsistent',confidence='low')),False,30)
            chosen=pair[-1]['hypothesis_id']
            return (json.dumps(dict(decision='SELECT_TRANSFORM',hypothesis_id=chosen,program='UP',
                coordinate_system='ALI_A',source_point_count=1,
                evidence='This alternative is consistent with all source points',confidence='medium')),False,30)
    model=Model()
    result=registration_pipeline._local_model_registration_decisions(None,'p',model,receipt)
    decision=result['local_model_registration']['frames'][0]
    assert model.calls==3
    assert decision['status']=='selected'
    assert decision['decisive_comparison_count']==2
    assert len(decision['unresolved_comparisons'])==1


def test_frame_remains_abstained_when_every_registration_pair_is_ambiguous():
    options=[dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation_mm=[offset,0,0],
        cad_points_mm=[[1+offset,2,3]],residuals_mm=[.02],max_residual_mm=.02) for offset in (0,1,2)]
    receipt=dict(drawing_no='generic',cad_sha256='cad',msop_sha256='msop',frames=[dict(
        program='UP',coordinate_system='ALI_A',source_points=[dict(name='P1',xyz=[1,2,3],pages=[4])],
        triangle_refinement=dict(best=options[0],alternatives=options[1:]))])
    class Model(LocalModel):
        def _generate_once(self,messages,**kwargs):
            return (json.dumps(dict(decision='ABSTAIN',hypothesis_id=None,program='UP',
                coordinate_system='ALI_A',source_point_count=1,evidence='No unique transform',
                confidence='low')),False,30)
    result=registration_pipeline._local_model_registration_decisions(None,'p',Model(),receipt)
    decision=result['local_model_registration']['frames'][0]
    assert decision['status']=='abstained'
    assert decision['reason']=='all_registration_comparisons_abstained'
    assert len(decision['unresolved_comparisons'])==2
