import json
from types import SimpleNamespace
from oracle_viewer import local_matching


def test_unrelated_frames_cannot_exhaust_a_fai_origin_search_budget(tmp_path,monkeypatch):
    points=[{'name':f'P{i}','xyz':xyz,'program':'UP','coordinate_system':'A','pages':[1]}
        for i,xyz in enumerate([[1,1,0],[2,1,0],[1,2,0]])]
    frame={'program':'UP','coordinate_system':'A','rotation':[[1,0,0],[0,1,0],[0,0,1]],'translation_mm':[3,3,0]}
    unrelated=[{**frame,'program':'DOWN','coordinate_system':f'UNUSED_{i}','translation_mm':[9,9,9]} for i in range(5)]
    store=SimpleNamespace(_dir=lambda _:tmp_path,annotations=[])
    store.annotate=lambda *args,**kwargs:store.annotations.append((args,kwargs))
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-1.png'])
    monkeypatch.setattr(local_matching,'_planar_frame_hypotheses',lambda *args:[])
    monkeypatch.setattr(local_matching,'_refine_model_selected_xyz_frames',lambda mesh,points,frames:(frames,[]))
    monkeypatch.setattr(local_matching,'_normal_origin_frame_hypotheses',lambda *args:[])
    monkeypatch.setattr(local_matching,'_review_visual_entity_group',lambda *args,**kwargs:{
        'status':'confirmed','feature_links':[]})
    checked=[]
    def project(mesh,source,frames,detail,choices):
        assert len(source)==3 and len(frames)==1 and frames[0]['coordinate_system']=='A'
        checked.append(frames[0]['translation_mm'])
        if frames[0]['translation_mm']!=[0.,3.,0.]:return None,{'reason':'test_complete_group_gate'}
        return ([{'face_id':'f','position':p['xyz'],'label_offset':[0,0]} for p in source],[],[]),None
    monkeypatch.setattr(local_matching,'_project_model_selected_xyz_group',project)
    class Model:
        def _generate_once(self,messages,**kwargs):
            return json.dumps({'decision':'AUTO_MARK','fai':1,'candidate_id':'O01',
                'source_point_count':3,'confidence':'low','evidence':'Full group agrees with original source.','source_pages':[1]}),False,60
    mesh={'source_sha256':'cad','faces':[{'id':'f','positions':[0,0,0,10,0,0,0,10,0],'indices':[0,1,2]}]}
    candidate={'known_source_points':points,'required_marker_count':3,'evidence':{
        'registration_diagnostic':{'selected_frame_transforms':[frame,*unrelated]}}}
    result=local_matching._mark_model_selected_xyz_group(store,{'id':'p'},
        {'number':1,'document_id':'doc','pages':[1]}, {'methods':[{'page':1,'text':'FAI 1.'}]},Model(),mesh,candidate)
    assert result['result']=='accepted' and len(store.annotations)==1
    assert [0.,3.,0.] in checked
