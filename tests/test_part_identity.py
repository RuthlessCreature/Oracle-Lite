import json
from types import SimpleNamespace
from oracle_viewer.part_identity import cad_parts,select_physical_parts


def mesh():
    return {'source_sha256':'test','faces':[
        {'id':'left','body':1,'prototype_body':4,'positions':[0,0,0,1,0,0,0,1,0],'indices':[0,1,2]},
        {'id':'right','body':2,'prototype_body':4,'positions':[10,0,0,11,0,0,10,1,0],'indices':[0,1,2]}]}


def test_catalog_keeps_repeated_original_bodies_and_assembly_positions():
    parts=cad_parts(mesh())
    assert [p['part_id'] for p in parts]==['B1','B2']
    assert parts[0]['prototype_body']==parts[1]['prototype_body']
    assert parts[0]['bbox_mm']['min']==[0,0,0] and parts[1]['bbox_mm']['min']==[10,0,0]
    assert parts[0]['face_ids']==['left']


def test_only_native_exact_body_selection_is_accepted_and_never_marks_points(tmp_path):
    class Model:
        ids=['B1','B2']
        def _generate_once(self,messages,**kwargs):
            assert messages[0]['attachments'][0]['images'][0]=='original-msop.png'
            return json.dumps({'decision':'SELECT_PARTS','fai':4,'part_ids':self.ids,
                'source_feature':'FANG','evidence':'Two repeated parts match the original leader.',
                'confidence':'low'}),False,80
    model=Model();store=SimpleNamespace(_dir=lambda _:tmp_path)
    selected,audit=select_physical_parts(store,{'id':'p'}, {'number':4}, {'methods':[]},model,mesh(),['original-msop.png'])
    assert len(selected)==2 and audit['does_not_authorize_measurement_anchors']
    model.ids=['B3']
    selected,audit=select_physical_parts(store,{'id':'p'}, {'number':4}, {'methods':[]},model,mesh(),['original-msop.png'])
    assert selected is None and audit['status']=='unconfirmed'


def test_measurement_markers_cannot_escape_native_selected_body(tmp_path,monkeypatch):
    from oracle_viewer import local_matching
    from oracle_viewer.cad_view_atlas import project_point_to_view_uv
    m=mesh();store=SimpleNamespace(_dir=lambda _:tmp_path,annotations=[])
    store.annotate=lambda *args:store.annotations.append(args)
    monkeypatch.setattr(local_matching,'_method_page_images',lambda *args:['/tmp/msop-1.png'])
    class Model:
        def info(self):return {'image_inference':'strata_native_vision'}
        def _generate_once(self,messages,**kwargs):
            return json.dumps({'decision':'AUTO_MARK','fai':1,'markers':[{
                'view_id':'+Z','uv':project_point_to_view_uv(m,'+Z',[10.2,.2,0])[0],
                'feature':'CIR','source_pages':[1]}],'evidence':'Model chose this feature.','confidence':'low'}),False,80
    result=local_matching._mark_visual_fallback(store,{'id':'p','documents':[]},
        {'number':1,'document_id':'doc','pages':[1]},
        {'methods':[{'page':1,'text':'FAI 1 circle location.'}],'dependencies':[]},Model(),m,
        {'evidence':{'part_identity_context':{'part_ids':['B1'],'cad_sha256':'test'}}})
    assert not store.annotations and result['result']=='abstain_or_invalid'
    assert result['geometry_diagnostic']['reason']=='marker_outside_local_model_selected_parts'


def test_large_body_catalog_is_screened_completely_without_truncation(tmp_path):
    m={'source_sha256':'large','faces':[{'id':f'f{i}','body':i,
        'positions':[i*2,0,0,i*2+1,0,0,i*2,1,0],'indices':[0,1,2]} for i in range(1,70)]}
    seen=[]
    class Model:
        def _generate_once(self,messages,**kwargs):
            payload=json.loads(messages[0]['attachments'][0]['parsed_text'])
            if 'all_original_bodies' in payload:
                assert len(payload['all_original_bodies'])==69
                return json.dumps({'decision':'SELECT_HOST_PARTS','fai':4,'part_ids':['B1'],
                    'evidence':'Global source and host topology comparison.'}),False,40
            if 'visible_parts' in payload:
                ids=[p['part_id'] for p in payload['visible_parts']];seen.extend(ids)
                assert '主体零件' in messages[0]['content']
                return json.dumps({'decision':'SCREEN_PARTS','fai':4,
                    'part_ids':['B1'] if 'B1' in ids else [],'evidence':'Host geometry comparison.'}),False,40
            return json.dumps({'decision':'SELECT_PARTS','fai':4,'part_ids':['B1'],
                'source_feature':'Rail','evidence':'Original leader and host shape.','confidence':'low'}),False,40
    parts,audit=select_physical_parts(SimpleNamespace(_dir=lambda _:tmp_path),{'id':'p'},
        {'number':4},{'methods':[]},Model(),m,['original-msop.png'])
    assert parts[0]['part_id']=='B1' and len(seen)==69 and len(set(seen))==69
    assert len(audit['screening'])==4 and audit['does_not_authorize_measurement_anchors']
