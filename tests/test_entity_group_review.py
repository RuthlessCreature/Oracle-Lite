import hashlib
import json
from types import SimpleNamespace

import pytest

from oracle_viewer.local_matching import _review_visual_entity_group


@pytest.mark.parametrize('change,confirmed',[
    ({},True),
    ({'decision':'REJECT','all_entities_match':False,
      'issues':[{'marker_ids':[1],'reason':'Points are on the rail instead of the nut.'}]},False),
    ({'all_instances_covered':False},False),
    ({'reviewed_marker_ids':[1]},False),
    ({'reviewed_marker_ids':[1,True]},False),
    ({'fai':2},False),
    ({'decision':'AUTO_MARK'},False),
])
def test_independent_entity_review_requires_complete_native_confirmation(tmp_path,change,confirmed):
    mesh={'source_sha256':'source','faces':[{'id':'plane','positions':[
        0,0,0,10,0,0,10,10,0,0,10,0],'indices':[0,1,2,0,2,3]}]}
    anchors=[{'face_id':'plane','position':[2,2,0]},
             {'face_id':'plane','position':[3,3,0]}]
    before=json.dumps(anchors)
    locations=[{'feature':'NUT','view_id':'+Z'}]*2
    class Model:
        def _generate_once(self,messages,**kwargs):
            attachment=messages[0]['attachments'][0]
            assert attachment['images'][0]=='original-msop.png'
            assert 'all_instances_covered' in messages[0]['content']
            content=json.loads(attachment['parsed_text'])
            assert 'raw_output' not in content['drawing_manifest']
            result={'decision':'CONFIRM_ENTITY_GROUP','fai':1,'reviewed_marker_ids':[1,2],
                'all_entities_match':True,'all_instances_covered':True,'issues':[],
                'feature_assignments':[dict(marker_id=i,source_names=[],instance_path=[],evidence='Source weld leader') for i in [1,2]],
                'evidence':'Both locations match the nut and its drawing leader.'}
            return json.dumps({**result,**change}),False,80
    audit=_review_visual_entity_group(SimpleNamespace(_dir=lambda _:tmp_path),{'id':'p'},
        {'number':1},{'methods':[{'page':1,'text':'Measure nut welds.'}]},Model(),mesh,
        anchors,locations,['original-msop.png'],drawing_manifest={'raw_output':'Prior rationale is not evidence'})
    assert (audit['status']=='confirmed') is confirmed
    assert json.dumps(anchors)==before
    assert not audit['human_selected_or_moved_points']
    assert hashlib.sha256(__import__('pathlib').Path(audit['receipt']).read_bytes()).hexdigest()==audit['receipt_sha256']


def test_entity_review_refuses_truncated_confirmation(tmp_path):
    mesh={'source_sha256':'source','faces':[{'id':'plane','positions':[
        0,0,0,10,0,0,0,10,0],'indices':[0,1,2]}]}
    class Model:
        def _generate_once(self,*args,**kwargs):
            return json.dumps({'decision':'CONFIRM_ENTITY_GROUP','fai':1,'reviewed_marker_ids':[1],
                'all_entities_match':True,'all_instances_covered':True,'issues':[],'evidence':'matches'}),True,90
    audit=_review_visual_entity_group(SimpleNamespace(_dir=lambda _:tmp_path),{'id':'p'},
        {'number':1},{},Model(),mesh,[{'face_id':'plane','position':[1,1,0]}],
        [{'feature':'nut'}],[])
    assert audit['status']=='rejected_or_unconfirmed'
