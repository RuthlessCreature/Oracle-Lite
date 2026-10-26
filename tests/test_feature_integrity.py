import copy
from oracle_viewer.feature_integrity import feature_links, consistency_conflicts, VERSION
from oracle_viewer.store import ViewerStore


def example(name='PIN_CIR',instance=None,position=None,program='UP'):
    instance=[1,2] if instance is None else instance
    anchor={'face_id':'face','position':position or [1,2,3]}
    mesh={'faces':[{'id':'face','instance_path':instance}]}
    detail={'methods':[{'page':3,'program':program,'measured_system':'ALI_A',
                        'text':f'Measure 1 circle {name}.'}]}
    item={'document_id':'doc','number':7}
    locations=[{'feature':name}]
    assignments=[dict(marker_id=1,source_names=[name],instance_path=instance,evidence='Printed circle leader matches this body.')]
    return item,detail,[anchor],locations,assignments,mesh


def verified_project(links):
    return {'model':{'source_sha256':'cad'},'annotations':{'doc:7':dict(
        status='auto_marked',source_sha256='cad',feature_verification=dict(
            integrity_version=VERSION,status='confirmed',feature_links=links))}}


def test_shared_named_circle_rejects_wrong_instance_and_far_anchor_without_moving_it():
    links,errors=feature_links(*example());assert not errors
    p=verified_project(links)
    args=example(instance=[1,4],position=[40,2,3]);before=copy.deepcopy(args)
    proposed,errors=feature_links(*args);assert not errors
    conflicts=consistency_conflicts(p,8,proposed)
    assert conflicts[0]['different_instance'] and conflicts[0]['distance_mm']==39
    assert args==before
    proposed,errors=feature_links(*example(position=[4,2,3]));assert not errors
    assert consistency_conflicts(p,8,proposed)[0]['different_instance'] is False


def test_generic_names_are_not_invented_source_identities_and_model_cannot_rename_a_circle():
    args=list(example());args[4]=[dict(marker_id=1,source_names=['OTHER_CIR'],instance_path=[1,2],evidence='guess')]
    assert feature_links(*args)[1]
    args[1]['methods'][0]['text']+=' OTHER_CIR'
    assert 'review_cannot_rename' in feature_links(*args)[1][0]
    args[4]=None
    assert feature_links(*args)[1]==['every_marker_needs_source_entity_assignment']


def test_same_name_in_different_program_or_alignment_is_not_merged():
    links,_=feature_links(*example());p=verified_project(links)
    other,_=feature_links(*example(program='DOWN',position=[40,2,3]))
    assert not consistency_conflicts(p,8,other)
    args=list(example(position=[40,2,3]));args[1]['methods'][0]['measured_system']='ALI_B'
    other,_=feature_links(*args)
    assert not consistency_conflicts(p,8,other)


def test_line_samples_can_move_on_same_entity_but_cannot_change_instance():
    links,_=feature_links(*example(name='EDGE_LIN'));p=verified_project(links)
    proposed,_=feature_links(*example(name='EDGE_LIN',position=[40,2,3]))
    assert not consistency_conflicts(p,8,proposed)
    proposed,_=feature_links(*example(name='EDGE_LIN',instance=[1,4]))
    assert consistency_conflicts(p,8,proposed)


def test_stale_or_different_cad_peers_do_not_authorize_or_veto_new_locations():
    links,_=feature_links(*example());p=verified_project(links)
    proposed,_=feature_links(*example(position=[40,2,3]))
    p['annotations']['doc:7']['status']='stale'
    assert not consistency_conflicts(p,8,proposed)
    p['annotations']['doc:7'].update(status='auto_marked',source_sha256='other')
    assert not consistency_conflicts(p,8,proposed)


def test_upgrade_quarantines_unverified_marks_preserving_all_coordinates(tmp_path):
    store=ViewerStore(tmp_path);p=store.create('generic','any-drawing','A')
    annotation=dict(origin='local_model_candidate',status='auto_marked',anchors=[dict(face_id='face',position=[1,2,3])])
    p['annotations']={'doc:1':copy.deepcopy(annotation),'doc:2':{**annotation,'origin':'manual'},
        'doc:3':{**annotation,'feature_verification':dict(integrity_version=VERSION,status='confirmed')}}
    store._write(p)
    assert store.require_current_feature_integrity(p['id'])==['doc:1']
    result=store.get(p['id'])
    assert result['annotations']['doc:1']['anchors']==annotation['anchors']
    assert result['annotations']['doc:1']['stale_reason']=='feature_identity_review_required'
    assert result['annotations']['doc:2']['status']=='auto_marked'
    assert not store.require_current_feature_integrity(p['id'])


def test_conflicting_peer_is_quarantined_instead_of_becoming_a_preferred_answer(tmp_path):
    store=ViewerStore(tmp_path);p=store.create('generic','different-project','B')
    p['annotations']={'doc:7':dict(status='auto_marked',anchors=[dict(face_id='face',position=[1,2,3])])};store._write(p)
    store.quarantine_feature_conflicts(p['id'],[dict(peer_fai_key='doc:7',distance_mm=39)])
    annotation=store.get(p['id'])['annotations']['doc:7']
    assert annotation['status']=='stale' and annotation['anchors'][0]['position']==[1,2,3]


def test_expanded_source_names_and_composite_frame_remain_source_scoped():
    args=list(example(name='PART_P02'))
    args[1]={'methods':[dict(page=3,text='Output system ALI_A & ALI_B\nMeasure 2 vector points PART_P01~P02.')]}
    links,errors=feature_links(*args)
    assert not errors and links[0]['source_name']=='PART_P02'
    assert links[0]['coordinate_system']=='ALI_A&ALI_B'
