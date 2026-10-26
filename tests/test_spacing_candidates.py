import numpy as np

from oracle_viewer.spacing_candidates import evenly_spaced_candidates


def plane_mesh():
    return {'source_sha256':'spacing-test','faces':[{'id':'plane','surface':'plane',
        'positions':[0,0,0,30,0,0,30,30,0,0,30,0],'indices':[0,1,2,0,2,3]}]}


def test_line_hypothesis_has_complete_equal_spacing_and_never_changes_input():
    anchors=[{'face_id':'plane','position':[x,10,0]} for x in [0,1,2,2.1,20]]
    before=[row['position'][:] for row in anchors]
    choices=evenly_spaced_candidates(plane_mesh(),anchors,'+Z',5)
    assert choices and choices[0]['pattern']=='LINE'
    assert np.allclose([row['position'][0] for row in choices[0]['anchors']],[0,5,10,15,20])
    assert all(row['requires_local_model_selection'] for row in choices)
    assert [row['position'] for row in anchors]==before


def test_circle_hypothesis_is_available_without_assuming_line_layout():
    anchors=[{'face_id':'plane','position':[15+4*np.cos(t),15+4*np.sin(t),0]}
        for t in [0,.1,1.5,2,4,5.5]]
    choices=evenly_spaced_candidates(plane_mesh(),anchors,'+Z',6)
    circle=next(row for row in choices if row['pattern']=='CIRCLE')
    assert len(circle['anchors'])==6 and circle['nearest_neighbor_ratio']<1.001


def test_even_spacing_hypothesis_cannot_fill_voids_or_hidden_surfaces():
    mesh=plane_mesh()
    mesh['faces']=[{'id':str(i),'surface':'plane','positions':[
        x,0,0,x+5,0,0,x+5,30,0,x,30,0],'indices':[0,1,2,0,2,3]}
        for i,x in enumerate([0,15])]
    anchors=[{'face_id':'0' if x<=5 else '1','position':[x,10,0]} for x in [0,2,4,16,20]]
    assert evenly_spaced_candidates(mesh,anchors,'+Z',5)==[]
    mesh=plane_mesh();cover={**mesh['faces'][0],'id':'cover',
        'positions':[v+1 if i%3==2 else v for i,v in enumerate(mesh['faces'][0]['positions'])]}
    mesh['faces'].append(cover)
    anchors=[{'face_id':'plane','position':[x,10,0]} for x in [0,1,2,2.1,20]]
    assert evenly_spaced_candidates(mesh,anchors,'+Z',5)==[]
