from oracle_viewer.point_names import expand_point_clause
from oracle_viewer.msop_coordinates import _explicit_measure_point_names
from oracle_viewer.local_matching import _declared_marker_manifest


def test_expands_printed_shared_prefix_ranges_and_identifier_spacing_without_xyz_inference():
    assert expand_point_clause('PART_ S01~S03, S05, S07~S08.') is None
    assert expand_point_clause('PART_ S01~S03, S05, S07~S08')==[
        'PART_S01','PART_S02','PART_S03','PART_S05','PART_S07','PART_S08']
    assert expand_point_clause('F1_PNT_Z1/Z2, F2_PNT_Z1/Z2')==[
        'F1_PNT_Z1','F1_PNT_Z2','F2_PNT_Z1','F2_PNT_Z2']
    assert expand_point_clause('PART_S03~S01') is None
    assert expand_point_clause('PART_S01~OTHER_S03') is None


def test_complete_named_groups_on_two_surfaces_are_retained_as_distinct_source_names():
    methods=[{'page':1,'text':'Measure 4 vector points TOP_U01~U03,U05 as Plane View shows. '
        'Measure 4 vector points METAL_ S01~S03, S05.'}]
    names=['TOP_U01','TOP_U02','TOP_U03','TOP_U05','METAL_S01','METAL_S02','METAL_S03','METAL_S05']
    points={name:{'name':name,'occurrences':[]} for name in names}
    assert _explicit_measure_point_names(methods,points)==set(names)
    manifest=_declared_marker_manifest(methods)
    assert [g['declared_count'] for g in manifest]==[4,4]
    assert {n for g in manifest for n in g['source_point_names']}==set(names)
