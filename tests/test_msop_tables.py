from oracle_viewer.msop_tables import feature_coordinate_rows, table_point_manifest
from oracle_viewer.candidate_mining import extract_named_points
from oracle_viewer.msop_coordinates import _explicit_measure_point_names
from oracle_viewer.local_matching import _declared_marker_manifest
from oracle_viewer.automatic_candidates import _required_marker_count


def page(text):
    return dict(page=7, text=text, fai_ids=[3], program=None, measured_system=None)


def test_feature_xyz_table_keeps_output_scope_and_literal_point_names():
    p=page('Output system ALI_ABC\nMeasure 2 points and construct 1 basic set.\n'
           'Features X Y Z\nPART_P01 1.20 -2 0\nPART_P02 1.40 -2 -0.03\n')
    points=extract_named_points([p])
    assert set(points)=={'PART_P01','PART_P02'}
    row=points['PART_P02']['occurrences'][0]
    assert row['xyz']==[1.4,-2,-0.03]
    assert row['coordinate_system']=='ALI_ABC'
    assert row['program'] is None
    assert row['coordinate_frame_source']=='explicit_output_system_table'
    assert _explicit_measure_point_names([p],points)==set(points)
    assert _declared_marker_manifest([p])[0]['source_point_names']==list(points)
    assert _required_marker_count([p],points)==2


def test_two_column_xy_table_does_not_manufacture_z():
    p=page('Output system ALI_ABC\nMeasure 4 points.\nFeatures X Y\n'
           'TOP_P01 1 2 BOT_P01 1 2\nTOP_P02 3 4 BOT_P02 3 4\n')
    rows=feature_coordinate_rows(p)
    assert len(rows)==4
    assert all(row['axes']==['X','Y'] for row in rows)
    assert extract_named_points([p])=={}
    assert table_point_manifest(p)['declared_count']==4
    assert _required_marker_count([p])==4


def test_incomplete_or_conflicting_table_does_not_define_a_complete_manifest():
    p=page('Measure 3 points.\nFeatures X Y Z\nPART_P01 1 2 3\nPART_P02 4 5 6\n')
    assert table_point_manifest(p) is None
    p['text']+='PART_P01 9 2 3\n'
    assert table_point_manifest(p) is None
    assert extract_named_points([p])['PART_P01']['coordinate_conflict']


def test_table_header_required_and_numeric_column_count_exact():
    assert feature_coordinate_rows(page('PART_P01 1 2 3'))==[]
    p=page('Features X Y\nPART_P01 1 2 3\nPART_P02 1e-3 -.2\n')
    assert [r['name'] for r in feature_coordinate_rows(p)]==['PART_P02']


def test_output_frame_expression_is_not_truncated_or_chosen_from_conflicting_labels():
    p=page('Output system ALI_FIRST & ALI_SECOND\nFeatures X Y Z\nPART_P01 1 2 3\n')
    assert feature_coordinate_rows(p)[0]['coordinate_system']=='ALI_FIRST&ALI_SECOND'
    p['text']='Output system ALI_FIRST\nOutput system ALI_SECOND\nFeatures X Y Z\nPART_P01 1 2 3\n'
    assert feature_coordinate_rows(p)[0]['coordinate_system'] is None


def test_scan_and_line_range_tables_preserve_literal_limits_and_names():
    from oracle_viewer.msop_tables import table_measurement_manifest
    p=page('FAI 47\nMeasure 2 top lines.\nFeature Start point End point Direction\n'
           'FAI47_LIN 50.88 49.88 Y-axis\nFAI50_LIN2 -2 3 X-axis\n')
    manifest=table_measurement_manifest(p)
    assert manifest['source_point_names']==['FAI47_LIN','FAI50_LIN2']
    assert manifest['feature_ranges'][0]['start']=='50.88'
    assert manifest['feature_ranges'][0]['direction_axis']=='Y'
    assert extract_named_points([p])=={}
    assert _declared_marker_manifest([p])[0]['declared_count']==2
    assert _required_marker_count([p])==2
    p['text']=p['text'].replace('Measure 2 top lines','Measure 3 top lines')
    assert table_measurement_manifest(p) is None


def test_pdf_side_by_side_headers_and_drawing_label_do_not_hide_feature_tables():
    from oracle_viewer.msop_tables import table_measurement_manifest
    p=page('Measure 4 points.\nFeatures X Y    Features X Y\n'
           'PART_P01 1 2 PART_P03 5 6\nPART_P02 3 4 PART_P04 7 8\n')
    assert len(feature_coordinate_rows(p))==4
    assert table_measurement_manifest(p)['declared_count']==4
    p=page('Measure 2 top lines.\n B     Feature Start point End point Direction\n'
           'PART_LIN1 1 2 X-axis\nPART_LIN2 3 4 Y-axis\n')
    assert table_measurement_manifest(p)['declared_count']==2
