from oracle_viewer.candidate_mining import (
    extract_named_points,
    extract_feature_labels,
    referenced_points,
    nearest_xy_triangle,
    project_points_to_common_face,
    screen_point_references,
    semantic_point_references,
    resolve_dominant_source_page_conflicts,
)
from oracle_viewer.msop_coordinates import _coordinate_rows, _explicit_measure_point_names


def test_two_frames_on_one_page_follow_explicit_point_clauses():
    page=dict(page=1,program='UP',measured_system='ALI_A',text='''
    Measure 2 vector points DATUM_E01, E02 under ALI_A.
    Measure 2 vector points EDGE_P01, P02 under alignment ALI_B.
    X, Y, Z (mm) 1 2 3 DATUM_E01
    X, Y, Z (mm) 2 2 3 DATUM_E02
    X, Y, Z (mm) 4 5 6 EDGE_P01
    X, Y, Z (mm) 5 5 6 EDGE_P02 I, J, K 0 1 0
    ''')
    named=extract_named_points([page])
    assert named['DATUM_E02']['occurrences'][0]['coordinate_system']=='ALI_A'
    assert named['EDGE_P01']['occurrences'][0]['coordinate_system']=='ALI_B'
    rows=_coordinate_rows([page],[],[],[page],named)
    edge=next(r for r in rows if r['name']=='EDGE_P02')
    assert edge['coordinate_system']=='ALI_B' and edge['ijk']==[0,1,0]


def test_mixed_measured_and_dimension_systems_scope_continuation_xyz_by_point():
    method = {
        'page': 45, 'heading': 'FAI 35', 'fai_ids': [35], 'program': 'UP',
        'measured_system': 'ALI_ABD', 'dimension_system': 'ALI_ABC',
        'text': '''Measured System ALI_ABD • Measure 2 vector points N2_PNT1 ~ N2_PNT2 and construct 1 plane N2_PLN1.
Dimension System ALI_ABC • Measure 2 vector points N3_PNT1 ~ N3_PNT2 and construct 1 plane N3_PLN1.
• Measure 2 vector points N4_PNT1 ~ N4_PNT2 and construct 1 plane N4_PLN1.''',
    }
    location = {
        'page': 46, 'heading': 'FAI 35', 'fai_ids': [35], 'text': '''Location Definitions
X, Y, Z (mm) 1 2 3 N2_PNT1
X, Y, Z (mm) 2 2 3 N2_PNT2
X, Y, Z (mm) 3 2 3 N3_PNT1
X, Y, Z (mm) 4 2 3 N3_PNT2
X, Y, Z (mm) 5 2 3 N4_PNT1
X, Y, Z (mm) 6 2 3 N4_PNT2''',
    }
    named = extract_named_points([method, location])
    assert {named[f'N2_PNT{i}']['occurrences'][0]['coordinate_system'] for i in (1, 2)} == {'ALI_ABD'}
    assert {named[f'N3_PNT{i}']['occurrences'][0]['coordinate_system'] for i in (1, 2)} == {'ALI_ABC'}
    assert {named[f'N4_PNT{i}']['occurrences'][0]['coordinate_system'] for i in (1, 2)} == {'ALI_ABC'}
    assert {r['feature'] for r in named['N2_PNT1']['constructed_point_sets']} == {'N2_PLN1'}
    assert {r['feature'] for r in named['N3_PNT1']['constructed_point_sets']} == {'N3_PLN1'}
    assert {r['feature'] for r in named['N4_PNT1']['constructed_point_sets']} == {'N4_PLN1'}


def test_point_named_in_two_system_blocks_remains_unresolved():
    page = {'page': 1, 'program': 'UP', 'text': '''
Measured System ALI_A • Measure 1 vector point POINT_P01.
Dimension System ALI_B • Measure 1 vector point POINT_P01.
X, Y, Z (mm) 1 2 3 POINT_P01'''}
    row = extract_named_points([page])['POINT_P01']['occurrences'][0]
    assert row['coordinate_system'] is None
    assert row['frame_assignment_conflict'] is True


def test_composite_frame_is_passed_to_model_as_explicit_alternatives():
    from oracle_viewer.automatic_candidates import _source_point_evidence
    page = {'page': 57, 'heading': 'FAI 43', 'fai_ids': [43], 'program': 'UP',
        'measured_system': 'ALI_ABC_A&ALI_ABD', 'text': '''
Measured System ALI_ABC_A& ALI_ABD • Measure 1 vector point DATUMC_PNT_P01.
Location Definitions
X, Y, Z (mm) 1 2 3 DATUMC_PNT_P01'''}
    named=extract_named_points([page])
    occurrence=named['DATUMC_PNT_P01']['occurrences'][0]
    assert occurrence['coordinate_system'] is None
    assert occurrence['coordinate_system_candidates']==['ALI_ABC_A','ALI_ABD']
    assert occurrence['frame_assignment_conflict'] is True
    point=_source_point_evidence(_coordinate_rows([page],[],[],[page],named))[0]
    assert point['xyz'] is None and point['requires_variant_choice'] is True
    assert {variant['coordinate_system'] for variant in point['coordinate_variants']} == {'ALI_ABC_A','ALI_ABD'}


def test_explicit_measure_list_excludes_unrelated_xyz_definitions_on_same_page():
    pages = [
        {'page': 1, 'program': 'DOWN', 'measured_system': 'ALI_ABC_A',
         'text': 'Measure 4 vector points DATUMC_E02,E03,E42,E43 and construct a line DATUMC_TOP.\n'
                 'Location Definitions\n'
                 'X, Y, Z (mm) 9.34 72.13 0.08 DATUMC_E02\n'
                 'X, Y, Z (mm) 18.68 72.13 0.08 DATUMC_E03\n'
                 'X, Y, Z (mm) -18.68 72.13 0.08 DATUMC_E42\n'
                 'X, Y, Z (mm) -9.34 72.13 0.08 DATUMC_E43\n'
                 'X, Y, Z (mm) 12.53 -72.30 0.53 DATUMC_E21'},
    ]
    named = extract_named_points(pages)
    assert _explicit_measure_point_names(pages, named) == {
        'DATUMC_E02', 'DATUMC_E03', 'DATUMC_E42', 'DATUMC_E43'}
    rows = _coordinate_rows(pages, [], [], pages, named)
    assert {row['name'] for row in rows} == {
        'DATUMC_E02', 'DATUMC_E03', 'DATUMC_E42', 'DATUMC_E43'}


def test_unnamed_plane_view_point_group_uses_only_exact_count_same_page_xyz_rows():
    text='Measure 3 vector points as Plane View show.\nLocation Definitions\n'
    text+='\n'.join(f'X, Y, Z (mm) {i} 2 1 EDGE_PNT{i}' for i in range(1,4))
    page=dict(page=12,program='UP',measured_system='ALI_A',text=text)
    named=extract_named_points([page])
    assert _explicit_measure_point_names([page],named)=={'EDGE_PNT1','EDGE_PNT2','EDGE_PNT3'}
    rows=_coordinate_rows([page],[],['EDGE_PNT1'],[page],named)
    assert {row['name'] for row in rows}=={'EDGE_PNT1','EDGE_PNT2','EDGE_PNT3'}
    page['text']+='\nX, Y, Z (mm) 4 2 1 AUX_PNT4'
    named=extract_named_points([page])
    assert _explicit_measure_point_names([page],named) is None


def test_repeated_plane_view_method_page_reuses_unique_same_frame_xyz_group():
    method = 'Measure 3 vector points as Plane View show.'
    coordinate = ('Location Definitions\n' + '\n'.join(
        f'X, Y, Z (mm) {i} 2 1 EDGE_PNT{i}' for i in range(1, 4)))
    pages = [
        dict(page=20, program='UP', measured_system='ALI_A', text=method),
        dict(page=21, program='UP', measured_system='ALI_A', text=method + '\n' + coordinate),
    ]
    named = extract_named_points(pages)

    expected = {f'EDGE_PNT{i}' for i in range(1, 4)}
    assert _explicit_measure_point_names(pages, named) == expected
    rows = _coordinate_rows(pages, [], [], pages, named)
    assert {row['name'] for row in rows} == expected


def test_conflicting_explicit_frame_clauses_are_not_first_match_wins():
    page=dict(page=1,program='UP',measured_system='ALI_A',text='''
    Measure POINT_P01 under ALI_A. Measure POINT_P01 under ALI_B.
    X, Y, Z (mm) 1 2 3 POINT_P01''')
    row=_coordinate_rows([page],[],[],[page])[0]
    assert row['coordinate_conflict'] and row['coordinate_system'] is None


def test_dominant_complete_coordinate_page_resolves_duplicate_labels_with_provenance():
    rows=[]
    for name,x,old_xyz in [('RAIL_PNT_X1',1,[1,1,2]),('RAIL_PNT_X2',2,[2,1,2])]:
        rows.append(dict(name=name,point_key=name,xyz=old_xyz,pages=[2,10],program='UP',
            coordinate_system='ALI_ABD',coordinate_conflict=True,scope_ambiguous=False,
            occurrences=[dict(page=2,xyz=old_xyz,program='UP',coordinate_system='ALI_ABD'),
                         dict(page=10,xyz=[x,2,1.5],program='UP',coordinate_system='ALI_ABD')]))
    for name,x in [('RAIL_PNT_X3',3),('RAIL_PNT_X4',4)]:
        rows.append(dict(name=name,point_key=name,xyz=[x,2,1.5],pages=[10],program='UP',
            coordinate_system='ALI_ABD',coordinate_conflict=False,scope_ambiguous=False,
            occurrences=[dict(page=10,xyz=[x,2,1.5],program='UP',coordinate_system='ALI_ABD')]))
    resolved,audit=resolve_dominant_source_page_conflicts(rows)
    assert audit==[dict(selected_page=10,program='UP',coordinate_system='ALI_ABD',
        covered_point_count=4,source_point_count=4,page_group_coverage=1.0,
        resolved_point_names=['RAIL_PNT_X1','RAIL_PNT_X2'],source_variants_preserved=True)]
    assert all(not row['coordinate_conflict'] for row in resolved)
    assert resolved[0]['xyz']==[1,2,1.5]
    assert len(resolved[0]['source_coordinate_variants'])==2
    assert rows[0]['coordinate_conflict'] is True


def test_coordinate_page_conflict_resolver_rejects_partial_or_tied_pages():
    rows=[dict(name=f'PNT_{i}',point_key=f'p{i}',xyz=[i,0,0],pages=[1,2],program='UP',
        coordinate_system='FRAME',coordinate_conflict=True,occurrences=[
            dict(page=1,xyz=[i,0,0],program='UP',coordinate_system='FRAME'),
            dict(page=2,xyz=[i,0,0],program='UP',coordinate_system='FRAME')]) for i in range(2)]
    resolved,audit=resolve_dominant_source_page_conflicts(rows)
    assert not audit and all(row['coordinate_conflict'] for row in resolved)


def test_same_point_name_in_flip_programs_resolves_to_requested_program():
    pages = [
        {'page': 1, 'program': 'UP', 'measured_system': 'ALI_ABD', 'text': 'X, Y, Z (mm) 1 2 3 PNT_A'},
        {'page': 2, 'program': 'DOWN', 'measured_system': 'ALI_ABD', 'text': 'X, Y, Z (mm) -1 2 -3 PNT_A'},
        {'page': 3, 'program': 'DOWN', 'measured_system': 'ALI_ABD', 'text': 'Measure PNT_A'},
    ]
    named = extract_named_points(pages)
    assert not named['PNT_A']['coordinate_conflict']
    rows = _coordinate_rows([pages[2]], [], ['PNT_A'], pages)
    assert len(rows) == 1 and rows[0]['xyz'] == [-1, 2, -3]
    assert rows[0]['program'] == 'DOWN' and rows[0]['pages'] == [2]


def test_contextless_reference_cannot_select_first_of_multiple_scopes():
    pages = [
        {'page': 1, 'program': 'UP', 'measured_system': 'ALI_ABD', 'text': 'X, Y, Z (mm) 1 2 3 PNT_A'},
        {'page': 2, 'program': 'DOWN', 'measured_system': 'ALI_ABD', 'text': 'X, Y, Z (mm) -1 2 -3 PNT_A'},
        {'page': 3, 'text': 'Measure PNT_A'},
    ]
    row = _coordinate_rows([pages[2]], [], ['PNT_A'], pages)[0]
    assert row['scope_ambiguous'] and row['coordinate_conflict']
    assert len(row['unresolved_scopes']) == 2


def test_local_definition_does_not_hide_conflicting_same_scope_occurrence():
    pages = [
        {'page': 1, 'program': 'UP', 'measured_system': 'ALI_ABD', 'text': 'X, Y, Z (mm) 1 2 3 PNT_A'},
        {'page': 2, 'program': 'UP', 'measured_system': 'ALI_ABD', 'text': 'X, Y, Z (mm) 4 2 3 PNT_A'},
    ]
    row = _coordinate_rows([pages[1]], [], [], pages)[0]
    assert row['coordinate_conflict'] and row['pages'] == [1, 2]


def test_method_page_location_definition_resolves_repeated_same_scope_point():
    pages = [
        {'page': 1, 'heading': 'FAI 1', 'program': 'UP', 'measured_system': 'ALI_ABD',
         'text': 'Measure 3 vector points PNT_A, PNT_B, PNT_C.\n'
                 'Location Definitions\nX, Y, Z (mm) 1 2 3 PNT_A'},
        {'page': 2, 'program': 'UP', 'measured_system': 'ALI_ABD',
         'text': 'X, Y, Z (mm) 1.02 2 3 PNT_A'},
    ]
    row = _coordinate_rows([pages[0]], [], ['PNT_A'], pages)[0]
    assert row['xyz'] == [1.0, 2.0, 3.0]
    assert row['pages'] == [1]
    assert not row['coordinate_conflict']


def test_repeated_coordinates_within_printed_precision_keep_all_provenance():
    pages = [
        {'page': 1, 'program': 'UP', 'measured_system': 'ALI_ABD',
         'text': 'X, Y, Z (mm) -12.47 62.09 1.03 N1_PNT1'},
        {'page': 2, 'program': 'UP', 'measured_system': 'ALI_ABD',
         'text': 'X, Y, Z (mm) -12.48 62.09 1.03 N1_PNT1'},
        {'page': 3, 'program': 'UP', 'measured_system': 'ALI_ABD',
         'text': 'Measure N1_PNT1.'},
    ]
    row = _coordinate_rows([pages[2]], [], ['N1_PNT1'], pages)[0]
    assert not row['coordinate_conflict']
    assert row['coordinate_precision_equivalent']
    assert abs(row['coordinate_variant_span_mm'] - 0.01) < 1e-12
    assert len(row['occurrences']) == 2


def test_coarse_print_precision_cannot_be_used_as_a_location_tolerance():
    pages = [
        {'page': 1, 'program': 'UP', 'measured_system': 'ALI_ABD',
         'text': 'X, Y, Z (mm) 1 2 3 PNT_A'},
        {'page': 2, 'program': 'UP', 'measured_system': 'ALI_ABD',
         'text': 'X, Y, Z (mm) 1.5 2 3 PNT_A'},
        {'page': 3, 'program': 'UP', 'measured_system': 'ALI_ABD',
         'text': 'Measure PNT_A.'},
    ]
    row = _coordinate_rows([pages[2]], [], ['PNT_A'], pages)[0]
    assert row['coordinate_conflict']


def test_continuation_vectors_preserve_program_and_alignment():
    pages = [
        {'page': 1, 'heading': 'DATUM A', 'program': 'DOWN', 'measured_system': 'ALI_ABD', 'text': 'Measure PNT_A'},
        {'page': 2, 'heading': 'DATUM A', 'text': 'X, Y, Z (mm) 1 2 3 PNT_A I, J, K 0 0 -1'},
        {'page': 3, 'heading': 'DATUM B', 'text': 'X, Y, Z (mm) 4 5 6 PNT_B'},
    ]
    rows = _coordinate_rows([pages[0]], [], ['PNT_A'], pages)
    assert rows[0]['program'] == 'DOWN' and rows[0]['ijk'] == [0, 0, -1]
    assert extract_named_points(pages)['PNT_B']['occurrences'][0]['program'] is None


def test_numeric_ranges_and_abbreviations_expand_only_existing_family_points():
    names = ['E04_OUT', 'E05_OUT', 'E06_OUT', 'E07_OUT', 'E05_IN',
             'W04_PNT_Z1', 'W04_PNT_Z2', 'W04_PNT_Z3', 'DATUMB_E09', 'DATUMB_E10', 'DATUMB_E12']
    pages = [{'page': 1, 'program': 'UP', 'measured_system': 'ALI_ABD',
              'text': '\n'.join(f'X, Y, Z (mm) 1 2 3 {name}' for name in names)}]
    method = [{'page': 2, 'program': 'UP', 'measured_system': 'ALI_ABD',
               'text': 'Measure E04_OUT~E07_OUT. Measure W04_PNT_Z1,Z3. Measure DATUMB_E09,E10,E12.'}]
    refs = referenced_points(pages, method, extract_named_points(pages))
    assert set(refs) == {'E04_OUT', 'E05_OUT', 'E06_OUT', 'E07_OUT', 'W04_PNT_Z1', 'W04_PNT_Z3',
                         'DATUMB_E09', 'DATUMB_E10', 'DATUMB_E12'}


def test_measurement_ranges_resolve_short_labels_to_full_pnt_table_names():
    names = [f'DATUMB_PNT_P{n:02d}' for n in (*range(9, 18), *range(33, 42))]
    points = '\n'.join(f'X, Y, Z (mm) {i} 2 0 {name}' for i, name in enumerate(names))
    names += ['F1_CIR_L', 'F1_CIR_R']
    points += '\nX, Y, Z (mm) 90 2 0 F2_PNT_R2'
    points += '\nX, Y, Z (mm) -6.39 4.62 2.11 F1_CIR_L\nX, Y, Z (mm) 6.39 4.62 2.11 F1_CIR_R'
    page = {'page': 47, 'program': 'UP', 'measured_system': 'ALI_ABC',
            'text': 'Measure 2 vector circles F1_CIR_L& F1_CIR_R, use top circle.\n'
                    'Measure 9 vector points DATUMB_P09~DATUMB_P17 and construct a line DATUMB_LEFT.\n'
                    'Measure 9 vector points DATUMB_P33~DATUMB_P41 and construct a line DATUMB_RIGHT.\n'
                    + points}
    named = extract_named_points([page])
    expected = set(names)
    assert _explicit_measure_point_names([page], named) == expected
    rows = _coordinate_rows([page], [], [], [page], named)
    assert {row['name'] for row in rows} == expected
    assert 'F2_PNT_R2' not in {row['name'] for row in rows}


def test_method_model_receives_distinct_program_points_and_their_own_mappings():
    import json
    from threading import RLock
    from oracle_viewer.channel_routes import propose
    class Model:
        _lock = RLock()
        def _generate_once(self, messages, **kwargs):
            self.evidence = json.loads(messages[0]['attachments'][0]['parsed_text'])
            return json.dumps({'routes':[{'number':1,'channel':'method_review_required',
                                         'proposal':'检查程序归属','requirements':['原文点位待复核']}]}), False, 40
    model = Model()
    task = {'number':1,'dependencies':[],'anchors':[], 'msop_coordinate_map':{
        'source_points':[{'name':'PNT_A','point_key':'up','program':'UP','xyz':[1,2,3]},
                         {'name':'PNT_A','point_key':'down','program':'DOWN','xyz':[-1,2,-3]}],
        'mapped_points':[{'name':'PNT_A','point_key':'up','cad_position_mm':[10,20,30]},
                         {'name':'PNT_A','point_key':'down','cad_position_mm':[-10,20,-30]}]}}
    propose(model,[task],lambda *args:None)
    points = model.evidence['tasks'][0]['explicit_msop_coordinates']['points']
    assert points[0]['program']=='UP' and points[0]['cad_position_mm']==[10,20,30]
    assert points[1]['program']=='DOWN' and points[1]['cad_position_mm']==[-10,20,-30]


def test_alignment_reused_by_flip_programs_requires_program_pinned_calibration():
    from oracle_viewer.msop_coordinates import map_task_points
    pages=[{'page':1,'program':'UP','measured_system':'ALI_ABD','text':'X, Y, Z (mm) .2 .2 0 PNT_A'},
           {'page':2,'program':'DOWN','measured_system':'ALI_ABD','text':'X, Y, Z (mm) -.2 .2 0 PNT_A'}]
    mesh={'faces':[{'id':'f','positions':[0,0,2,1,0,2,0,1,2],'indices':[0,1,2]}]}
    task={'methods':[pages[0]],'anchors':[{'face_id':'f','position':[.2,.2,2]}]}
    calibration={'hypotheses':[{'name':'unscoped','systems':['ALI_ABD']}], 'max_xy_residual_mm':.1}
    rejected=map_task_points(task,mesh,calibration,pages)
    assert rejected['mapped_points']==[]
    calibration['hypotheses'][0]['program']='UP'
    mapped=map_task_points(task,mesh,calibration,pages)
    assert len(mapped['mapped_points'])==1 and mapped['mapped_points'][0]['program']=='UP'


def test_method_batches_split_after_repeated_duplicate_ids_without_truncation():
    import json
    from threading import RLock
    from oracle_viewer.channel_routes import propose
    class Model:
        _lock=RLock()
        seen=[]
        def _generate_once(self,messages,**kwargs):
            tasks=json.loads(messages[0]['attachments'][0]['parsed_text'])['tasks']
            self.seen.append([t['number'] for t in tasks])
            rows=[{'number':t['number'],'channel':'method_review_required','proposal':'保留原文',
                   'requirements':['基准待验证']} for t in tasks]
            if len(rows)>1:rows.append(rows[-1].copy())
            return json.dumps({'routes':rows}),False,60
    model=Model();progress=[]
    tasks=[{'number':n,'dependencies':[],'anchors':[],'features':[f'PNT_{n}']} for n in [41,42,43]]
    routes,attempts=propose(model,tasks,lambda *args:progress.append(args))
    assert [r['number'] for r in routes]==[41,42,43]
    assert model.seen[:3]==[[41,42,43]]*3 and model.seen[3:]==[[41],[42],[43]]
    assert len(attempts)==6 and len([a for a in attempts if a.get('validation_error')])==3
    assert progress[-1][0:2]==(3,3)


def test_named_point_index_preserves_cross_page_provenance():
    pages = [
        {"page": 4, "fai_ids": [8], "text": "X, Y, Z (mm) 1.0 2.0 3.0 N3_PNT1"},
        {"page": 5, "fai_ids": [9], "text": "X, Y, Z (mm) 4.0 5.0 6.0 N3_PNT2"},
    ]
    points = extract_named_points(pages)
    assert points["N3_PNT1"]["pages"] == [4]
    assert points["N3_PNT2"]["fai_ids"] == [9]


def test_semantic_feature_lookup_uses_specific_shared_token():
    points = {
        "N3_PNT1": {"name": "N3_PNT1"},
        "N2_PNT1": {"name": "N2_PNT1"},
        "DATUMC_PNT1": {"name": "DATUMC_PNT1"},
    }
    refs = semantic_point_references(["CIR_N3_VISION"], points)
    assert [p["name"] for p in refs["CIR_N3_VISION"]] == ["N3_PNT1"]
    assert "STRAP_CIR" not in semantic_point_references(
        ["STRAP_CIR"], {"F1_CIR_L": {"name": "F1_CIR_L"}})


def test_semantic_point_lookup_does_not_match_only_shared_ordinal():
    assert not semantic_point_references(
        ['R1_PNT_P3'], {'R3_PNT_P3': {'name': 'R3_PNT_P3'}})


def test_explicit_measurement_set_is_not_expanded_by_constructed_family_line():
    corpus=[dict(page=1,program='UP',measured_system='ALI_A',text='\n'.join(
        f'X, Y, Z (mm) {i} 0 0 EDGE_P{i:02d}' for i in range(1,7)))]
    method=[dict(page=2,program='UP',measured_system='ALI_A',
                 text='Measure EDGE_P01,P02,P03 and construct EDGE_LIN.')]
    refs=referenced_points(None,method,extract_named_points(corpus),['EDGE_LIN'])
    assert set(refs)=={'EDGE_P01','EDGE_P02','EDGE_P03'}


def test_vision_feature_without_explicit_point_set_keeps_cross_page_fallback():
    corpus=[dict(page=1,program='UP',measured_system='ALI_A',text='\n'.join(
        f'X, Y, Z (mm) {i} 0 0 N8_PNT{i}' for i in range(1,4)))]
    method=[dict(page=2,program='UP',measured_system='ALI_A',text='Measure CIR_N8_VISION.')]
    assert set(referenced_points(None,method,extract_named_points(corpus),['CIR_N8_VISION']))=={
        'N8_PNT1','N8_PNT2','N8_PNT3'}


def test_vision_feature_uses_msop_constructed_plane_point_group():
    corpus = [
        {'page': 1, 'program': 'UP', 'measured_system': 'ALI_A', 'text': '\n'.join(
            f'X, Y, Z (mm) {i} 0 0 N8_PNT{i}' for i in range(1, 9))},
        {'page': 2, 'program': 'UP', 'measured_system': 'ALI_A', 'text':
         'Measure 4 vector points N8_PNT1~PNT4 and construct a plane N8_PLN1.\n'
         'Measure 2 vector points N8_PNT5~N8_PNT6 and construct a line N8_ZLIN.'},
    ]
    method = [{'page': 3, 'program': 'UP', 'measured_system': 'ALI_A',
               'text': 'Measure 1 vision circle CIR_N8_VISION.'}]
    points = extract_named_points(corpus)
    refs = referenced_points(corpus, method, points, ['CIR_N8_VISION'])
    assert set(refs) == {'N8_PNT1', 'N8_PNT2', 'N8_PNT3', 'N8_PNT4'}
    assert all(row['semantic_source_pages'] == [1] for row in refs.values())


def test_sibling_fai_nominal_axes_select_the_matching_repeated_xyz_table():
    pages = []
    for page, y in [(1, 10), (2, 20)]:
        pages.append({'page': page, 'program': 'UP', 'measured_system': 'ALI_A',
            'text': '\n'.join(f'X, Y, Z (mm) {x} {y} 0 N8_PNT{i}'
                               for i, x in enumerate((1, 2, 3, 2), 1))})
    pages.extend([
        {'page': 3, 'program': 'UP', 'measured_system': 'ALI_A',
         'text': 'Measure 4 vector points N8_PNT1~PNT4 and construct a plane N8_PLN1.'},
        {'page': 4, 'fai_ids': [1], 'program': 'UP', 'measured_system': 'ALI_A',
         'features': ['CIR_N8_VISION'], 'specification': {'nominal': '20', 'upper': '.1', 'lower': '-.1'},
         'text': 'Measure 1 vision circle CIR_N8_VISION. Dimension Y location.'},
        {'page': 5, 'fai_ids': [2], 'program': 'UP', 'measured_system': 'ALI_A',
         'features': ['CIR_N8_VISION'], 'specification': {'nominal': '2', 'upper': '.1', 'lower': '-.1'},
         'text': 'Measure 1 vision circle CIR_N8_VISION. Dimension X location.'},
    ])
    rows = _coordinate_rows([pages[3]], [], ['CIR_N8_VISION'], pages)
    assert len(rows) == 4
    assert all(row['pages'] == [2] for row in rows)
    assert all(row['semantic_resolution']['basis'] ==
               'sibling FAI nominal coordinate and dimension axis' for row in rows)


def test_abbreviated_measurement_point_lists_resolve_from_corpus():
    corpus = [
        {"page": 4, "fai_ids": [8], "text": "X, Y, Z (mm) 1 2 3 DATUMC_PNT_P04"},
        {"page": 5, "fai_ids": [9], "text": "X, Y, Z (mm) 1 2 3 DATUMC_PNT_P23"},
    ]
    points = extract_named_points(corpus)
    method = [{"page": 10, "text": "Measure DATUMC_P04/03/47/46, P22/23/27/28"}]
    labels = extract_feature_labels(method)
    refs = referenced_points(corpus, method, points, labels)
    assert set(refs) == {"DATUMC_PNT_P04", "DATUMC_PNT_P23"}


def test_screening_applies_calibrated_xy_transform_and_keeps_inliers():
    vertices = [[-1.0, 2.0, 4.0], [-2.0, 3.0, 4.0], [80.0, 80.0, 0.0]]
    face_ids = ["face-a", "face-b", "outlier"]
    points = [
        {"name": "A", "xyz": [1.0, 2.0, 0.0], "pages": [1]},
        {"name": "B", "xyz": [2.0, 3.0, 0.0], "pages": [1]},
    ]
    result = screen_point_references(
        points, vertices, face_ids,
        {"x_sign": -1, "y_sign": 1, "x_offset_mm": 0, "y_offset_mm": 0},
        max_xy_residual_mm=0.01, min_inlier_ratio=1.0)
    assert result["valid"]
    assert [row["face_id"] for row in result["inliers"]] == ["face-a", "face-b"]


def test_triangle_projection_avoids_mesh_vertex_snap_offset():
    triangles = [[[0, 0, 2], [2, 0, 2], [0, 2, 2]]]
    distance, face_id, position = nearest_xy_triangle(triangles, ["face-a"], .5, .5)
    assert distance == 0
    assert face_id == "face-a"
    assert position == [.5, .5, 2]


def test_joint_plane_projection_keeps_points_on_one_best_supported_face():
    triangles = [
        [[0, 0, 2], [2, 0, 2], [0, 2, 2]],
        [[2, 0, 2], [2, 2, 2], [0, 2, 2]],
        [[0, 0, 3], [2, 0, 3], [0, 2, 3]],
        [[2, 0, 3], [2, 2, 3], [0, 2, 3]],
    ]
    points = [
        {"name": "P1", "mapped_xy": [.5, .5]},
        {"name": "P2", "mapped_xy": [1.5, .5]},
        {"name": "P3", "mapped_xy": [.5, 1.5]},
    ]
    result = project_points_to_common_face(
        points, triangles, ["lower", "lower", "upper", "upper"],
        ["lower", "upper"], max_xy_residual_mm=.01)
    assert result["face_id"] == "lower"
    assert result["coverage"] == 1
    assert result["ambiguous"]
    assert result["alternatives"][0]["face_id"] == "upper"
    assert result["z_span_mm"] == 0
    assert {row["face_id"] for row in result["inliers"]} == {"lower"}
    assert {row["position"][2] for row in result["inliers"]} == {2}


def test_fai_continuation_inherits_coordinate_page_context_and_dimension_frame():
    pages = [
        {'page': 88, 'heading': 'FAI 7 0.00 +0.15', 'fai_ids': [7], 'program': 'DOWN',
         'measured_system': 'ALI_ITER', 'dimension_system': 'ALI_ABD', 'text': 'Measure DATUMA_U01.'},
        {'page': 89, 'heading': 'FAI 7 0.00 +0.15 / FAI 7 0.00 ±0.075', 'fai_ids': [7],
         'text': 'X, Y, Z (mm) 0.00 70.14 0.00 DATUMA_U01'},
    ]
    row = _coordinate_rows([pages[0]], [], ['DATUMA_U01'], pages)[0]
    assert row['program'] == 'DOWN'
    assert row['coordinate_system'] == 'ALI_ABD'
    assert row['pages'] == [89]


def test_constructed_axis_line_uses_explicit_alignment_frame_for_shared_support():
    pages = [
        {'page': 68, 'fai_ids': [51], 'program': 'UP', 'measured_system': 'ALI_ABC_A',
         'text': 'Measure 2 vector points Y1_PNT_Y2, Y1_PNT_Y3 under alignment ALI_ABD and construct a line Y1_YLIN_BASE.\n'
                 'X, Y, Z (mm) -24.57 69.48 .4 Y1_PNT_Y2\n'
                 'X, Y, Z (mm) 27.17 69.48 .4 Y1_PNT_Y3'},
        {'page': 69, 'fai_ids': [52], 'program': 'UP', 'measured_system': 'ALI_ABC_A',
         'text': 'Measure 2 vector points Y1_PNT_Y1, Y1_PNT_Y4 under alignment ALI_ABD and construct a line Y1_YLIN_TIP.\n'
                 'X, Y, Z (mm) -24.57 69.48 2.83 Y1_PNT_Y1\n'
                 'X, Y, Z (mm) 27.17 69.48 2.83 Y1_PNT_Y4'},
    ]
    points = extract_named_points(pages)
    groups = {g['feature']: g for p in points.values() for g in p.get('constructed_point_sets', [])}
    assert groups['Y1_YLIN_BASE']['family'] == 'Y1'
    assert groups['Y1_YLIN_BASE']['coordinate_system'] == 'ALI_ABD'
    assert groups['Y1_YLIN_BASE']['coordinate_pages'] == [68]
    assert groups['Y1_YLIN_TIP']['coordinate_pages'] == [69]


def test_point_number_parses_axis_qualified_identifiers():
    from oracle_viewer.candidate_mining import _point_number
    assert _point_number('Y1_PNT_Y2') == 2
    assert _point_number('W04_PNT_Z3') == 3
    assert _point_number('N3_PNT1') == 1


def test_semantic_fallback_never_crosses_up_down_programs():
    coordinates = [{'page': 1, 'program': 'DOWN', 'measured_system': 'ALI_ABC',
                    'text': '\n'.join(f'X, Y, Z (mm) {i} 0 0 CHASSIS_X{i:02d}' for i in range(1, 5))}]
    method = [{'page': 2, 'program': 'UP', 'measured_system': 'ALI_ABD_VISION',
               'features': ['CHASSIS_LIN'], 'text': 'Measure 1 vision line CHASSIS_LIN.'}]
    assert referenced_points(None, method, extract_named_points(coordinates), ['CHASSIS_LIN']) == {}


def test_sibling_constructed_line_rows_are_calibration_support_not_target_markers():
    from oracle_viewer.candidate_mining import constructed_support_rows
    pages = [
        {'page': 68, 'fai_ids': [51], 'program': 'UP', 'measured_system': 'ALI_ABC_A',
         'text': 'Measure 2 vector points Y1_PNT_Y2, Y1_PNT_Y3 under alignment ALI_ABD and construct a line\n'
                 ' Probe MASTER_P5 Method Y1_YLIN_BASE.\n'
                 'X, Y, Z (mm) -24.57 69.48 .4 Y1_PNT_Y2\nX, Y, Z (mm) 27.17 69.48 .4 Y1_PNT_Y3'},
        {'page': 69, 'fai_ids': [52], 'program': 'UP', 'measured_system': 'ALI_ABC_A',
         'text': 'Measure 2 vector points Y1_PNT_Y1, Y1_PNT_Y4 under alignment ALI_ABD and construct a line\n'
                 ' Probe MASTER_P5 Method Y1_YLIN_TIP.\n'
                 'X, Y, Z (mm) -24.57 69.48 2.83 Y1_PNT_Y1\nX, Y, Z (mm) 27.17 69.48 2.83 Y1_PNT_Y4'},
    ]
    points = extract_named_points(pages)
    target = _coordinate_rows([pages[0]], [], [], pages, points)
    support = constructed_support_rows(target, [pages[0]], points)
    assert {row['name'] for row in target} == {'Y1_PNT_Y2', 'Y1_PNT_Y3'}
    assert {row['name'] for row in support} == {'Y1_PNT_Y1', 'Y1_PNT_Y2', 'Y1_PNT_Y3', 'Y1_PNT_Y4'}
    assert {row['coordinate_system'] for row in support} == {'ALI_ABD'}
