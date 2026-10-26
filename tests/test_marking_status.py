import json
from oracle_viewer.api import automatic_marking_sources_ready
from oracle_viewer.marking_status import marker_coverage, unmarked_diagnostics


def test_coverage_counts_actual_anchors_not_training_or_empty_annotations():
    items = [dict(number=1, annotation=dict(origin='local_model_candidate', status='auto_marked', anchors=[])),
             dict(number=2, annotation=dict(origin='local_model_candidate', status='auto_marked', anchors=[{}])),
             dict(number=3, binding=dict(face_ids=['f'])), dict(number=4),
             dict(number=5, annotation=dict(origin='local_model_candidate', status='stale', anchors=[{}]))]
    result = marker_coverage(items)
    assert result == dict(total=5, marked=1, local_model_marked=1,
                          unmarked=4, unmarked_numbers=[1, 3, 4, 5])


def test_unmarked_diagnostics_preserve_program_frame_and_reject_stale_receipt(tmp_path):
    root = tmp_path/'automatic-learning'
    root.mkdir()
    path = root/'registration.json'
    path.write_text(json.dumps(dict(cad_sha256='cad', msop_sha256='msop', frames=[
        dict(program='DOWN', coordinate_system='ALI_A', source_to_cad_registered=False,
             triangle_refinement=dict(status='surface_residual_rejected', threshold_mm=.1,
                                      best=dict(max_residual_mm=.9)))])))
    page = dict(page=1, program='DOWN', measured_system='ALI_A',
                text='X, Y, Z (mm) 1 2 3 PNT_A')
    class Store:
        def _dir(self, pid): return tmp_path
        def active_msop(self, project): return dict(sha256='msop', method_pages=[page])
        def items(self, project): return [dict(number=1, features=['PNT_A'], pages=[1])]
        def detail(self, pid, n): return dict(methods=[page])
    project = dict(id='p', model=dict(source_sha256='cad'))
    result = unmarked_diagnostics(Store(), project)[0]
    assert result['reason'] == 'registration_not_validated'
    assert result['frames'][0]['max_residual_mm'] == .9
    project['model']['source_sha256'] = 'new-cad'
    assert unmarked_diagnostics(Store(), project)[0]['frames'][0]['max_residual_mm'] is None


def test_msop_without_xyz_still_queues_the_local_visual_marking_path():
    project=dict(model=dict(status='ready',source_sha256='cad'))
    msop=dict(sha256='msop',method_pages=[dict(page=1,fai_ids=[1],text='FAI 1')])
    assert automatic_marking_sources_ready(project,msop)
    assert not automatic_marking_sources_ready(dict(model=dict(status='missing',source_sha256='cad')),msop)
    assert not automatic_marking_sources_ready(project,dict(sha256='msop',method_pages=[]))
