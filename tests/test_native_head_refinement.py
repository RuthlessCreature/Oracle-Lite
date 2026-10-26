import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from train_native_head_refinement import page_partition,syntax_token_ids
from benchmark_native_head_refinement import acceptance
from oracle_viewer.native_head_training import refinement_status
import json,hashlib

def test_page_partition_never_splits_a_page_or_reuses_warm_start_pages():
    rows=[dict(project_group=g,page=p,position=i) for g in ['A','B'] for p in range(20) for i in range(3)]
    prior=[dict(project_group=g,page=p,split='train') for g in ['A','B'] for p in [0,1]]
    fit,dev=page_partition(rows,prior)
    assert not dev.intersection({('A',0),('A',1),('B',0),('B',1)})
    for g in ['A','B']:
        for p in range(20):assert len({fit[i] for i,r in enumerate(rows) if (r['project_group'],r['page'])==(g,p)})==1
    assert np.array_equal(fit,page_partition(rows,prior)[0]) and (~fit).any() and fit.any()

def test_syntax_whitelist_excludes_names_numeric_tokens_and_axis_labels():
    class Tokenizer:
        def encode(self,text,parse_special=False):return list(range(7))
        def decode(self,ids):return ['{','points','xyz_mm','NUT_P001','123','x','```'][ids[0]]
    assert syntax_token_ids(Tokenizer())==[0,1,2,6]

def test_full_answer_gate_rejects_coordinate_regression_despite_better_format():
    before=dict(pages=2,semantic_exact_pages=2,correct_points=10,expected_points=10,schema_valid_pages=1,truncated_pages=0)
    after={**before,'semantic_exact_pages':1,'correct_points':9,'schema_valid_pages':2}
    assert acceptance(before,after)==(False,True)
    assert acceptance(before,{**before,'schema_valid_pages':2})==(True,True)
    assert acceptance(before,before)==(True,False)

def test_refinement_status_rejects_changed_artifact_and_external_report(tmp_path):
    config=tmp_path/'config/training_backends';config.mkdir(parents=True)
    bundle=tmp_path/'reports/refinement';bundle.mkdir(parents=True)
    adapter=bundle/'adapter.oha';adapter.write_bytes(b'checked')
    report=dict(model_identity={'model_alias':'test'},adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest())
    (bundle/'training-report.json').write_text(json.dumps(report))
    profile=dict(model_alias='test',source_output_refinement_report='reports/refinement/training-report.json')
    path=config/'strata-native.json';path.write_text(json.dumps(profile))
    assert refinement_status(tmp_path)['stage']=='trained_not_evaluated'
    adapter.write_bytes(b'changed');assert refinement_status(tmp_path)['stage']=='stale_artifact'
    profile['source_output_refinement_report']='../external.json';path.write_text(json.dumps(profile))
    assert refinement_status(tmp_path) is None
