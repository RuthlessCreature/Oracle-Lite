import importlib.util,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from report_native_head_benchmark import metrics

def row(actual,expected=None):
 return dict(project_group='heldout',page=1,output=actual,expected=expected or {'points':[{'name':'P1','xyz_mm':[1,2,3]}]},valid_json=not actual.startswith('```'),exact_structured_answer=False,finish_reason='stop')

def test_reordering_is_not_schema_or_coordinate_error():
 points=[{'name':'P1','xyz_mm':[1,2,3]},{'name':'P2','xyz_mm':[4,5,6]}];r=metrics([row(json.dumps({'points':points[::-1]}),{'points':points})]);assert r['schema_valid_pages']==r['semantic_exact_pages']==1

def test_fenced_correct_coordinates_fail_raw_contract():
 r=metrics([row('```json\n{"points":[{"name":"P1","xyz_mm":[1,2,3]}]}\n```')]);assert r['schema_valid_pages']==0 and r['semantic_exact_pages']==1

@pytest.mark.parametrize('value',[[1,2,4],[1,2,True],[1,2,float('nan')],['1',2,3]])
def test_bad_values_are_not_correct_coordinates(value):
 r=metrics([row(json.dumps({'points':[{'name':'P1','xyz_mm':value}]}))]);assert r['semantic_exact_pages']==r['correct_points']==0

def test_duplicate_records_do_not_count_as_complete():
 r=metrics([row('{"points":[{"name":"P1","xyz_mm":[1,2,3]},{"name":"P1","xyz_mm":[1,2,3]}]}')]);assert r['semantic_exact_pages']==0

def test_bare_array_coordinates_are_distinct_from_schema_failure():
 r=metrics([row('[{"name":"P1","xyz_mm":[1,2,3]}]')])
 assert r['schema_valid_pages']==0 and r['semantic_exact_pages']==1 and r['correct_points']==1
