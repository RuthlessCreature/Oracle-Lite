import collections
import json

import pytest

from oracle_viewer import integrated_schemes as module


def catalogue(count):
    return {'items': [dict(id=f'S{i}', kind='surface_sensor' if i % 2 else 'confocal_sensor',
                           vendor='vendor', spec={'range_mm': i, 'source_details': str(i) * 350})
                      for i in range(count)]}


def test_all_sensors_are_reviewed_in_bounded_batches_and_only_model_choices_advance(monkeypatch):
    cat = catalogue(40)
    routes = [dict(number=number, channel='hybrid_metrology', requirements=['retain all points'])
              for number in range(1, 20)]
    calls = []

    def generate(model, prompt, context, max_tokens):
        assert context['method_routes'] == routes
        assert len(json.dumps(module.table_evidence(context), ensure_ascii=False, separators=(',', ':'))) <= 10000
        rows = context['catalogue']
        calls.append([r['id'] for r in rows])
        # Deliberately choose the last candidates: the tool must not pick the first.
        ids = [next(r['id'] for r in reversed(rows) if r['kind'] == kind)
               for kind in sorted({r['kind'] for r in rows})]
        return {'selected_ids': ids, 'review': 'conditional candidates'}, {}

    monkeypatch.setattr(module, 'generate_json', generate)
    ids, attempts = module.shortlist_sensors(None, cat, routes, {}, lambda *args: None, budget_chars=10000)
    screened = [i for a in attempts if a['stage'] == 'sensor_catalogue_batch' for i in a['candidate_ids']]
    assert collections.Counter(screened) == collections.Counter(i['id'] for i in cat['items'])
    assert len(calls) > 1
    assert ids == attempts[-1]['selected_ids']
    assert all(len(a['selected_ids']) <= 6 for a in attempts)
    assert any(a['stage'] == 'sensor_catalogue_merge' for a in attempts)


def test_invented_sensor_retried_without_accepting_tool_fallback(monkeypatch):
    calls = []

    def generate(*args):
        calls.append(1)
        return {'selected_ids': ['invented'], 'review': ''}, {}

    monkeypatch.setattr(module, 'generate_json', generate)
    with pytest.raises(ValueError, match='Local sensor shortlist validation failed'):
        module.shortlist_sensors(None, catalogue(2), [dict(number=1, channel='hybrid_metrology')], {}, lambda *args: None)
    assert len(calls) == 3


def test_2d_routes_do_not_request_depth_sensor_choices(monkeypatch):
    monkeypatch.setattr(module, 'generate_json', lambda *args: pytest.fail('unneeded sensor call'))
    assert module.shortlist_sensors(None, catalogue(4), [dict(number=1, channel='2d_vision')], {}, lambda *args: None) == ([], [])
