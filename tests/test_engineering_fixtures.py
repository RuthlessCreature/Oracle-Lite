import json
from oracle_lite.engineering_fixtures import build_cases


def test_fictional_engineering_labels_have_independent_numeric_checks(tmp_path):
    rows = build_cases(tmp_path, seed=2026, count=16)
    assert rows == build_cases(tmp_path, seed=2026, count=16)
    assert all(row['fictional'] and row['text'] == row['answer'] for row in rows)
    import re
    for row in rows:
        if row['fixture_family'] == 0:
            fov, pixels, pitch = re.search(r'FOV=(\d+) mm; horizontal pixels=(\d+); pixel pitch=([\d.]+)', row['context']).groups()
            expected = json.loads(row['answer'])
            assert expected['sampling_um_per_px'] == round(float(fov) * 1000 / int(pixels), 4)
            assert expected['magnification'] == round(int(pixels) * float(pitch) / 1000 / float(fov), 4)
            assert expected['accuracy_verified'] is False
        if row['fixture_family'] == 1:
            expected = json.loads(row['answer'])
            assert expected['y_origin'] == 'BD_ORIGN'
            assert expected['nominal_mm'] != expected['actual_mm']
        if row['fixture_family'] == 3:
            assert len(row['images']) == 1 and row['expected']['actual_y_mm'] is None
        if row['fixture_family'] == 5:
            import ast
            bounds = ast.literal_eval(re.search(r'分别为(\[[^\]]+\])', row['context']).group(1))
            expected = json.loads(row['answer'])
            assert expected['worst_case_um'] == sum(bounds)
            assert expected['accuracy_verified'] is False


def test_train_and_holdout_use_different_values_and_images(tmp_path):
    train = build_cases(tmp_path / 'train', seed=11, count=24)
    heldout = build_cases(tmp_path / 'test', seed=97, count=24)
    assert not set(row['context'] for row in train) & set(row['context'] for row in heldout)
    assert not set(path for row in train for path in row['images']) & set(path for row in heldout for path in row['images'])


def test_equivalent_score_does_not_hide_wrong_units_identifiers_or_extra_fields():
    from oracle_lite.engineering_evaluation import equivalent_mapping
    assert equivalent_mapping({'tol': '±0.12 mm'}, {'tol': '+0.12/-0.12'})
    assert not equivalent_mapping({'tol': '±0.12 um'}, {'tol': '+0.12/-0.12'})
    assert not equivalent_mapping({'tol': '±0.12'}, {'tol': '+0.12/-0.15'})
    assert not equivalent_mapping({'id': 'PARTmm'}, {'id': 'PART'})
    assert not equivalent_mapping({'actual': 0.0}, {'actual': None})
    assert not equivalent_mapping({'x': 1, 'invented': 2}, {'x': 1})
