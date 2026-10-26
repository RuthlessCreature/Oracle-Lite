from oracle_lite.grounding import GROUNDING_PROMPT_VERSION, SYSTEM_TEXT, instruction_conversation
from oracle_lite.grounding_stress import score_json_mapping
from oracle_lite.robust_instruction_fixtures import build_fixtures
from oracle_lite.instruction_fixtures import build_fixtures as build_method_fixtures
from oracle_lite.novel_template_stress import build_cases as build_novel_template_cases


def test_grounding_prompt_limits_narrow_answers_without_losing_conflict_handling():
    assert GROUNDING_PROMPT_VERSION == "grounded-conversation-v10-engineering"
    assert "Do not expand a singular or narrow question into a catalog" in SYSTEM_TEXT.replace("\n", " ")
    assert "report those values and" in SYSTEM_TEXT
    assert "their scopes" in SYSTEM_TEXT
    assert "compare those" in SYSTEM_TEXT
    assert "every directly matching row" in SYSTEM_TEXT
    assert "exact printed label" in SYSTEM_TEXT
    assert "preserve each available revision" in SYSTEM_TEXT
    prompt = instruction_conversation("What is the probe?", "Probe: P5", "page 4")
    assert "What is the probe?" in prompt
    assert "Probe: P5" in prompt


def test_json_mapping_score_separates_schema_and_values_from_trailing_citation():
    expected = {"Row 1": {"label": "OUTER WIDTH", "value": "12.34", "tolerance": "±0.04"}}
    valid = '{"Row 1":{"label":"OUTER WIDTH","value":"12.34","tolerance":"±0.04"}}'
    trailing = valid + "\n[Source 1: drawing.pdf, page 2]"
    merged = '{"Row 1":{"label":"OUTER WIDTH","value":"12.34 ±0.04"}}'

    assert score_json_mapping(valid, expected) == {
        "json_only": True, "mapping_exact": True, "fields_matched": 3,
        "expected_fields": 3, "expected_values_present": 3,
        "parsed_with_trailing_text": False,
    }
    trailing_score = score_json_mapping(trailing, expected)
    assert trailing_score["mapping_exact"] and not trailing_score["json_only"]
    assert trailing_score["parsed_with_trailing_text"]
    merged_score = score_json_mapping(merged, expected)
    assert not merged_score["mapping_exact"]
    assert not merged_score["json_only"]


def test_robust_instruction_fixtures_are_deterministic_fictional_json_tasks():
    first = build_fixtures(seed=18, count=20)
    assert first == build_fixtures(seed=18, count=20)
    assert all(row["fictional"] and row["text"] == row["answer"] for row in first)
    assert [row["fixture_kind"] for row in first] == [0, 1, 2, 3, 4] * 4
    for row in first:
        assert row["answer"].startswith("{") and row["answer"].endswith("}")


def test_feature_identifiers_in_json_fixture_are_strings():
    rows = build_method_fixtures(seed=2026, count=80)
    feature_rows = [row for row in rows if row["fixture_kind"] == 4 and row["answer"].startswith("{")]
    assert feature_rows
    import json
    for row in feature_rows:
        features = json.loads(row["answer"])["features"]
        assert features and all(isinstance(feature, str) for feature in features)


def test_novel_template_suite_is_randomized_balanced_and_truthful():
    import json

    cases = build_novel_template_cases(seed=44, count=36)
    assert cases == build_novel_template_cases(seed=44, count=36)
    assert [case["family"] for case in cases] == [family for _ in range(6) for family in range(6)]
    assert all(case["fictional"] and case["template_split"] == "held_out_surface_form"
               for case in cases)
    for case in cases:
        expected = json.loads(case["answer"])
        assert expected == case["expected"]
        assert case["text"] == case["answer"]
        assert all(value in case["context"] for value in _scalar_values(expected)
                   if isinstance(value, str))


def _scalar_values(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _scalar_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _scalar_values(child)
    else:
        yield value
