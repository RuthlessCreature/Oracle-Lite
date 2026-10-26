from oracle_lite.evaluation import evaluate_cases
from talker.retrieval import RetrievedPassage
import pytest


class Retriever:
    def search(self, question, limit=3):
        return [RetrievedPassage('pressure-spec.pdf', 'page 4', 'Revision B pressure limit: 12 kPa')]


def test_evaluation_requires_correct_source_and_supporting_values():
    report = evaluate_cases(Retriever(), [
        {'question': 'Revision B pressure?', 'expected_evidence': [
            {'source': 'pressure-spec.pdf', 'section': 'page 4', 'contains': ['12 kPa']}]},
        {'question': 'Other source pressure?', 'expected_evidence': [
            {'source': 'other-spec.pdf', 'contains': ['12 kPa']}]},
    ])
    assert report['mode'] == 'retrieval-only'
    assert report['evidence_passed'] == 1
    assert 'generated_answer' not in report['results'][0]


def test_evaluation_does_not_conflate_retrieval_success_with_generation():
    report = evaluate_cases(Retriever(), [
        {'question': 'Pressure limit?', 'expected_evidence': [{'contains': ['12 kPa']}],
         'answer_contains': ['12 kPa'], 'answer_excludes': ['90 kPa']},
    ], answer_fn=lambda messages: 'The pressure limit is 90 kPa.')
    assert report['mode'] == 'retrieval+generation'
    assert report['evidence_passed'] == 1
    assert report['results'][0]['answer_string_checks_passed'] is False


def test_generated_answer_without_assertions_is_unscored():
    report = evaluate_cases(Retriever(), [
        {'question': 'Pressure?', 'expected_evidence': [{'contains': ['12 kPa']}]},
    ], answer_fn=lambda messages: 'Unchecked answer')
    assert report['results'][0]['answer_string_checks_passed'] is None


def test_field_value_patterns_reject_swapped_values():
    case = {'question': 'What are pressure and temperature limits?',
            'split': 'held-out', 'expected_evidence': [{'source': 'pressure-spec.pdf'}],
            'answer_patterns': [r'Pressure\s*[:=]\s*12\s*kPa', r'Temperature\s*[:=]\s*90\s*C']}
    wrong = evaluate_cases(Retriever(), [case], answer_fn=lambda _: 'Pressure: 90 C; Temperature: 12 kPa')
    right = evaluate_cases(Retriever(), [case], answer_fn=lambda _: 'Pressure = 12 kPa; Temperature: 90 C')
    assert wrong['answer_passed'] == 0
    assert right['by_split']['held-out']['answer_passed'] == 1


def test_unanswerable_case_does_not_count_irrelevant_retrieval_as_evidence():
    case = {'question': 'What is the unknown calibration interval?', 'unanswerable': True,
            'answer_any_of': ['not specified', 'cannot determine'],
            'answer_forbidden_patterns': [r'\d+\s*(days|months|years)']}
    right = evaluate_cases(Retriever(), [case], answer_fn=lambda _: 'The interval is not specified.')
    wrong = evaluate_cases(Retriever(), [case], answer_fn=lambda _: 'It is not specified, probably 12 months.')
    assert right['evidence_scored'] == right['evidence_passed'] == 0
    assert right['results'][0]['evidence_passed'] is None
    assert right['abstention_passed'] == 1
    assert wrong['abstention_passed'] == 0


def test_invalid_suite_is_rejected_before_any_generation():
    calls = []
    with pytest.raises(ValueError, match='abstention'):
        evaluate_cases(Retriever(), [
            {'question': 'Valid?', 'expected_evidence': [{'contains': ['12 kPa']}]},
            {'question': 'Unknown?', 'unanswerable': True},
        ], answer_fn=lambda _: calls.append('generated'))
    assert calls == []


def test_empty_evidence_constraint_is_not_an_automatic_pass():
    with pytest.raises(ValueError, match='constrain'):
        evaluate_cases(Retriever(), [{'question': 'Pressure?', 'expected_evidence': [{}]}])
