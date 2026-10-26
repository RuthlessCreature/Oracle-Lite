import json
from scripts.train_cloud_qwen import score_answer


def test_manifest_score_checks_fai_names_and_decision():
    truth=dict(decision='CONFIRM_MANIFEST',fai=23,source_point_names=['P_1','P_2'],evidence='Measure 2 points',confidence='high')
    expected=json.dumps(truth)
    assert score_answer(expected,expected)['exact']
    for changed in ({'fai':24},{'source_point_names':['P_1']},{'source_point_names':['P_1','P_1']},{'decision':'ABSTAIN'},{'confidence':True}):
        assert not score_answer(json.dumps(dict(truth,**changed)),expected)['exact']
    assert score_answer('```json\n'+expected+'\n```',expected)['exact']
    assert not score_answer('```json\n'+expected+'\n```',expected)['valid_json']


def test_abstention_cannot_pass_as_confirmed_empty_manifest():
    truth=dict(decision='ABSTAIN',fai=1,source_point_names=[],evidence='Bad candidate',confidence='high')
    assert score_answer(json.dumps(truth),json.dumps(truth))['exact']
    assert not score_answer(json.dumps(dict(truth,decision='CONFIRM_MANIFEST')),json.dumps(truth))['exact']
