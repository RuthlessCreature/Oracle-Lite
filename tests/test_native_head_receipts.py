import json
from pathlib import Path
from oracle_viewer.native_head_training import attach_source_receipt,backbone_fingerprint,file_sha,validated_source_receipt

def fixture(tmp_path):
 bundle=tmp_path/'bundle';bundle.mkdir();base=tmp_path/'base';base.write_bytes(b'backbone');adapter=bundle/'adapter.oha';adapter.write_bytes(b'fixture-adapter')
 state={'model_identity':{'model_alias':'strata-test'},'curriculum':{'test_count':2,'file_sha256':{'train':'a','test':'b'}},'phases':{},'geometry_frames':[{'training_eligible':False}]}
 report={'adapter_trained':True,'model_identity':state['model_identity'],'source_curriculum_sha256':state['curriculum']['file_sha256'],'backbone_fingerprint':backbone_fingerprint([base]),'adapter_sha256':file_sha(adapter),'train_samples':96,'test_samples':48,'parameters':100,'steps':120,'full_heldout':{'adapter':{'pages':2,'semantic_exact_pages':2,'truncated_pages':0,'correct_points':2,'expected_points':2,'schema_valid_pages':2}}};(bundle/'training-report.json').write_text(json.dumps(report));return state,bundle,base,report

def test_matching_receipt_reused_without_claiming_geometry(tmp_path):
 state,bundle,_,_=fixture(tmp_path);assert attach_source_receipt(state,bundle,tmp_path)
 assert state['status']=='source_adapter_validated_geometry_pending' and state['adapter_weights_changed']
 assert not state['weights_changed'] and not state['geometry_frames'][0]['training_eligible']

def test_stale_source_not_reused(tmp_path):
 state,bundle,_,_=fixture(tmp_path);state['curriculum']['file_sha256']['train']='changed';assert validated_source_receipt(state,bundle) is None

def test_other_model_not_reused(tmp_path):
 state,bundle,_,_=fixture(tmp_path);state['model_identity']['model_alias']='other';assert validated_source_receipt(state,bundle) is None

def test_changed_backbone_not_reused(tmp_path):
 state,bundle,base,_=fixture(tmp_path);base.write_bytes(b'changed');assert validated_source_receipt(state,bundle) is None

def test_changed_adapter_not_reused(tmp_path):
 state,bundle,_,_=fixture(tmp_path);(bundle/'adapter.oha').write_bytes(b'changed');assert validated_source_receipt(state,bundle) is None

def test_regressed_result_is_not_a_verified_receipt(tmp_path):
 state,bundle,_,r=fixture(tmp_path);r['full_heldout']['adapter']['semantic_exact_pages']=1;(bundle/'training-report.json').write_text(json.dumps(r));assert validated_source_receipt(state,bundle) is None
