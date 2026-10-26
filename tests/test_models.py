from oracle_lite.models import DEFAULT_BASE_MODEL_ID


def test_default_base_is_qwen35_multimodal_base():
    assert DEFAULT_BASE_MODEL_ID == "Qwen/Qwen3.5-9B-Base"


def test_model_cache_requires_all_indexed_weights_and_processor_files(tmp_path):
    import json
    from oracle_lite.models import model_files_complete
    (tmp_path / 'config.json').write_text('{}')
    assert not model_files_complete(tmp_path)
    for name in ('preprocessor_config.json','tokenizer_config.json','tokenizer.json'):
        (tmp_path / name).write_text('{}')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'layer_a':'part-a.safetensors', 'layer_b':'part-b.safetensors'}}))
    (tmp_path / 'part-a.safetensors').write_bytes(b'weight-a')
    assert not model_files_complete(tmp_path)
    (tmp_path / 'part-b.safetensors').write_bytes(b'')
    assert not model_files_complete(tmp_path)
    (tmp_path / 'part-b.safetensors').write_bytes(b'weight-b')
    assert model_files_complete(tmp_path)
