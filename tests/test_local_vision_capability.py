import json

from talker.model import (TalkerModel, _local_vision_model_ready, _strata_native_vision_ready,
                          _VISION_CAPABILITY_CACHE)


def test_strata_info_does_not_claim_vision_when_local_service_is_down(tmp_path, monkeypatch):
    _VISION_CAPABILITY_CACHE.clear()
    monkeypatch.setenv('ORACLE_LOCAL_VISION_BASE_URL', 'http://127.0.0.1:7871')
    monkeypatch.setenv('ORACLE_LOCAL_VISION_MODEL', 'oracle-qwen35-adapted')
    monkeypatch.setattr('urllib.request.urlopen',
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError('offline')))
    model = TalkerModel(tmp_path, openai_base_url='http://127.0.0.1:8081',
                        openai_model='qwen3.8-flash-next-iq2_xs', backend_label='strata-native')
    assert model.info()['image_inference'] == 'unsupported_by_native_text_model'


def test_strata_info_recognizes_only_the_configured_loopback_vision_model(tmp_path, monkeypatch):
    _VISION_CAPABILITY_CACHE.clear()
    monkeypatch.setenv('ORACLE_LOCAL_VISION_BASE_URL', 'http://127.0.0.1:7871')
    monkeypatch.setenv('ORACLE_LOCAL_VISION_MODEL', 'oracle-qwen35-adapted')

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({'data': [{'id': 'oracle-qwen35-adapted'}]}).encode()

    monkeypatch.setattr('urllib.request.urlopen', lambda request, timeout: Response())
    model = TalkerModel(tmp_path, openai_base_url='http://127.0.0.1:8081',
                        openai_model='qwen3.8-flash-next-iq2_xs', backend_label='strata-native')
    assert model.info()['image_inference'] == 'local_auxiliary_vision'


def test_local_vision_probe_rejects_non_loopback_and_wrong_model(monkeypatch):
    _VISION_CAPABILITY_CACHE.clear()
    assert not _local_vision_model_ready('https://vision.example', 'oracle-qwen35-adapted')
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b'{"data":[{"id":"other-model"}]}'
    monkeypatch.setattr('urllib.request.urlopen', lambda request, timeout: Response())
    assert not _local_vision_model_ready('http://127.0.0.1:7871', 'oracle-qwen35-adapted')


def test_strata_native_vision_requires_loaded_matching_model_and_images(tmp_path, monkeypatch):
    _VISION_CAPABILITY_CACHE.clear()
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            return json.dumps({'status': 'ok', 'loaded': True, 'images': True,
                               'model': 'qwen3.8-flash-next-iq2_xs'}).encode()
    monkeypatch.setattr('urllib.request.urlopen', lambda request, timeout: Response())
    assert _strata_native_vision_ready('http://127.0.0.1:8081', 'qwen3.8-flash-next-iq2_xs')
    model = TalkerModel(tmp_path, openai_base_url='http://127.0.0.1:8081',
                        openai_model='qwen3.8-flash-next-iq2_xs', backend_label='strata-native')
    assert model.info()['image_inference'] == 'strata_native_vision'
