from oracle_lite.models import DEFAULT_BASE_MODEL_ID


def test_default_base_is_qwen35_multimodal_base():
    assert DEFAULT_BASE_MODEL_ID == "Qwen/Qwen3.5-9B-Base"
