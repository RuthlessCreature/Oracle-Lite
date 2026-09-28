from pathlib import Path


def test_training_extra_includes_torchvision():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    train_section = text.split("[project.optional-dependencies]", 1)[1]
    assert '"torchvision>=0.21"' in train_section


def test_training_preflight_imports_torchvision_before_model_download():
    text = Path("oracle_lite/training.py").read_text(encoding="utf-8")
    torchvision_pos = text.index("import torchvision")
    ensure_model_pos = text.index("ensure_base_model(app_cfg")
    assert torchvision_pos < ensure_model_pos
    assert "Training runtime dependency preflight failed before model download" in text


def test_rtx4080_training_avoids_stock_peft_fp32_prepare():
    text = Path("oracle_lite/training.py").read_text(encoding="utf-8")
    assert "prepare_model_for_kbit_training" not in text
    assert "_prepare_kbit_model_memory_safe" in text
    assert 'autocast_adapter_dtype=False' in text
    assert 'exclude_modules=cfg["exclude_modules"]' in text
    assert '"PYTORCH_ALLOC_CONF"' in text
    assert 'dtype=torch.bfloat16' in text
    assert 'torch_dtype=torch.bfloat16' not in text


def test_package_versions_match():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    init = Path("oracle_lite/__init__.py").read_text(encoding="utf-8")
    assert 'version = "0.5.2"' in pyproject
    assert '__version__ = "0.5.2"' in init
