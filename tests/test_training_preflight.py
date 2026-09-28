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
