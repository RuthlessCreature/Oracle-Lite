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


def test_trainingarguments_compat_uses_warmup_steps_when_ratio_name_is_absent():
    from oracle_lite.training import _build_training_arguments

    class FakeArgs:
        def __init__(
            self,
            output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            max_steps=1,
            warmup_steps=0,
            bf16=False,
        ):
            self.output_dir = output_dir
            self.warmup_steps = warmup_steps

    cfg = {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "learning_rate": 5e-5,
        "num_train_epochs": 1.0,
        "max_steps": 10,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "logging_steps": 10,
        "save_steps": 50,
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "seed": 42,
    }
    args, dropped = _build_training_arguments(
        FakeArgs,
        output_dir=Path("/tmp/train"),
        cfg=cfg,
    )
    assert args.warmup_steps == 0.03
    assert "warmup_ratio" not in dropped


def test_base_multimodal_prompt_does_not_require_chat_template():
    from oracle_lite.training import _base_multimodal_prompt

    class FakeProcessor:
        vision_start_token = "<VS>"
        image_token = "<IMG>"
        vision_end_token = "<VE>"

    prompt = _base_multimodal_prompt(FakeProcessor())
    assert prompt.startswith("<VS><IMG><VE>")
    assert "SOURCE_GROUNDED_TEXT" in prompt

    training = Path("oracle_lite/training.py").read_text(encoding="utf-8")
    assert ".apply_chat_template(" not in training


def test_package_versions_match_054():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    init = Path("oracle_lite/__init__.py").read_text(encoding="utf-8")
    assert 'version = "0.5.4"' in pyproject
    assert '__version__ = "0.5.4"' in init
