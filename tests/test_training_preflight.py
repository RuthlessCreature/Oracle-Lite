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


def test_long_sample_budget_is_preserved_across_vram_backoff():
    from oracle_lite.training import _text_backoff_lengths

    assert _text_backoff_lengths(1024, preserve_requested=False) == [1024, 768, 512, 384]
    assert _text_backoff_lengths(2048, preserve_requested=True) == [2048, 2048, 2048, 2048]


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


def test_rtx4080_preset_keeps_explicit_vram_headroom():
    text = Path("oracle_lite/training.py").read_text(encoding="utf-8")
    assert '"text_max_length": 1024' in text
    assert '"vision_max_pixels": 196_608' in text
    assert 'max_memory={0: "9GiB", "cpu": "20GiB"}' in text
    assert "control.should_save = True" in text
    assert "control.should_training_stop = True" in text
    assert "torch.cuda.empty_cache()" in text
    assert "post-step VRAM headroom below safety reserve" in text


def test_package_versions_match_062():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    init = Path("oracle_lite/__init__.py").read_text(encoding="utf-8")
    assert 'version = "0.6.2"' in pyproject
    assert '__version__ = "0.6.2"' in init


def _instruction_collator(max_length):
    import torch
    from types import SimpleNamespace
    from oracle_lite.training import MultimodalDomainCollator

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text, *, add_special_tokens, **kwargs):
            ids = ([1] if add_special_tokens else []) + [ord(char) + 10 for char in text]
            return {'input_ids': torch.tensor([ids], dtype=torch.long)}

    return MultimodalDomainCollator(SimpleNamespace(tokenizer=Tokenizer()),
                                   {'text_max_length': max_length})


def test_instruction_loss_supervises_complete_answer_and_eos_only():
    collator = _instruction_collator(8192)
    row = {'mode': 'instruction', 'question': 'Pressure limit?',
           'context': 'Pressure limit: 12 kPa', 'answer': '12 kPa'}
    batch = collator([row])
    labels = batch['labels'][0]
    expected = [ord(char) + 10 for char in row['answer']] + [2]
    assert labels[labels != -100].tolist() == expected
    assert labels[:len(labels)-len(expected)].eq(-100).all()
    assert batch['attention_mask'].eq(1).all()


def test_chat_checkpoint_instruction_prefix_matches_inference_protocol(tmp_path):
    from oracle_lite.grounding import render_processor_prompt
    from talker.model import TalkerModel
    tokenizer = _instruction_collator(8192).processor.tokenizer
    class ChatProcessor:
        chat_template = 'native'
        def __init__(self):
            self.tokenizer = tokenizer
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
            assert messages[0]['role'] == 'system'
            assert messages[1]['role'] == 'user'
            return '<native-system>'+messages[0]['content']+'<native-user>'+messages[1]['content'][-1]['text']+'<native-assistant>'
    processor = ChatProcessor()
    collator = _instruction_collator(8192)
    collator.processor = processor
    row = {'mode':'instruction','question':'Pressure?', 'context':'Pressure = 12 kPa',
           'answer':'12 kPa', 'source_path':'spec.txt'}
    batch = collator([row])
    messages = [{'role':'user','content':row['question'],'attachments':[{
        'filename':'Retrieved corpus passages','parsed_text':'[Source 1: spec.txt]\n'+row['context'],'images':[]}]}]
    context, images = TalkerModel(tmp_path)._context(messages,14000)
    expected = render_processor_prompt(processor, context)
    assert images == []
    assert batch['input_ids'][0][batch['labels'][0] == -100].tolist() == [1]+[ord(c)+10 for c in expected]
    assert expected.endswith('<native-assistant>')


def test_instruction_prefix_matches_actual_talker_conversation(tmp_path):
    from talker.model import TalkerModel
    from oracle_lite.grounding import render_grounded_prompt

    row = {'mode': 'instruction', 'question': 'Pressure limit?',
           'context': 'Pressure limit: 12 kPa', 'answer': '12 kPa',
           'source_path': '/documents/pressure-spec.pdf'}
    batch = _instruction_collator(8192)([row])
    talker = TalkerModel(tmp_path)
    messages = [{'role': 'user', 'content': row['question'], 'attachments': [{
        'filename': 'Retrieved corpus passages',
        'parsed_text': '[Source 1: pressure-spec.pdf]\n' + row['context'],
        'images': [],
    }]}]
    conversation, images = talker._context(messages, 14000)
    assert images == []
    expected_prompt = render_grounded_prompt(conversation)
    prefix = batch['input_ids'][0][batch['labels'][0] == -100].tolist()
    assert prefix == [1] + [ord(char) + 10 for char in expected_prompt]


def test_instruction_does_not_train_answer_after_dropping_its_evidence():
    import pytest
    collator = _instruction_collator(512)
    row = {'mode': 'instruction', 'question': 'Pressure limit?',
           'context': 'irrelevant ' * 100 + 'Pressure limit: 12 kPa',
           'answer': '12 kPa'}
    with pytest.raises(ValueError, match='never silently truncated'):
        collator([row])


def test_streaming_training_handles_empty_images_and_mixed_metadata_across_shards(tmp_path):
    import json
    from oracle_lite.training import load_training_dataset

    rows = [
        {'mode': 'text', 'text': '12 kPa', 'images': [], 'segment_metadata': {'tag': None}},
        {'mode': 'multimodal', 'text': 'pressure 12 kPa', 'images': ['page.png'],
         'segment_metadata': {'tag': 'table', 'nested': {'column': 1}}},
        {'mode': 'instruction', 'text': '12 kPa', 'images': [],
         'question': 'Pressure?', 'answer': '12 kPa', 'context': 'Pressure: 12 kPa'},
    ]
    paths = []
    for index, row in enumerate(rows):
        path = tmp_path / f'shard-{index}.jsonl'
        path.write_text(json.dumps(row) + '\n')
        paths.append(path)
    assert list(load_training_dataset(paths)) == rows
    assert list(load_training_dataset(paths[::-1])) == rows[::-1]
    from torch.utils.data import DataLoader
    streamed = list(DataLoader(load_training_dataset(paths), batch_size=None, num_workers=0))
    assert streamed == rows


def test_external_instruction_input_is_validated_frozen_and_mode_filtered(tmp_path):
    import hashlib
    import json
    from oracle_lite.training_data import freeze_instruction_shard
    from oracle_lite.training import load_training_dataset
    row = {'mode': 'instruction', 'text': '12 kPa', 'answer': '12 kPa',
           'question': 'Pressure?', 'context': 'Pressure: 12 kPa', 'images': []}
    source = tmp_path / 'input.jsonl'
    original = (json.dumps(row) + '\n').encode()
    source.write_bytes(original)
    frozen, count, digest = freeze_instruction_shard(source, tmp_path / 'frozen')
    source.write_text('changed after freeze')
    assert count == 1 and digest == hashlib.sha256(original).hexdigest()
    assert frozen.read_bytes() == original
    raw = tmp_path / 'raw.jsonl'
    raw.write_text(json.dumps({'mode': 'text', 'text': 'raw document'}) + '\n')
    assert list(load_training_dataset([raw, frozen], mode='instruction')) == [row]


def test_external_instruction_input_rejects_mismatched_answer_and_cleans_temporary(tmp_path):
    import json
    import pytest
    from oracle_lite.training_data import freeze_instruction_shard
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'mode': 'instruction', 'text': '90 kPa', 'answer': '12 kPa',
                                 'question': 'Pressure?', 'context': 'Pressure: 12 kPa'}))
    directory = tmp_path / 'frozen'
    with pytest.raises(ValueError, match='text=answer'):
        freeze_instruction_shard(source, directory)
    assert list(directory.iterdir()) == []


def test_visual_instruction_freezes_image_bytes_and_masks_image_tokens(tmp_path):
    import json
    import torch
    from PIL import Image
    from oracle_lite.training_data import freeze_instruction_shard
    from oracle_lite.training import MultimodalDomainCollator

    original = tmp_path / 'source.png'
    Image.new('RGB', (32, 32), 'white').save(original)
    image_bytes = original.read_bytes()
    row = {'mode': 'instruction', 'question': 'Pressure?', 'context': 'Read image.',
           'text': '12 kPa', 'answer': '12 kPa', 'images': ['source.png']}
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(row)+'\n')
    frozen, count, digest = freeze_instruction_shard(source, tmp_path / 'frozen')
    frozen_row = json.loads(frozen.read_text())
    Image.new('RGB', (32, 32), 'black').save(original)
    assert Path(frozen_row['images'][0]).read_bytes() == image_bytes
    assert count == 1

    tokenizer = _instruction_collator(8192).processor.tokenizer
    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer
        def __call__(self, *, text, images, **kwargs):
            assert images[0].getpixel((0, 0)) == (255, 255, 255)
            assert text[0].startswith('<|vision_start|><|image_pad|><|vision_end|>\n')
            return {'input_ids': torch.tensor([[1, 101, 102, 103]]),
                    'mm_token_type_ids': torch.tensor([[0, 1, 1, 0]]),
                    'pixel_values': torch.ones((4, 3)), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    batch = MultimodalDomainCollator(Processor(), {'text_max_length':4096})([frozen_row])
    assert batch['labels'][0, :4].eq(-100).all()
    assert batch['labels'][0, 4:].tolist() == [ord(c)+10 for c in '12 kPa']+[2]
    assert batch['image_grid_thw'].tolist() == [[1, 2, 2]]
    assert tuple(batch['pixel_values'].shape) == (4, 3)
    assert batch['mm_token_type_ids'].shape == batch['input_ids'].shape
    assert batch['mm_token_type_ids'][0, :4].tolist() == [0, 1, 1, 0]
    assert batch['mm_token_type_ids'][0, 4:].eq(0).all()
