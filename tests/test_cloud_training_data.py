import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cloud_prep', ROOT / 'scripts/prepare_cloud_qwen_training.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_printed_xy_does_not_acquire_an_invented_z():
    page = {'text': 'Features     X       Y\nFAI14_P01   2.000   3.000\n', 'page': 1}
    rows = module.coordinate_rows(page)
    assert rows == [{'name': 'FAI14_P01', 'axes': ['X', 'Y'], 'values_mm': [2., 3.]}]


def test_conflicting_source_values_do_not_become_training_truth():
    page = {'text': 'X, Y, Z (mm) 1 2 3 PNT_A\nX, Y, Z (mm) 1 2 4 PNT_A', 'page': 1}
    with pytest.raises(ValueError, match='conflicting'):
        module.coordinate_rows(page)


def test_dataset_is_project_disjoint_and_never_reads_xml(tmp_path):
    import fitz
    sources = {}
    for group in ['train_project', 'heldout_project']:
        pdf = tmp_path / f'{group}.pdf'
        with fitz.open() as doc:
            doc.new_page()
            doc.set_metadata({'title': group})
            doc.save(pdf)
        sources[group] = {'path': str(pdf), 'filename': 'MSOP.pdf', 'sha256': module.sha(pdf.read_bytes()),
                         'method_pages': [{'page': 1, 'text': 'X, Y, Z (mm) 1 2 3 PNT_A'}]}
    class Store:
        def listing(self): return [{'id': 'train_project'}, {'id': 'heldout_project'}]
        def get(self, pid):
            return {'id': pid, 'drawing_no': pid, 'reference_files': [{'filename': 'forbidden.xml', 'path': '/nonexistent/XML_must_not_be_read'}]}
        def active_msop(self, project): return sources[project['id']]
    output = tmp_path / 'dataset'
    result = module.prepare(Store(), output, 'heldout_project', set())
    assert result['train_groups'] == ['train_project']
    assert result['validation_groups'] == ['heldout_project']
    for split in ('train', 'validation'):
        row = json.loads((output / f'{split}.jsonl').read_text())
        assert row['xml_used_as_input'] is False
        assert 'forbidden.xml' not in row['context']
        assert row['source_file'] == 'MSOP.pdf'
        assert row['geometry_ground_truth'] is False
    sources['heldout_project'] = sources['train_project']
    with pytest.raises(ValueError, match='Same MSOP bytes'):
        module.prepare(Store(), tmp_path/'leaking_dataset', 'heldout_project', set())


def test_multimodal_supervision_masks_prompt_and_preserves_image_metadata(tmp_path):
    import torch
    from PIL import Image
    spec = importlib.util.spec_from_file_location('cloud_train', ROOT / 'scripts/train_cloud_qwen.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    Image.new('RGB', (32, 32)).save(tmp_path / 'page.png')
    class Tokenizer:
        eos_token_id = 99
        def __call__(self, *args, **kwargs): return {'input_ids': torch.tensor([[20, 21]])}
    class Processor:
        tokenizer = Tokenizer()
        def apply_chat_template(self, *args, **kwargs): return 'source prompt'
        def __call__(self, *args, **kwargs):
            return {'input_ids': torch.tensor([[10, 11, 12]]),
                    'mm_token_type_ids': torch.tensor([[1, 1, 0]]),
                    'pixel_values': torch.ones(1, 3, 32, 32),
                    'image_grid_thw': torch.tensor([[1, 2, 2]])}
    row = {'id': 'source:1', 'images': ['page.png'], 'question': 'extract XYZ',
           'source_file': 'MSOP.pdf', 'page': 1, 'context': 'printed XYZ', 'answer': '{}'}
    batch = trainer.Collator(Processor(), tmp_path, 6)([row])
    assert batch['labels'].tolist() == [[-100, -100, -100, 20, 21, 99]]
    assert batch['mm_token_type_ids'].tolist() == [[1, 1, 0, 0, 0, 0]]
    assert batch['image_grid_thw'].tolist() == [[1, 2, 2]]
    with pytest.raises(ValueError, match='no truncated supervision'):
        trainer.Collator(Processor(), tmp_path, 5)([row])


def test_coordinate_benchmark_rejects_invented_axes_and_wrong_values():
    spec = importlib.util.spec_from_file_location('cloud_scores', ROOT / 'scripts/train_cloud_qwen.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    points = [{'name': 'PNT_A', 'axes': ['X', 'Y'], 'values_mm': [1., 2.]},
              {'name': 'PNT_B', 'axes': ['X', 'Y', 'Z'], 'values_mm': [3., 4., 5.]}]
    expected = json.dumps({'points': points})
    reordered = trainer.score_answer(json.dumps({'points': list(reversed(points))}), expected)
    assert reordered['exact'] and not reordered['strict_order_exact']
    assert reordered['correct_points'] == 2
    fenced = trainer.score_answer('```json\n'+expected+'\n```', expected)
    assert fenced['exact'] and fenced['correct_points'] == 2
    assert fenced['schema_valid'] and not fenced['valid_json']
    invented = json.loads(expected)
    invented['points'][0].update(axes=['X', 'Y', 'Z'], values_mm=[1., 2., 0.])
    result = trainer.score_answer(json.dumps(invented), expected)
    assert not result['exact'] and result['correct_points'] == 1 and result['invented_points'] == 1
    wrong = json.loads(expected)
    wrong['points'][1]['values_mm'][2] = 5.01
    result = trainer.score_answer(json.dumps(wrong), expected)
    assert not result['exact'] and result['correct_points'] == 1


def test_answer_only_loss_and_gradients_equal_full_masked_causal_loss():
    import copy
    from types import SimpleNamespace
    import torch
    from torch.nn import functional as F
    spec = importlib.util.spec_from_file_location('cloud_loss', ROOT / 'scripts/train_cloud_qwen.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    torch.manual_seed(42)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(17, 8)
            self.head = torch.nn.Linear(8, 17)
        def forward(self, input_ids, logits_to_keep=None):
            hidden = self.embedding(input_ids)
            if logits_to_keep is not None: hidden = hidden[:, logits_to_keep]
            return SimpleNamespace(logits=self.head(hidden))
    full = Model()
    selected = copy.deepcopy(full)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    labels = tokens.clone()
    labels[:, :4] = -100
    loss = F.cross_entropy(full(tokens).logits[:, :-1].reshape(-1, 17), labels[:, 1:].reshape(-1))
    efficient, outputs = trainer.answer_only_loss(selected, {'input_ids': tokens, 'labels': labels})
    torch.testing.assert_close(efficient, loss)
    assert outputs.logits.shape == (1, 4, 17)
    loss.backward()
    efficient.backward()
    for before, after in zip(full.parameters(), selected.parameters()):
        torch.testing.assert_close(before.grad, after.grad)
