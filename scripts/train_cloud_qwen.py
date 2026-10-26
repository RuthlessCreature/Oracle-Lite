"""Isolated Qwen multimodal QLoRA trainer; existing base checkpoint stays read-only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import re
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_answer(text, expected):
    """Compare named coordinates independent of row order; count invented points."""
    truth = json.loads(expected)
    if 'source_point_names' in truth:
        return score_manifest_answer(text, truth)
    expected_rows = {(r['name'], tuple(r['axes'])): r['values_mm'] for r in truth['points']}
    result = {'valid_json': False, 'schema_valid': False, 'exact': False, 'strict_order_exact': False,
              'correct_points': 0, 'expected_points': len(expected_rows), 'invented_points': 0}
    try:
        strict_json = True
        try:
            parsed = json.loads(text)
        except ValueError:
            fenced = re.fullmatch(r'\s*```(?:json)?\s*\n?(.*?)\n?```\s*', text, flags=re.S | re.I)
            if not fenced:
                return result
            parsed = json.loads(fenced.group(1))
            strict_json = False
        if not isinstance(parsed, dict) or set(parsed) != {'points'} or not isinstance(parsed['points'], list):
            return result
        actual = {}
        for row in parsed['points']:
            if not isinstance(row, dict) or set(row) != {'name', 'axes', 'values_mm'}:
                return result
            if not isinstance(row['name'], str) or row['axes'] not in (['X', 'Y'], ['X', 'Y', 'Z']):
                return result
            if not isinstance(row['values_mm'], list) or len(row['values_mm']) != len(row['axes']):
                return result
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row['values_mm']):
                return result
            key = row['name'], tuple(row['axes'])
            if key in actual:
                return result
            actual[key] = row['values_mm']
        result.update(valid_json=strict_json, schema_valid=True, strict_order_exact=parsed == truth,
                      invented_points=len(actual.keys() - expected_rows.keys()))
        result['correct_points'] = sum(key in actual and all(abs(a-b) <= 1e-9 for a, b in zip(actual[key], values))
                                       for key, values in expected_rows.items())
        result['exact'] = actual.keys() == expected_rows.keys() and result['correct_points'] == len(expected_rows)
    except (ValueError, TypeError, KeyError):
        pass
    return result


def score_manifest_answer(text, truth):
    expected=truth['source_point_names']
    result=dict(valid_json=False,schema_valid=False,exact=False,strict_order_exact=False,
                correct_points=0,expected_points=len(expected),invented_points=0)
    try:
        try: parsed=json.loads(text);strict=True
        except ValueError:
            fenced=re.fullmatch(r'\s*```(?:json)?\s*\n?(.*?)\n?```\s*',text,flags=re.S|re.I)
            if not fenced:return result
            parsed=json.loads(fenced.group(1));strict=False
        if not isinstance(parsed,dict):return result
        names=parsed.get('source_point_names')
        if (not isinstance(names,list) or any(not isinstance(n,str) for n in names)
                or len(names)!=len(set(names)) or type(parsed.get('fai')) is not int
                or parsed['fai']!=truth['fai'] or parsed.get('decision')!=truth['decision']
                or not isinstance(parsed.get('evidence'),str) or not parsed['evidence'].strip()
                or parsed.get('confidence') not in ('high','medium','low')):return result
        result.update(valid_json=strict,schema_valid=True,correct_points=len(set(names)&set(expected)),
            invented_points=len(set(names)-set(expected)),exact=set(names)==set(expected),
            strict_order_exact=names==expected)
    except (TypeError,ValueError):pass
    return result


def answer_only_loss(model, batch):
    """Exact causal SFT loss without materializing logits for masked evidence."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    inputs = dict(batch)
    labels = inputs.pop('labels')
    if labels.shape[0] != 1:
        raise ValueError('Answer-only loss requires batch size 1')
    positions = torch.where(labels[0, 1:] != -100)[0]
    if not len(positions):
        raise ValueError('No supervised answer tokens')
    targets = labels[0, positions + 1]
    outputs = model(**inputs, logits_to_keep=positions)
    logits = outputs.logits[0]
    def cross_entropy(values, truth):
        return F.cross_entropy(values.float(), truth, reduction='sum')
    parts = [checkpoint(cross_entropy, logits[start:start+64], targets[start:start+64],
                        use_reentrant=False, preserve_rng_state=False)
             for start in range(0, len(targets), 64)]
    return torch.stack(parts).sum() / len(targets), outputs


def load_rows(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    result = {}
    for split in ('train', 'validation'):
        path = root / f'{split}.jsonl'
        if digest(path) != manifest['sha256'][split]:
            raise ValueError(f'{split} dataset fingerprint changed')
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        for row in rows:
            if row.get('xml_used_as_input') is not False or row.get('geometry_ground_truth') is not False:
                raise ValueError('Source-only dataset policy violation')
            for image in row['images']:
                resolved = (root / image).resolve()
                if not resolved.is_relative_to(root.resolve()) or digest(resolved) != row['image_sha256']:
                    raise ValueError('Image path/fingerprint mismatch')
            if '.xml' in row['source_file'].lower():
                raise ValueError('Existing XML cannot be model input')
        result[split] = rows
    groups = {k: {r['project_group'] for r in v} for k, v in result.items()}
    if not all(result.values()) or groups['train'] & groups['validation']:
        raise ValueError('Training and validation must be drawing-disjoint')
    if ({r['source_sha256'] for r in result['train']} &
            {r['source_sha256'] for r in result['validation']}):
        raise ValueError('Identical source documents leaked across project split')
    return manifest, result


class Collator:
    def __init__(self, processor, root, limit):
        self.processor, self.root, self.limit = processor, root, limit

    def prompt(self, row):
        from PIL import Image
        messages = [
            {'role': 'system', 'content': 'Read measurement evidence exactly. Output the requested JSON only. Never infer missing coordinate axes or CAD positions.'},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': str(self.root / row['images'][0])},
                {'type': 'text', 'text': row['question'] + '\nSource: ' + row['source_file'] +
                 f" page {row['page']}\nNative page text:\n" + row['context']},
            ]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)
        with Image.open(self.root / row['images'][0]) as image:
            batch = self.processor(text=[prompt], images=[image.convert('RGB')], return_tensors='pt')
        return batch

    def __call__(self, features):
        import torch
        if len(features) != 1:
            raise ValueError('Image/text batches require batch_size=1')
        row = features[0]
        batch = self.prompt(row)
        answer = self.processor.tokenizer(row['answer'], add_special_tokens=False, return_tensors='pt')['input_ids']
        eos = self.processor.tokenizer.eos_token_id
        if eos is not None:
            answer = torch.cat([answer, answer.new_tensor([[eos]])], dim=1)
        prompt_len = batch['input_ids'].shape[1]
        total = prompt_len + answer.shape[1]
        if total > self.limit:
            raise ValueError(f"{row['id']}: {total} tokens exceed {self.limit}; no truncated supervision accepted")
        batch['input_ids'] = torch.cat([batch['input_ids'], answer], dim=1)
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        for key in ('mm_token_type_ids', 'token_type_ids'):
            if key in batch:
                batch[key] = torch.cat([batch[key], batch[key].new_zeros(answer.shape)], dim=1)
        batch['labels'] = batch['input_ids'].clone()
        batch['labels'][:, :prompt_len] = -100
        return dict(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['preflight', 'train', 'evaluate'], default='preflight')
    parser.add_argument('--max-length', type=int, default=16384)
    parser.add_argument('--max-pixels', type=int, default=786432)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--eval-pages', type=int, default=6)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--compare-base', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = {'stage': args.stage, 'status': 'running', 'started_at': time.time(),
              'training_scope': 'source_task_pending_manifest', 'geometry_training': False,
              'XML_training': False, 'base_weights_modified': False, 'adapter_training_started': False}
    save(args.output / 'status.json', status)
    try:
        import torch
        from transformers import AutoProcessor
        manifest, rows = load_rows(args.dataset)
        status['training_scope'] = manifest.get('training_scope', 'MSOP_source_coordinate_reading')
        processor = AutoProcessor.from_pretrained(str(args.base), local_files_only=True,
                                                 min_pixels=65536, max_pixels=args.max_pixels)
        collator = Collator(processor, args.dataset, args.max_length)
        if args.stage == 'preflight':
            lengths = []
            for split, items in rows.items():
                for index, row in enumerate(items):
                    batch = collator([row])
                    lengths.append({'id': row['id'], 'split': split, 'tokens': batch['input_ids'].shape[1],
                                    'supervised_tokens': int((batch['labels'] != -100).sum())})
                    if index % 10 == 0:
                        status.update(split=split, done=index + 1, total=len(items))
                        save(args.output / 'status.json', status)
            save(args.output / 'preflight.json', {'passed': True, 'manifest': manifest, 'lengths': lengths,
                                                 'max_tokens': max(r['tokens'] for r in lengths)})
        else:
            from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForImageTextToText, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback
            if torch.cuda.device_count() != 1:
                raise ValueError('Trainer must receive exactly one isolated GPU')
            available = torch.cuda.mem_get_info()[0]
            if available < 28 * 1024**3:
                raise ValueError('Need at least 28 GiB free on isolated training GPU; refusing to compete with live inference')
            model = AutoModelForImageTextToText.from_pretrained(
                str(args.base), local_files_only=True, dtype=torch.bfloat16, device_map={'': 0},
                attn_implementation='sdpa',
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False
            if hasattr(model.config, 'text_config'):
                model.config.text_config.use_cache = False
            if args.stage == 'train':
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                        gradient_checkpointing_kwargs={'use_reentrant': False})
                model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                    target_modules='all-linear', exclude_modules=r'.*(?:visual|vision).*',
                    task_type='CAUSAL_LM', bias='none'))
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                status.update(trainable_parameters=trainable, adapter_training_started=True,
                              device=torch.cuda.get_device_name(0), dataset_sha256=manifest['sha256'])
                save(args.output / 'status.json', status)
                class Dataset(torch.utils.data.Dataset):
                    def __len__(self): return len(rows['train'])
                    def __getitem__(self, index): return rows['train'][index]
                class Progress(TrainerCallback):
                    def on_log(self, _args, state, control, logs=None, **kwargs):
                        status.update(global_step=state.global_step, max_steps=state.max_steps,
                                      epoch=state.epoch, metrics=logs or {}, updated_at=time.time())
                        save(args.output / 'status.json', status)
                training_args = TrainingArguments(
                    output_dir=str(args.output / 'checkpoints'), per_device_train_batch_size=1,
                    gradient_accumulation_steps=4, num_train_epochs=1, max_steps=args.max_steps,
                    learning_rate=5e-5, warmup_steps=.05, bf16=True, fp16=False,
                    gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                    logging_steps=1, save_steps=2, save_total_limit=2,
                    optim='adamw_torch', report_to=[], remove_unused_columns=False,
                    dataloader_num_workers=0, seed=61313599)
                class AnswerTrainer(Trainer):
                    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                        loss, outputs = answer_only_loss(model, inputs)
                        return (loss, outputs) if return_outputs else loss
                trainer = AnswerTrainer(model=model, args=training_args, train_dataset=Dataset(),
                                  data_collator=collator, callbacks=[Progress()])
                trainer.model_accepts_loss_kwargs = False
                trained = trainer.train()
                final = args.output / 'adapter-final'
                trainer.save_model(str(final))
                processor.save_pretrained(str(final))
                save(args.output / 'training-report.json', {'metrics': trained.metrics,
                    'dataset_manifest': manifest, 'base_checkpoint': str(args.base),
                    'base_config_sha256': digest(args.base / 'config.json'),
                    'adapter_sha256': digest(final / 'adapter_model.safetensors'),
                    'training_scope': status['training_scope'], 'base_weights_modified': False,
                    'is_geometry_training': False, 'is_XML_training': False, 'deployed': False})
            else:
                if args.adapter:
                    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
                model.eval()
                items = sorted(rows['validation'], key=lambda r: r['id'])
                if args.eval_pages > 0:
                    indices = sorted({round(i * (len(items)-1) / max(1, args.eval_pages-1)) for i in range(args.eval_pages)})
                    items = [items[i] for i in indices]
                results = []
                for index, row in enumerate(items):
                    batch = collator.prompt(row)
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    target_tokens = len(processor.tokenizer.encode(row['answer'], add_special_tokens=False))
                    from contextlib import nullcontext
                    modes = ['base', 'adapter'] if args.compare_base and args.adapter else ['adapter' if args.adapter else 'base']
                    for mode in modes:
                        context = model.disable_adapter() if args.adapter and mode == 'base' else nullcontext()
                        with context, torch.inference_mode():
                            generated = model.generate(**batch, do_sample=False, use_cache=True,
                                                       max_new_tokens=min(8192, target_tokens+512))
                        text = processor.tokenizer.decode(generated[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
                        results.append({'id': row['id'], 'mode': mode, **score_answer(text, row['answer']), 'output': text})
                    save(args.output / 'evaluation.json', {'adapter': str(args.adapter) if args.adapter else None,
                         'complete': index == len(items)-1, 'pages': index+1,
                         'exact_pages': sum(r['exact'] for r in results),
                         'valid_json_pages': sum(r['valid_json'] for r in results), 'items': results,
                         'scope': 'source_coordinates_only_not_FAI_geometry_or_XML'})
                    status.update(done=index+1, total=len(items), updated_at=time.time())
                    save(args.output / 'status.json', status)
        status.update(status='complete', finished_at=time.time())
        save(args.output / 'status.json', status)
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', finished_at=time.time())
        save(args.output / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
