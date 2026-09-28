from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

import yaml

from .config import AppConfig
from .db import Registry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_training_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = {"base_model", "dataset_dir", "output_dir"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing training config keys: {', '.join(missing)}")
    return raw


def run_cpt_training(app_cfg: AppConfig, training_config_path: str | Path) -> str:
    """Run local QLoRA continued pretraining.

    Heavy ML imports are intentionally lazy so scan/ingest/snapshot commands work
    without CUDA or training dependencies installed.
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from exc

    cfg = _load_training_config(training_config_path)
    dataset_dir = Path(cfg["dataset_dir"]).expanduser().resolve()
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta_path = dataset_dir / "dataset.json"
    if not dataset_meta_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {dataset_meta_path}")
    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    snapshot_id = dataset_meta["snapshot_id"]

    shard_paths = sorted(str(p) for p in dataset_dir.glob("part-*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No CPT shards found under {dataset_dir}")

    registry = Registry(app_cfg.registry_path)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    with registry.connect() as conn:
        conn.execute(
            """INSERT INTO training_runs(run_id,snapshot_id,kind,status,config_path,output_dir,started_at)
               VALUES(?,?,'cpt','running',?,?,?)""",
            (run_id, snapshot_id, str(Path(training_config_path).resolve()), str(output_dir), _now()),
        )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available. Oracle-Lite V0.1 CPT path requires NVIDIA CUDA.")

        if cfg.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True

        compute_dtype_name = str(cfg.get("compute_dtype", "bfloat16")).lower()
        compute_dtype = torch.bfloat16 if compute_dtype_name in {"bf16", "bfloat16"} else torch.float16

        quant = BitsAndBytesConfig(
            load_in_4bit=bool(cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["base_model"],
            use_fast=True,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            quantization_config=quant,
            device_map="auto",
            torch_dtype=compute_dtype,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        )
        model.config.use_cache = False

        gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

        lora = LoraConfig(
            r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            target_modules=cfg.get("target_modules", "all-linear"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

        raw_ds = load_dataset("json", data_files=shard_paths, split="train")
        eos = tokenizer.eos_token or ""
        block_size = int(cfg.get("max_seq_length", 2048))

        def tokenize_batch(batch):
            return tokenizer(
                [text + eos for text in batch["text"]],
                add_special_tokens=False,
                truncation=False,
            )

        tokenized = raw_ds.map(
            tokenize_batch,
            batched=True,
            remove_columns=raw_ds.column_names,
            desc="Tokenizing corpus",
        )

        def group_texts(examples):
            concatenated = {
                key: list(chain.from_iterable(examples[key]))
                for key in examples.keys()
            }
            total_length = len(concatenated["input_ids"])
            total_length = (total_length // block_size) * block_size
            result = {
                key: [values[i : i + block_size] for i in range(0, total_length, block_size)]
                for key, values in concatenated.items()
            }
            result["labels"] = [ids.copy() for ids in result["input_ids"]]
            return result

        train_ds = tokenized.map(
            group_texts,
            batched=True,
            desc=f"Packing into {block_size}-token blocks",
        )

        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(cfg.get("micro_batch_size", 1)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
            learning_rate=float(cfg.get("learning_rate", 1e-4)),
            num_train_epochs=float(cfg.get("num_train_epochs", 1.0)),
            max_steps=int(cfg.get("max_steps", -1)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_steps=int(cfg.get("save_steps", 100)),
            save_total_limit=int(cfg.get("save_total_limit", 3)),
            bf16=compute_dtype is torch.bfloat16,
            fp16=compute_dtype is torch.float16,
            gradient_checkpointing=gradient_checkpointing,
            optim=str(cfg.get("optim", "paged_adamw_8bit")),
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
            seed=int(cfg.get("seed", 42)),
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=default_data_collator,
        )

        resume = cfg.get("resume_from_checkpoint", "auto")
        checkpoint = None
        if resume == "auto":
            checkpoint = get_last_checkpoint(str(output_dir))
        elif isinstance(resume, str) and resume not in {"", "none", "false"}:
            checkpoint = resume
        elif resume is True:
            checkpoint = True

        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(output_dir / "adapter-final"))
        tokenizer.save_pretrained(str(output_dir / "adapter-final"))

        with registry.connect() as conn:
            conn.execute(
                "UPDATE training_runs SET status='completed',ended_at=? WHERE run_id=?",
                (_now(), run_id),
            )
        return run_id

    except Exception:
        with registry.connect() as conn:
            conn.execute(
                "UPDATE training_runs SET status='failed',ended_at=? WHERE run_id=?",
                (_now(), run_id),
            )
        raise
