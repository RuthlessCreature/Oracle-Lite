from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from .config import AppConfig
from .db import Registry


RTX4080_16GB_CPT_PRESET = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "compute_dtype": "bfloat16",
    "allow_tf32": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "max_seq_length": 2048,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": True,
    "learning_rate": 1e-4,
    "num_train_epochs": 1.0,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "optim": "paged_adamw_8bit",
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 3,
    "dataloader_num_workers": 0,
    "seed": 42,
    "trust_remote_code": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned[-80:] or "model"


def run_cpt_training(
    app_cfg: AppConfig,
    *,
    snapshot_id: str,
    base_model: str | Path,
    max_steps: int | None = None,
) -> str:
    """Run local 4-bit QLoRA continued pretraining using RTX 4080 defaults.

    User configuration remains limited to MiniMax key, corpus path and output
    path. Training knobs live in this built-in preset; only the local base-model
    path and target snapshot are selected at run time.
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

    cfg = dict(RTX4080_16GB_CPT_PRESET)
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)

    base_model = Path(base_model).expanduser().resolve()
    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")

    dataset_dir = (app_cfg.datasets_dir / snapshot_id / "cpt").resolve()
    dataset_meta_path = dataset_dir / "dataset.json"
    if not dataset_meta_path.exists():
        raise FileNotFoundError(
            f"CPT dataset not built for snapshot {snapshot_id}. "
            f"Run: oracle-lite build-cpt {snapshot_id}"
        )

    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    if dataset_meta.get("snapshot_id") != snapshot_id:
        raise ValueError("Dataset manifest snapshot_id does not match requested snapshot")

    shard_paths = sorted(str(p) for p in dataset_dir.glob("part-*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No CPT shards found under {dataset_dir}")

    output_dir = (
        app_cfg.training_dir
        / snapshot_id
        / f"{_slug(base_model.name)}-rtx4080-qlora"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = Registry(app_cfg.registry_path)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    with registry.connect() as conn:
        conn.execute(
            """INSERT INTO training_runs(run_id,snapshot_id,kind,status,config_path,output_dir,started_at)
               VALUES(?,?,'cpt','running',NULL,?,?)""",
            (run_id, snapshot_id, str(output_dir), _now()),
        )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available. Local CPT requires NVIDIA CUDA.")

        if cfg["allow_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True

        compute_dtype = torch.bfloat16

        quant = BitsAndBytesConfig(
            load_in_4bit=cfg["load_in_4bit"],
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=compute_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model),
            use_fast=True,
            trust_remote_code=cfg["trust_remote_code"],
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(base_model),
            quantization_config=quant,
            device_map="auto",
            torch_dtype=compute_dtype,
            trust_remote_code=cfg["trust_remote_code"],
        )
        model.config.use_cache = False

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["gradient_checkpointing"],
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg["lora_r"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                bias="none",
                target_modules=cfg["target_modules"],
                task_type="CAUSAL_LM",
            ),
        )

        raw_ds = load_dataset("json", data_files=shard_paths, split="train")
        eos = tokenizer.eos_token or ""
        block_size = cfg["max_seq_length"]

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
                key: [
                    values[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
                for key, values in concatenated.items()
            }
            result["labels"] = [ids.copy() for ids in result["input_ids"]]
            return result

        train_ds = tokenized.map(
            group_texts,
            batched=True,
            desc=f"Packing into {block_size}-token blocks",
        )
        if len(train_ds) == 0:
            raise ValueError(
                f"Dataset is smaller than one {block_size}-token training block."
            )

        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=cfg["micro_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            num_train_epochs=cfg["num_train_epochs"],
            max_steps=int(cfg.get("max_steps", -1)),
            warmup_ratio=cfg["warmup_ratio"],
            weight_decay=cfg["weight_decay"],
            logging_steps=cfg["logging_steps"],
            save_steps=cfg["save_steps"],
            save_total_limit=cfg["save_total_limit"],
            bf16=True,
            fp16=False,
            gradient_checkpointing=cfg["gradient_checkpointing"],
            optim=cfg["optim"],
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=cfg["dataloader_num_workers"],
            seed=cfg["seed"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=default_data_collator,
        )

        checkpoint = get_last_checkpoint(str(output_dir))
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(output_dir / "adapter-final"))
        tokenizer.save_pretrained(str(output_dir / "adapter-final"))

        (output_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "snapshot_id": snapshot_id,
                    "base_model": str(base_model),
                    "preset": "rtx4080_16gb_qlora_cpt_v1",
                    "preset_values": cfg,
                    "completed_at": _now(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

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
