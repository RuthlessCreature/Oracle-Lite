from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig
from .db import Registry
from .models import DEFAULT_BASE_MODEL_ID, ensure_base_model


RTX4080_16GB_MULTIMODAL_PRESET = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "compute_dtype": "bfloat16",
    "allow_tf32": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "text_max_length": 2048,
    "visual_text_max_chars": 6000,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "gradient_checkpointing": True,
    "learning_rate": 5e-5,
    "num_train_epochs": 1.0,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "optim": "paged_adamw_8bit",
    "logging_steps": 10,
    "save_steps": 50,
    "save_total_limit": 3,
    "dataloader_num_workers": 0,
    "seed": 42,
    "trust_remote_code": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MultimodalDomainCollator:
    """Batch-size-1 collator mixing text CLM and source-grounded VLM supervision."""

    def __init__(self, processor, cfg: dict):
        self.processor = processor
        self.cfg = cfg

    def __call__(self, features: list[dict]):
        if len(features) != 1:
            raise ValueError("RTX4080 multimodal preset requires micro_batch_size=1")

        item = features[0]
        mode = item.get("mode", "text")
        text = (item.get("text") or "").strip()

        if mode != "multimodal" or not item.get("images"):
            batch = self.processor.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg["text_max_length"],
                padding=False,
            )
            labels = batch["input_ids"].clone()
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100
            batch["labels"] = labels
            return batch

        target_text = text[: self.cfg["visual_text_max_chars"]]
        if not target_text:
            raise ValueError("Visual record has no source-grounded target text")

        user_content = [
            {"type": "image", "path": str(Path(image_path).resolve())}
            for image_path in item["images"]
        ]
        user_content.append({
            "type": "text",
            "text": (
                "读取这些原始资料图像，理解其中可见的文字、表格、图示与版面结构。"
                "仅依据图像本身，不补充外部事实。"
            ),
        })

        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            *prompt_messages,
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
        ]

        prompt_batch = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        batch = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        prompt_len = min(prompt_batch["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_len] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch


def run_domain_training(
    app_cfg: AppConfig,
    *,
    snapshot_id: str,
    max_steps: int | None = None,
) -> str:
    """Train Qwen3.5-9B-Base on mixed text + image/text domain records.

    The default model is downloaded automatically into output/models.
    Vision encoder base parameters are frozen for RTX 4080 memory safety, while
    images still flow through the native vision tower and language-side LoRA learns
    from the visual embeddings.
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from exc

    cfg = dict(RTX4080_16GB_MULTIMODAL_PRESET)
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)

    model_path = ensure_base_model(app_cfg)
    dataset_dir = (app_cfg.datasets_dir / snapshot_id / "domain").resolve()
    dataset_meta_path = dataset_dir / "dataset.json"
    if not dataset_meta_path.exists():
        raise FileNotFoundError(
            f"Domain dataset not built for snapshot {snapshot_id}. "
            f"Run: oracle-lite build-domain {snapshot_id}"
        )

    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    shard_paths = sorted(str(p) for p in dataset_dir.glob("part-*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No domain shards found under {dataset_dir}")

    output_dir = (
        app_cfg.training_dir / snapshot_id / "qwen3.5-9b-base-multimodal-rtx4080"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = Registry(app_cfg.registry_path)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    with registry.connect() as conn:
        conn.execute(
            """INSERT INTO training_runs(run_id,snapshot_id,kind,status,config_path,output_dir,started_at)
               VALUES(?,?,'multimodal-domain','running',NULL,?,?)""",
            (run_id, snapshot_id, str(output_dir), _now()),
        )

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available. Local multimodal training requires NVIDIA CUDA.")

        if cfg["allow_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=cfg["trust_remote_code"],
        )
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_path),
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.bfloat16,
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

        # 4080 policy: keep native vision tower fixed. Multimodal samples still
        # traverse it; trainable LoRA lives outside visual/vision modules.
        for name, parameter in model.named_parameters():
            lowered = name.lower()
            if "visual" in lowered or "vision" in lowered:
                parameter.requires_grad = False

        raw_ds = load_dataset("json", data_files=shard_paths, split="train")
        train_ds = raw_ds.filter(
            lambda row: bool((row.get("text") or "").strip()),
            desc="Dropping unlabeled visual-only records",
        )
        if len(train_ds) == 0:
            raise ValueError("No trainable records remain after source-grounding checks")

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
            dataloader_num_workers=0,
            seed=cfg["seed"],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=MultimodalDomainCollator(processor, cfg),
        )

        checkpoint = get_last_checkpoint(str(output_dir))
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(output_dir / "adapter-final"))
        processor.save_pretrained(str(output_dir / "adapter-final"))

        (output_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "snapshot_id": snapshot_id,
                    "base_model_id": DEFAULT_BASE_MODEL_ID,
                    "base_model_path": str(model_path),
                    "dataset": dataset_meta,
                    "preset": "rtx4080_16gb_qwen35_multimodal_v1",
                    "preset_values": cfg,
                    "vision_base_frozen": True,
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
