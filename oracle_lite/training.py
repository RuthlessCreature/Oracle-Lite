from __future__ import annotations

import gc
import inspect
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig
from .db import Registry
from .models import DEFAULT_BASE_MODEL_ID, ensure_base_model
from .resources import (
    GIB,
    HostResourcePolicy,
    current_gpu_memory,
    wait_for_disk,
    wait_for_ram,
    wait_for_vram,
)


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
    "exclude_modules": r".*(?:visual|vision).*",
    "text_max_length": 1024,
    "visual_text_max_chars": 2000,
    "vision_max_pixels": 196_608,
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
    "save_total_limit": 2,
    "dataloader_num_workers": 0,
    "seed": 42,
    "trust_remote_code": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare_kbit_model_memory_safe(
    model,
    *,
    use_gradient_checkpointing: bool,
):
    """Prepare a quantized base model without PEFT's bulk BF16/FP16 -> FP32 cast.

    The stock PEFT helper intentionally promotes non-4-bit parameters to FP32.
    Qwen3.5-9B includes a sizeable BF16 vision tower, so that promotion can need
    several additional GiB on a 16 GiB RTX 4080. Oracle-Lite freezes the base
    model anyway, so keep frozen base parameters in their loaded dtype and only
    enable the gradient plumbing needed by LoRA.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            embeddings = model.get_input_embeddings()

            def _require_grad(_module, _inputs, output):
                if hasattr(output, "requires_grad_"):
                    output.requires_grad_(True)

            embeddings.register_forward_hook(_require_grad)

        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    return model


def _build_training_arguments(
    TrainingArguments,
    *,
    output_dir: Path,
    cfg: dict,
):
    """Build TrainingArguments against the installed Transformers signature.

    Transformers v5 has changed scheduler/trainer argument names across releases.
    Oracle-Lite keeps a small stable contract and maps/drops only non-critical
    compatibility arguments after inspecting the real local constructor.
    """
    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in parameters.values()
    )

    candidate = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": cfg["micro_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": cfg["learning_rate"],
        "num_train_epochs": 1.0,
        "max_steps": int(cfg["max_steps"]),
        "weight_decay": cfg["weight_decay"],
        "logging_steps": cfg["logging_steps"],
        "save_steps": cfg["save_steps"],
        "save_total_limit": cfg["save_total_limit"],
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": cfg["gradient_checkpointing"],
        "optim": cfg["optim"],
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "seed": cfg["seed"],
    }

    # warmup_steps is the most stable v5 surface and accepts a float in [0,1)
    # as a ratio in current Transformers documentation.
    if accepts_kwargs or "warmup_steps" in parameters:
        candidate["warmup_steps"] = float(cfg["warmup_ratio"])
    elif "warmup_ratio" in parameters:
        candidate["warmup_ratio"] = float(cfg["warmup_ratio"])

    required = {
        "output_dir",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_steps",
    }
    if not accepts_kwargs:
        missing = sorted(name for name in required if name not in parameters)
        if missing:
            raise RuntimeError(
                "Installed Transformers TrainingArguments is incompatible with "
                f"Oracle-Lite core training contract; missing: {missing}"
            )

    if accepts_kwargs:
        supported = candidate
        dropped: list[str] = []
    else:
        supported = {
            name: value
            for name, value in candidate.items()
            if name in parameters
        }
        dropped = sorted(set(candidate) - set(supported))

    return TrainingArguments(**supported), dropped


def _base_multimodal_prompt(processor) -> str:
    """Build the raw multimodal prefix used by Qwen VL base models.

    Base checkpoints do not require a chat template. Qwen's processor expands
    the image placeholder when processor(text=..., images=...) is called.
    """
    vision_start = getattr(processor, "vision_start_token", "<|vision_start|>")
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    vision_end = getattr(processor, "vision_end_token", "<|vision_end|>")
    return (
        f"{vision_start}{image_token}{vision_end}\n"
        "Continue from this source image using only information grounded in the "
        "visible document, table, diagram, drawing, or layout. Do not add "
        "external facts.\nSOURCE_GROUNDED_TEXT:\n"
    )


class MultimodalDomainCollator:
    """Batch-size-1 collator mixing text CLM and grounded VLM supervision."""

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

        # Dataset construction enforces one image per multimodal record.
        image_path = str(Path(item["images"][0]).resolve())
        prompt_text = _base_multimodal_prompt(self.processor)
        full_text = prompt_text + target_text

        # Qwen3.5-9B-Base can ship without a processor chat_template. Use the
        # base-native raw VL placeholder path instead of apply_chat_template().
        from PIL import Image

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

            # Compute the exact masked prefix length using the same visual input,
            # then release that temporary batch before constructing the training
            # batch so CPU memory does not hold two sets of pixel tensors.
            prompt_batch = self.processor(
                text=[prompt_text],
                images=[image],
                padding=False,
                return_tensors="pt",
            )
            prompt_len = int(prompt_batch["input_ids"].shape[1])
            del prompt_batch

            batch = self.processor(
                text=[full_text],
                images=[image],
                padding=False,
                return_tensors="pt",
            )

        labels = batch["input_ids"].clone()
        prompt_len = min(prompt_len, labels.shape[1])
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
    monitor=None,
) -> str:
    """Train with streaming data and RAM/disk/VRAM admission control."""
    # PyTorch 2.14 prefers PYTORCH_ALLOC_CONF; expandable segments reduce
    # allocator fragmentation for changing multimodal activation sizes.
    os.environ.setdefault(
        "PYTORCH_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.80",
    )
    try:
        import torch
        import torchvision  # noqa: F401 - required by Qwen3.5 visual/video processor
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            TrainerCallback,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "Training runtime dependency preflight failed before model download. "
            "Qwen3.5 multimodal processing requires a working torchvision build "
            "compatible with the installed torch. Reinstall Oracle-Lite training "
            "dependencies with: python -m pip install -U -e '.[train]'"
        ) from exc

    policy = HostResourcePolicy.auto(app_cfg.output_dir)
    cfg = dict(RTX4080_16GB_MULTIMODAL_PRESET)
    runtime_backoff = {
        "requested": False,
        "reason": None,
        "free_bytes": None,
        "step": None,
    }

    def _wait_state(kind: str, state: dict, resume_phase: str, resume_label: str) -> None:
        if monitor is None:
            return
        label = {
            "ram": "Waiting for RAM",
            "disk": "Waiting for disk space",
            "vram": "Waiting for GPU memory",
        }.get(kind, "Waiting for resources")
        monitor.update_phase(f"waiting_for_{kind}", label, status="paused")
        system_update = {"resource_wait": kind}
        if kind == "ram":
            system_update.update({
                "ram_available_gb": state.get("available_gb"),
                "memory_reserve_gb": state.get("reserve_gb"),
                "memory_target_gb": state.get("target_gb"),
            })
        elif kind == "disk":
            system_update.update({
                "disk_free_gb": state.get("free_gb"),
                "disk_reserve_gb": state.get("reserve_gb"),
                "disk_target_gb": state.get("target_gb"),
            })
        elif kind == "vram":
            system_update.update({
                "gpu_free_gb": state.get("free_gb"),
                "gpu_required_free_gb": state.get("required_free_gb"),
            })
        monitor.update("system", system_update)

    def _ram_gate(
        resume_phase: str,
        resume_label: str,
        *,
        extra_required_bytes: int = 0,
    ) -> None:
        waited = False

        def on_wait(state: dict) -> None:
            nonlocal waited
            if not state.get("safe"):
                waited = True
                _wait_state("ram", state, resume_phase, resume_label)

        wait_for_ram(
            policy,
            extra_required_bytes=extra_required_bytes,
            on_wait=on_wait,
            poll_seconds=1.0,
            stable_samples=3,
        )
        if waited and monitor is not None:
            monitor.update_phase(resume_phase, resume_label, status="running")
            monitor.log("INFO", "RAM recovered; resuming.")

    def _disk_gate(
        resume_phase: str,
        resume_label: str,
        *,
        required_bytes: int,
    ) -> None:
        waited = False

        def on_wait(state: dict) -> None:
            nonlocal waited
            if not state.get("safe"):
                waited = True
                _wait_state("disk", state, resume_phase, resume_label)

        wait_for_disk(
            app_cfg.output_dir,
            policy,
            required_bytes=required_bytes,
            on_wait=on_wait,
            poll_seconds=2.0,
            stable_samples=2,
        )
        if waited and monitor is not None:
            monitor.update_phase(resume_phase, resume_label, status="running")
            monitor.log("INFO", "Disk headroom recovered; resuming.")

    def _vram_gate(
        resume_phase: str,
        resume_label: str,
        *,
        required_free_bytes: int,
    ) -> None:
        waited = False

        def on_wait(state: dict) -> None:
            nonlocal waited
            if not state.get("safe"):
                waited = True
                _wait_state("vram", state, resume_phase, resume_label)

        wait_for_vram(
            required_free_bytes=required_free_bytes,
            on_wait=on_wait,
            poll_seconds=2.0,
            stable_samples=2,
        )
        if waited and monitor is not None:
            monitor.update_phase(resume_phase, resume_label, status="running")
            monitor.log("INFO", "GPU memory recovered; resuming.")

    def _set_vision_budget(processor, pixels: int) -> None:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            return
        try:
            image_processor.size = {
                "shortest_edge": 65_536,
                "longest_edge": int(pixels),
            }
        except Exception:
            pass

    class _MonitorCallback(TrainerCallback):
        def __init__(self, runtime_monitor):
            self.runtime_monitor = runtime_monitor
            self.started = None

        def _push(self, state, **extra):
            if self.runtime_monitor is None:
                return
            if self.started is None:
                self.started = time.monotonic()
            total = int(getattr(state, "max_steps", 0) or 0)
            step = int(getattr(state, "global_step", 0) or 0)
            progress = (step / total * 100.0) if total > 0 else 0.0
            elapsed = max(0.0, time.monotonic() - self.started)
            eta = elapsed / step * (total - step) if step > 0 and total > step else None
            payload = {
                "step": step,
                "total_steps": total,
                "progress_percent": round(progress, 2),
                "epoch": getattr(state, "epoch", None),
                "elapsed_seconds": int(elapsed),
                "eta_seconds": int(eta) if eta is not None else None,
            }
            payload.update(extra)
            self.runtime_monitor.update_training(**payload)

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()
            if self.runtime_monitor is not None:
                self.runtime_monitor.update_phase("training", "Training model")
            self._push(state)

        def on_step_begin(self, args, state, control, **kwargs):
            # RAM/disk can recover by waiting. VRAM is different after the model
            # is resident: waiting on our own persistent allocations can deadlock.
            # Runtime VRAM is therefore measured after each optimizer step and
            # handled by graceful backoff/restart instead of an infinite wait.
            _ram_gate("training", "Training model")
            _disk_gate(
                "training",
                "Training model",
                required_bytes=policy.training_write_budget_bytes,
            )
            self._push(state)

        def on_step_end(self, args, state, control, **kwargs):
            gc.collect()
            torch.cuda.empty_cache()
            gpu_state = current_gpu_memory(0)
            if gpu_state is not None:
                free_bytes = int(gpu_state["free_bytes"])
                allocated = int(torch.cuda.memory_allocated(0))
                reserved = int(torch.cuda.memory_reserved(0))
                try:
                    peak_allocated = int(torch.cuda.max_memory_allocated(0))
                    peak_reserved = int(torch.cuda.max_memory_reserved(0))
                except Exception:
                    peak_allocated = allocated
                    peak_reserved = reserved

                if self.runtime_monitor is not None:
                    self.runtime_monitor.update("system", {
                        "gpu_free_gb": round(free_bytes / GIB, 2),
                        "torch_allocated_gb": round(allocated / GIB, 2),
                        "torch_reserved_gb": round(reserved / GIB, 2),
                        "torch_peak_allocated_gb": round(peak_allocated / GIB, 2),
                        "torch_peak_reserved_gb": round(peak_reserved / GIB, 2),
                    })

                if free_bytes < policy.gpu_step_reserve_bytes:
                    runtime_backoff.update({
                        "requested": True,
                        "reason": "post-step VRAM headroom below safety reserve",
                        "free_bytes": free_bytes,
                        "step": int(getattr(state, "global_step", 0) or 0),
                    })
                    # Ask Trainer to save this completed step and stop cleanly.
                    # The outer loop lowers batch footprint and resumes from the
                    # checkpoint; no process is killed.
                    control.should_save = True
                    control.should_training_stop = True
                    if self.runtime_monitor is not None:
                        self.runtime_monitor.log(
                            "WARNING",
                            "VRAM headroom below safety reserve after optimizer step; "
                            "saving checkpoint and lowering footprint before continuing.",
                            free_gb=round(free_bytes / GIB, 2),
                            reserve_gb=round(policy.gpu_step_reserve_bytes / GIB, 2),
                            step=runtime_backoff["step"],
                        )
                try:
                    torch.cuda.reset_peak_memory_stats(0)
                except Exception:
                    pass
            self._push(state)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            extra = {}
            for source, target in (
                ("loss", "loss"),
                ("learning_rate", "learning_rate"),
                ("grad_norm", "grad_norm"),
                ("epoch", "epoch"),
            ):
                if source in logs:
                    extra[target] = logs[source]
            self._push(state, **extra)
            if self.runtime_monitor is not None and logs:
                compact = {
                    k: v for k, v in logs.items()
                    if isinstance(v, (int, float, str))
                }
                self.runtime_monitor.log("INFO", "Trainer log", **compact)

        def on_save(self, args, state, control, **kwargs):
            checkpoint = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
            self._push(state, checkpoint=checkpoint)
            if self.runtime_monitor is not None:
                self.runtime_monitor.log(
                    "INFO",
                    "Checkpoint saved",
                    checkpoint=checkpoint,
                    step=int(state.global_step),
                )

        def on_train_end(self, args, state, control, **kwargs):
            self._push(state, progress_percent=100.0, eta_seconds=0)

    if monitor is not None:
        monitor.update("system", {
            **policy.as_dict(),
            "resource_policy": "admission-control-no-kill",
        })
        monitor.update("model", {"model_id": DEFAULT_BASE_MODEL_ID})
        monitor.update_phase("model_download", "Checking / downloading Qwen3.5-9B-Base")

    _disk_gate(
        "model_download",
        "Checking / downloading Qwen3.5-9B-Base",
        required_bytes=policy.model_download_budget_bytes,
    )
    model_path = ensure_base_model(app_cfg, monitor=monitor)

    if monitor is not None:
        monitor.update("model", {
            "model_id": DEFAULT_BASE_MODEL_ID,
            "local_path": str(model_path),
        })
        monitor.log("INFO", "Base model ready", path=str(model_path))

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

    records = int(dataset_meta.get("records", 0))
    if records <= 0:
        raise ValueError("Domain dataset contains no trainable records")

    # Streaming datasets do not implement __len__, so Trainer requires max_steps.
    computed_steps = max(
        1,
        math.ceil(
            records
            * float(cfg["num_train_epochs"])
            / (
                int(cfg["micro_batch_size"])
                * int(cfg["gradient_accumulation_steps"])
            )
        ),
    )
    cfg["max_steps"] = int(max_steps) if max_steps is not None else computed_steps

    output_dir = (
        app_cfg.training_dir / snapshot_id / "qwen3.5-9b-base-multimodal-rtx4080"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if monitor is not None:
        monitor.update_phase("training_prepare", "Preparing streaming training run")

    registry = Registry(app_cfg.registry_path)
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    with registry.connect() as conn:
        conn.execute(
            """INSERT INTO training_runs(
                   run_id,snapshot_id,kind,status,config_path,output_dir,started_at
               ) VALUES(?,?,'multimodal-domain','running',NULL,?,?)""",
            (run_id, snapshot_id, str(output_dir), _now()),
        )

    try:
        if monitor is not None:
            monitor.update_training(run_id=run_id, total_steps=cfg["max_steps"])

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not available. Local multimodal training requires NVIDIA CUDA."
            )

        if cfg["allow_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True

        _ram_gate(
            "model_load",
            "Loading processor and 4-bit multimodal model",
            extra_required_bytes=10 * GIB,
        )
        gpu_state = current_gpu_memory(0)
        if gpu_state is not None:
            pre_model_free = max(
                8 * GIB,
                int(gpu_state["total_bytes"]) - int(1.5 * GIB),
            )
            _vram_gate(
                "model_load",
                "Loading processor and 4-bit multimodal model",
                required_free_bytes=pre_model_free,
            )

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        if monitor is not None:
            monitor.update_phase(
                "model_load",
                "Loading processor and 4-bit multimodal model",
            )

        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=cfg["trust_remote_code"],
        )
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        _set_vision_budget(processor, cfg["vision_max_pixels"])

        while True:
            try:
                model = AutoModelForMultimodalLM.from_pretrained(
                    str(model_path),
                    quantization_config=quant,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "9GiB", "cpu": "20GiB"},
                    dtype=torch.bfloat16,
                    trust_remote_code=cfg["trust_remote_code"],
                )
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if monitor is not None:
                    monitor.update_phase(
                        "waiting_for_vram",
                        "Waiting for GPU memory before model load",
                        status="paused",
                    )
                    monitor.log(
                        "WARNING",
                        "CUDA OOM during model load; waiting and retrying without exiting.",
                    )
                gpu_state = current_gpu_memory(0)
                required = (
                    max(8 * GIB, int(gpu_state["total_bytes"]) - int(1.5 * GIB))
                    if gpu_state is not None
                    else 8 * GIB
                )
                _vram_gate(
                    "model_load",
                    "Loading processor and 4-bit multimodal model",
                    required_free_bytes=required,
                )
            except MemoryError:
                gc.collect()
                if monitor is not None:
                    monitor.update_phase(
                        "waiting_for_ram",
                        "Waiting for RAM before model load",
                        status="paused",
                    )
                _ram_gate(
                    "model_load",
                    "Loading processor and 4-bit multimodal model",
                    extra_required_bytes=10 * GIB,
                )

        model.config.use_cache = False
        _vram_gate(
            "adapter_prepare",
            "Preparing memory-safe QLoRA adapters",
            required_free_bytes=policy.gpu_step_reserve_bytes,
        )
        if monitor is not None:
            monitor.update_phase(
                "adapter_prepare",
                "Preparing memory-safe QLoRA adapters",
            )

        model = _prepare_kbit_model_memory_safe(
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
                exclude_modules=cfg["exclude_modules"],
                task_type="CAUSAL_LM",
            ),
            autocast_adapter_dtype=False,
        )

        trainable = []
        forbidden_visual_trainables = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            trainable.append(name)
            lowered = name.lower()
            if "visual" in lowered or "vision" in lowered:
                forbidden_visual_trainables.append(name)

        if not trainable:
            raise RuntimeError("LoRA adapter injection produced no trainable parameters")
        if forbidden_visual_trainables:
            raise RuntimeError(
                "Vision LoRA exclusion failed; refusing to train visual adapters on the "
                f"RTX4080 preset: {forbidden_visual_trainables[:5]}"
            )

        gc.collect()
        torch.cuda.empty_cache()
        if monitor is not None:
            monitor.update("model", {
                "trainable_parameter_tensors": len(trainable),
                "adapter_dtype_autocast": False,
                "vision_lora_excluded": True,
            })
            monitor.log(
                "INFO",
                "Memory-safe QLoRA preparation completed",
                trainable_parameter_tensors=len(trainable),
            )

        if monitor is not None:
            monitor.update_phase("dataset_stream", "Opening streaming JSONL dataset")

        # No Arrow materialization/cache: local JSONL is iterated record-by-record.
        train_ds = load_dataset(
            "json",
            data_files=shard_paths,
            split="train",
            streaming=True,
        ).filter(lambda row: bool((row.get("text") or "").strip()))

        args, dropped_training_args = _build_training_arguments(
            TrainingArguments,
            output_dir=output_dir,
            cfg=cfg,
        )
        if dropped_training_args and monitor is not None:
            monitor.log(
                "WARNING",
                "Installed Transformers omitted non-critical TrainingArguments",
                dropped=dropped_training_args,
            )

        def make_trainer():
            callbacks = [_MonitorCallback(monitor)] if monitor is not None else None
            return Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                data_collator=MultimodalDomainCollator(processor, cfg),
                callbacks=callbacks,
            )

        oom_level = 0
        oom_pixel_budgets = [196_608, 131_072, 98_304, 65_536]
        oom_text_lengths = [1024, 768, 512, 384]
        oom_visual_chars = [2000, 1500, 1000, 750]

        while True:
            runtime_backoff.update({
                "requested": False,
                "reason": None,
                "free_bytes": None,
                "step": None,
            })
            trainer = make_trainer()
            checkpoint = get_last_checkpoint(str(output_dir))
            if monitor is not None:
                monitor.update_training(checkpoint=checkpoint, run_id=run_id)
                monitor.update_phase("training", "Training model")
            try:
                trainer.train(resume_from_checkpoint=checkpoint)
                if runtime_backoff["requested"]:
                    try:
                        model.zero_grad(set_to_none=True)
                    except Exception:
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()

                    next_level = min(
                        oom_level + 1,
                        len(oom_pixel_budgets) - 1,
                    )
                    changed = next_level != oom_level
                    oom_level = next_level
                    cfg["vision_max_pixels"] = oom_pixel_budgets[oom_level]
                    cfg["text_max_length"] = oom_text_lengths[oom_level]
                    cfg["visual_text_max_chars"] = oom_visual_chars[oom_level]
                    _set_vision_budget(processor, cfg["vision_max_pixels"])

                    if monitor is not None:
                        monitor.update_phase(
                            "vram_backoff",
                            "Reducing training footprint after low VRAM headroom",
                            status="paused",
                        )
                        monitor.log(
                            "WARNING",
                            "Graceful VRAM backoff requested after a completed step.",
                            changed=changed,
                            oom_level=oom_level,
                            vision_max_pixels=cfg["vision_max_pixels"],
                            text_max_length=cfg["text_max_length"],
                            free_gb=(
                                round(int(runtime_backoff["free_bytes"]) / GIB, 2)
                                if runtime_backoff["free_bytes"] is not None
                                else None
                            ),
                        )
                    # The callback requested a checkpoint before stopping. Reuse it
                    # immediately on the next loop with the smaller footprint.
                    time.sleep(1.0)
                    continue
                break
            except torch.cuda.OutOfMemoryError:
                try:
                    model.zero_grad(set_to_none=True)
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

                oom_level = min(oom_level + 1, len(oom_pixel_budgets) - 1)
                cfg["vision_max_pixels"] = oom_pixel_budgets[oom_level]
                cfg["text_max_length"] = oom_text_lengths[oom_level]
                cfg["visual_text_max_chars"] = oom_visual_chars[oom_level]
                _set_vision_budget(processor, cfg["vision_max_pixels"])

                if monitor is not None:
                    monitor.update_phase(
                        "waiting_for_vram",
                        "CUDA OOM: lowering batch footprint and waiting",
                        status="paused",
                    )
                    monitor.log(
                        "WARNING",
                        "CUDA OOM caught; no process exit. Lowering visual/text budget and retrying.",
                        oom_level=oom_level,
                        vision_max_pixels=cfg["vision_max_pixels"],
                        text_max_length=cfg["text_max_length"],
                    )
                _vram_gate(
                    "training",
                    "Training model",
                    required_free_bytes=policy.gpu_step_reserve_bytes,
                )
                time.sleep(1.0)

        _disk_gate(
            "saving",
            "Saving final adapter and processor",
            required_bytes=policy.training_write_budget_bytes,
        )
        if monitor is not None:
            monitor.update_phase("saving", "Saving final adapter and processor")
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
                    "preset": "rtx4080_16gb_qwen35_multimodal_v4_vram_headroom",
                    "preset_values": cfg,
                    "smoke_test": max_steps is not None,
                    "streaming_dataset": True,
                    "vision_base_frozen": True,
                    "resource_policy": policy.as_dict(),
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
        if monitor is not None:
            monitor.update_training(
                progress_percent=100.0,
                eta_seconds=0,
                run_id=run_id,
            )
            monitor.log("INFO", "Training run completed", run_id=run_id)
        return run_id

    except Exception as exc:
        if monitor is not None:
            monitor.set_error(exc)
        with registry.connect() as conn:
            conn.execute(
                "UPDATE training_runs SET status='failed',ended_at=? WHERE run_id=?",
                (_now(), run_id),
            )
        raise
