from __future__ import annotations

import gc
import os
import threading
from pathlib import Path

from PIL import Image

from .resolver import AdapterBundle, discover_latest_adapter


SYSTEM_TEXT = """You are Oracle-Lite Talker, a local multimodal domain assistant.
Use the attached source material and conversation context when relevant.
Do not invent facts that are not supported by the conversation, attachments,
or the model's learned domain knowledge. Clearly say when evidence is missing.
For mechanical/CAD material, preserve dimensions, units, identifiers and
technical terminology exactly when they are present.
"""

MAX_HISTORY_MESSAGES = 10
MAX_CONTEXT_CHARS = 14_000
MAX_IMAGES = 2
VISION_PIXELS = 196_608


class TalkerModel:
    def __init__(self, training_output_dir: str | Path):
        self.training_output_dir = Path(training_output_dir).resolve()
        self.bundle: AdapterBundle | None = None
        self.model = None
        self.processor = None
        self.torch = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> AdapterBundle:
        os.environ.setdefault(
            "PYTORCH_ALLOC_CONF",
            "expandable_segments:True,garbage_collection_threshold:0.80",
        )
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Talker requires an NVIDIA CUDA GPU")

        bundle = discover_latest_adapter(self.training_output_dir)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(
            str(bundle.base_model_path),
            trust_remote_code=False,
        )
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            try:
                image_processor.size = {
                    "shortest_edge": 65_536,
                    "longest_edge": VISION_PIXELS,
                }
            except Exception:
                pass

        base_model = AutoModelForMultimodalLM.from_pretrained(
            str(bundle.base_model_path),
            quantization_config=quant,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "10GiB", "cpu": "20GiB"},
            dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        base_model.config.use_cache = True
        model = PeftModel.from_pretrained(
            base_model,
            str(bundle.adapter_dir),
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        self.bundle = bundle
        self.model = model
        self.processor = processor
        self.torch = torch
        return bundle

    def info(self) -> dict:
        bundle = self.bundle
        return {
            "loaded": self.loaded,
            "adapter_dir": str(bundle.adapter_dir) if bundle else None,
            "base_model_path": str(bundle.base_model_path) if bundle else None,
            "adapter_kind": bundle.source_kind if bundle else None,
        }

    def _image_tokens(self, count: int) -> str:
        processor = self.processor
        vision_start = getattr(processor, "vision_start_token", "<|vision_start|>")
        image_token = getattr(processor, "image_token", "<|image_pad|>")
        vision_end = getattr(processor, "vision_end_token", "<|vision_end|>")
        return "".join(
            f"{vision_start}{image_token}{vision_end}\n"
            for _ in range(count)
        )

    def _context(self, messages: list[dict], max_chars: int) -> tuple[str, list[str]]:
        selected = messages[-MAX_HISTORY_MESSAGES:]
        pieces: list[str] = []
        total = 0

        # Keep conversation text in chronological order, but select images from
        # newest to oldest so a follow-up refers to the most recent visual input.
        newest_images: list[str] = []
        for message in reversed(selected):
            for attachment in reversed(message.get("attachments", [])):
                for image_path in reversed(attachment.get("images", [])):
                    if Path(image_path).exists() and image_path not in newest_images:
                        newest_images.append(image_path)
                        if len(newest_images) >= MAX_IMAGES:
                            break
                if len(newest_images) >= MAX_IMAGES:
                    break
            if len(newest_images) >= MAX_IMAGES:
                break
        images = list(reversed(newest_images))

        for message in selected:
            role = "User" if message["role"] == "user" else "Assistant"
            content = (message.get("content") or "").strip()
            attachment_parts: list[str] = []
            for attachment in message.get("attachments", []):
                parsed = (attachment.get("parsed_text") or "").strip()
                if parsed:
                    attachment_parts.append(
                        f"[Attachment: {attachment['filename']}]\n{parsed[:4000]}"
                    )

            block = f"{role}: {content}"
            if attachment_parts:
                block += "\n" + "\n".join(attachment_parts)
            remaining = max_chars - total
            if remaining <= 0:
                break
            block = block[:remaining]
            pieces.append(block)
            total += len(block)

        return "\n\n".join(pieces), images

    def _input_device(self):
        torch = self.torch
        device_map = getattr(self.model, "hf_device_map", None) or {}
        for value in device_map.values():
            text = str(value)
            if text.startswith("cuda"):
                return torch.device(text)
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
        for parameter in self.model.parameters():
            if parameter.device.type == "cuda":
                return parameter.device
        return torch.device("cuda:0")

    def _generate_once(
        self,
        messages: list[dict],
        *,
        max_context_chars: int,
        max_new_tokens: int,
        max_images: int,
    ) -> str:
        torch = self.torch
        context, image_paths = self._context(messages, max_context_chars)
        image_paths = image_paths[-max_images:] if max_images else []

        prompt = (
            self._image_tokens(len(image_paths))
            + SYSTEM_TEXT
            + "\n\nConversation:\n"
            + context
            + "\n\nAssistant:"
        )

        images = []
        try:
            for path in image_paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))

            processor_kwargs = {
                "text": [prompt],
                "padding": False,
                "return_tensors": "pt",
            }
            if images:
                processor_kwargs["images"] = images
            inputs = self.processor(**processor_kwargs)

            device = self._input_device()
            for key, value in list(inputs.items()):
                if hasattr(value, "to"):
                    inputs[key] = value.to(device)

            input_length = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.25,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            generated = output[:, input_length:]
            text = self.processor.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0].strip()
            return text or "(The model returned an empty response.)"
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def answer(self, messages: list[dict]) -> str:
        if not self.loaded:
            raise RuntimeError("Talker model is not loaded")

        with self._lock:
            try:
                return self._generate_once(
                    messages,
                    max_context_chars=MAX_CONTEXT_CHARS,
                    max_new_tokens=384,
                    max_images=MAX_IMAGES,
                )
            except self.torch.cuda.OutOfMemoryError:
                gc.collect()
                self.torch.cuda.empty_cache()
                return self._generate_once(
                    messages,
                    max_context_chars=7_000,
                    max_new_tokens=192,
                    max_images=1,
                )
