from __future__ import annotations

import gc
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from oracle_lite.grounding import SYSTEM_TEXT, render_processor_prompt

from .resolver import AdapterBundle, discover_latest_adapter
from .retrieval import CorpusRetriever, query_requests_complete_list
from .engineering import ENGINEERING_INSTRUCTION, is_design_request, retrieval_query
from .engineering_plan import source_checked_circle_plan
from .inference_backend import OpenAICompatibleBackend


MAX_HISTORY_MESSAGES = 10
MAX_CONTEXT_CHARS = 18_000
RETRIEVAL_CONTEXT_CHARS = 12_000
MAX_IMAGES = 3
VISION_PIXELS = 786_432
_SCOPE_REVIEW_RE = re.compile(
    r"\b(?:datum|probe|program|step|version|revision|procedure|station|source)\b"
    r"|基准|探针|程序|步骤|版本|修订|工序", re.I,
)
_NUMBERED_SPEC_RE = re.compile(
    r"\b(?:FAI|item|feature|dimension|point)\s*#?\s*[A-Z]?\d+\b",
    re.I,
)
_SPEC_VALUE_RE = re.compile(
    r"\b(?:value|nominal|tolerance|limit|measurement)\b|数值|公差|限值|测量值",
    re.I,
)

# Long answers are generated in bounded chunks. This avoids a single huge
# generation request while preventing the old 384-token hard truncation.
GENERATION_CHUNK_TOKENS = 768
GENERATION_FALLBACK_CHUNK_TOKENS = 384
CONTINUATION_TAIL_CHARS = 8_000
_VISION_CAPABILITY_LOCK = threading.Lock()
_VISION_CAPABILITY_CACHE: dict[tuple[str, str], tuple[float, bool]] = {}


def _local_vision_model_ready(endpoint: str, model_name: str) -> bool:
    """Report vision only when the configured loopback model is serving now.

    An environment variable alone is not evidence that the local visual model
    is available. Viewer FAI completion must not silently present a text-only
    Strata service as capable of reading MSOP page images.
    """
    endpoint = (endpoint or '').strip().rstrip('/')
    model_name = (model_name or '').strip()
    if not endpoint or not model_name:
        return False
    try:
        parsed = urlparse(endpoint)
        if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            return False
    except ValueError:
        return False
    key = (endpoint, model_name)
    now = time.monotonic()
    with _VISION_CAPABILITY_LOCK:
        cached = _VISION_CAPABILITY_CACHE.get(key)
        if cached and now - cached[0] < 5:
            return cached[1]
    ready = False
    try:
        request = urllib.request.Request(endpoint + '/v1/models', headers={'Accept': 'application/json'})
        with urllib.request.urlopen(request, timeout=0.75) as response:
            payload = json.loads(response.read().decode('utf-8'))
        models = payload.get('data', []) if isinstance(payload, dict) else []
        ready = any(isinstance(row, dict) and row.get('id') == model_name for row in models)
    except (OSError, ValueError, urllib.error.URLError, TimeoutError):
        ready = False
    with _VISION_CAPABILITY_LOCK:
        _VISION_CAPABILITY_CACHE[key] = (now, ready)
    return ready


def _strata_native_vision_ready(endpoint: str, model_name: str) -> bool:
    """Check that this exact local Strata model has its native image encoder loaded."""
    endpoint = (endpoint or '').strip().rstrip('/')
    model_name = (model_name or '').strip()
    try:
        parsed = urlparse(endpoint)
        if parsed.scheme not in ('http', 'https') or parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            return False
        root = endpoint[:-3] if endpoint.endswith('/v1') else endpoint
    except (ValueError, TypeError):
        return False
    key = (root + '/health', model_name)
    now = time.monotonic()
    with _VISION_CAPABILITY_LOCK:
        cached = _VISION_CAPABILITY_CACHE.get(key)
        if cached and now - cached[0] < 5:
            return cached[1]
    ready = False
    try:
        request = urllib.request.Request(root + '/health', headers={'Accept': 'application/json'})
        with urllib.request.urlopen(request, timeout=0.75) as response:
            payload = json.loads(response.read().decode('utf-8'))
        ready = (isinstance(payload, dict) and payload.get('status') == 'ok'
                 and payload.get('loaded') is True and payload.get('images') is True
                 and payload.get('model') == model_name)
    except (OSError, ValueError, urllib.error.URLError, TimeoutError):
        ready = False
    with _VISION_CAPABILITY_LOCK:
        _VISION_CAPABILITY_CACHE[key] = (now, ready)
    return ready


class TalkerModel:
    def __init__(
        self,
        training_output_dir: str | Path,
        *,
        openai_base_url: str | None = None,
        openai_model: str | None = None,
        backend_label: str | None = None,
        openai_api_key: str | None = None,
    ):
        self.training_output_dir = Path(training_output_dir).resolve()
        base_url = (os.environ.get("ORACLE_TALKER_OPENAI_BASE_URL", "")
                    if openai_base_url is None else openai_base_url)
        model_name = (os.environ.get("ORACLE_TALKER_OPENAI_MODEL", "oracle-qwen35-adapted")
                      if openai_model is None else openai_model)
        backend_name = (os.environ.get("ORACLE_TALKER_INFERENCE_BACKEND", "openai-compatible")
                        if backend_label is None else backend_label)
        self.openai_base_url = base_url.strip().rstrip("/")
        self.openai_model = model_name.strip()
        self.openai_backend_label = backend_name.strip() or "openai-compatible"
        self.inference_backend = (OpenAICompatibleBackend(
            self.openai_base_url, self.openai_model,
            label=self.openai_backend_label,
            api_key=(os.environ.get("ORACLE_TALKER_OPENAI_API_KEY", "")
                     if openai_api_key is None else openai_api_key),
        ) if self.openai_base_url else None)
        self._openai_ready = False
        self.retriever = CorpusRetriever(
            self.training_output_dir,
            self.training_output_dir / "_talker" / "corpus_fts.sqlite3",
        )
        self.bundle: AdapterBundle | None = None
        self.model = None
        self.processor = None
        self.torch = None
        # Viewer marking holds the shared model lock across a complete FAI
        # review, while nested helpers also guard individual inference calls.
        # A regular Lock deadlocks when those helpers re-enter it on the same
        # thread; RLock preserves serialization while allowing that nesting.
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        if self.openai_base_url:
            return self._openai_ready
        return self.model is not None and self.processor is not None

    def load(self) -> AdapterBundle:
        try:
            bundle = discover_latest_adapter(self.training_output_dir)
        except (FileNotFoundError, RuntimeError, ValueError):
            if not self.openai_base_url:
                raise
            # A remote/self-hosted inference backend does not need a local
            # training checkpoint. Corpus retrieval can still use all indexed
            # local source data without binding it to one adapter run.
            bundle = None
        corpus_id = next((parent.name for parent in bundle.run_dir.parents
                          if parent.parent.name == "_corpora"), None) if bundle else None
        self.retriever.select_corpus(corpus_id)
        self.bundle = bundle

        # A self-hosted backend owns model weights and decoding. Talker keeps
        # local retrieval, prompts, and engineering evidence validation.
        if self.openai_base_url:
            if not self.inference_backend or not self.inference_backend.ready(timeout=5):
                raise RuntimeError(
                    f"Configured model endpoint is unavailable at {self.openai_base_url}"
                )
            self._openai_ready = True
            return bundle

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
            "adapter_dir": str(bundle.adapter_dir) if bundle and not self.openai_base_url else None,
            "base_model_path": str(bundle.base_model_path) if bundle and not self.openai_base_url else None,
            "adapter_kind": "remote-model" if self.openai_base_url else (bundle.source_kind if bundle else None),
            "adapter_source_kind": None if self.openai_base_url else (bundle.source_kind if bundle else None),
            "adapter_loaded_separately": bool(bundle and not self.openai_base_url),
            "inference_backend": self.openai_backend_label if self.openai_base_url else "transformers-peft",
            "model_alias": self.openai_model if self.openai_base_url else None,
            "generation_chunk_tokens": GENERATION_CHUNK_TOKENS,
            "generation_total_tokens": None,
            "generation_mode": "unbounded-until-complete",
            "corpus_retrieval": "local-fts5",
            "engineering_drafts": "source-checked-msop-circle-with-calculator",
            "image_inference": ('strata_native_vision' if self.openai_backend_label == 'strata-native' and
                _strata_native_vision_ready(self.openai_base_url, self.openai_model) else
                ('local_auxiliary_vision' if self.openai_backend_label == 'strata-native' and
                 _local_vision_model_ready(os.environ.get('ORACLE_LOCAL_VISION_BASE_URL', ''),
                                           os.environ.get('ORACLE_LOCAL_VISION_MODEL', '')) else
                 ('unsupported_by_native_text_model' if self.openai_backend_label == 'strata-native'
                  else 'configured_multimodal_backend'))),
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

    def _context(self, messages: list[dict], max_chars: int,
                 max_images: int = MAX_IMAGES) -> tuple[str, list[str]]:
        selected = messages[-MAX_HISTORY_MESSAGES:]

        newest_images: list[str] = []
        image_limit=max(0,int(max_images))
        image_messages=reversed(selected) if image_limit else ()
        for message in image_messages:
            for attachment in reversed(message.get("attachments", [])):
                for image_path in reversed(attachment.get("images", [])):
                    if Path(image_path).exists() and image_path not in newest_images:
                        newest_images.append(image_path)
                        if len(newest_images) >= image_limit:
                            break
                if len(newest_images) >= image_limit:
                    break
            if len(newest_images) >= image_limit:
                break
        images = list(reversed(newest_images))

        # Budget text newest-first so an old, large attachment never pushes the
        # current user request out of the prompt.
        kept_reversed: list[str] = []
        total = 0
        for message in reversed(selected):
            role = "User" if message["role"] == "user" else "Assistant"
            content = (message.get("content") or "").strip()
            attachment_parts: list[str] = []
            for attachment in message.get("attachments", []):
                parsed = (attachment.get("parsed_text") or "").strip()
                if parsed:
                    attachment_parts.append(
                        f"[Attachment: {attachment['filename']}]\n{parsed}"
                    )

            block = f"{role}: {content}"
            if attachment_parts:
                block += "\n" + "\n".join(attachment_parts)

            remaining = max_chars - total
            if remaining <= 0:
                break
            if len(block) > remaining:
                heading = f"{role}: {content}"
                notice = "\n[Attachment excerpt incomplete: remaining content was not supplied.]"
                if len(heading) + len(notice) >= remaining:
                    block = heading[:remaining]
                else:
                    attachments_text = "\n".join(attachment_parts)
                    budget = remaining - len(heading) - len(notice) - 1
                    boundary = attachments_text.rfind("\n", 0, budget + 1)
                    excerpt = attachments_text[:boundary] if boundary >= 0 else ""
                    block = heading + "\n" + excerpt + notice
            kept_reversed.append(block)
            total += len(block)

        return "\n\n".join(reversed(kept_reversed)), images

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

    def _eos_ids(self) -> set[int]:
        value = self.processor.tokenizer.eos_token_id
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            return {int(item) for item in value}
        return {int(value)}

    @staticmethod
    def _normalized_chunk(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _trim_overlap(existing: str, new_text: str) -> str:
        """Remove a repeated prefix when a continuation restates its tail."""
        if not existing or not new_text:
            return new_text
        max_overlap = min(len(existing), len(new_text), 2_000)
        for size in range(max_overlap, 19, -1):
            if existing[-size:] == new_text[:size]:
                return new_text[size:]
        return new_text

    def _continuation_prefix(self, complete_answer: str) -> str:
        if not complete_answer:
            return ""
        tail = complete_answer[-CONTINUATION_TAIL_CHARS:]
        return (
            "\n\n[CONTINUATION CONTEXT — do not repeat earlier content. "
            "Continue exactly after the final item/statement below until the "
            "user's request is fully complete.]\n"
            + tail
        )

    def _generate_once(
        self,
        messages: list[dict],
        *,
        max_context_chars: int,
        max_new_tokens: int,
        max_images: int,
        assistant_prefix: str = "",
        temperature: float = 0.0,
    ) -> tuple[str, bool, int]:
        """Generate one bounded chunk and report stop reason plus token count."""
        torch = self.torch
        context, image_paths = self._context(messages, max_context_chars,max_images=max_images)
        image_paths = image_paths[-max_images:] if max_images else []

        if self.openai_base_url:
            return self._generate_openai(context, image_paths, max_new_tokens, assistant_prefix,
                                         temperature=temperature)

        prompt = render_processor_prompt(
            self.processor, context, image_count=len(image_paths),
            image_prefix=self._image_tokens(len(image_paths)),
            assistant_prefix=assistant_prefix,
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
                sampling_kwargs=({'do_sample':True,'temperature':float(temperature)}
                                 if temperature>0 else {'do_sample':False})
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **sampling_kwargs,
                    repetition_penalty=1.05,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            generated = output[:, input_length:]
            generated_count = int(generated.shape[1])
            text = self.processor.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0]

            last_token = (
                int(generated[0, -1].item())
                if generated_count > 0
                else None
            )
            hit_limit = (
                generated_count >= max_new_tokens
                and (last_token is None or last_token not in self._eos_ids())
            )
            return text, hit_limit, generated_count
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_openai(
        self,
        context: str,
        image_paths: list[str],
        max_new_tokens: int,
        assistant_prefix: str,
        *,
        temperature: float = 0.0,
    ) -> tuple[str, bool, int]:
        """Run one generation chunk through the configured compatible backend."""
        endpoint=self.openai_base_url
        if image_paths and self.openai_backend_label == 'strata-native':
            if not _strata_native_vision_ready(self.openai_base_url, self.openai_model):
                endpoint = os.environ.get('ORACLE_LOCAL_VISION_BASE_URL', '').rstrip('/')
                model_name = os.environ.get('ORACLE_LOCAL_VISION_MODEL', '')
                if not _local_vision_model_ready(endpoint, model_name):
                    raise RuntimeError('当前本地 Strata 未加载视觉编码器，不能把图像当作已读取；请启用 Strata GPU vision')
                payload['model'] = model_name
            else:
                endpoint = self.openai_base_url
        if not self.inference_backend:
            raise RuntimeError("No OpenAI-compatible inference backend is configured")
        backend = self.inference_backend
        if endpoint != self.openai_base_url:
            # Strata's optional auxiliary vision model is a separate backend;
            # route this request to it without changing the primary model.
            backend = OpenAICompatibleBackend(
                endpoint, os.environ.get('ORACLE_LOCAL_VISION_MODEL', ''),
                label='openai-compatible',
            )
        result = backend.generate(context, image_paths, max_tokens=max_new_tokens,
                                  temperature=temperature, system_text=SYSTEM_TEXT,
                                  assistant_prefix=assistant_prefix)
        return result.text, result.truncated, result.generated_tokens

    def _answer_with_auto_continue(
        self,
        messages: list[dict],
        *,
        max_context_chars: int,
        chunk_tokens: int,
        max_images: int,
    ) -> str:
        """Continue chunk-by-chunk until EOS/completion, with no total token cap.

        The accumulated answer lives outside the model prompt. Only a rolling
        tail is sent back for continuity, so answer length itself does not make
        each subsequent inference request grow without bound.
        """
        answer = ""
        seen_chunks: set[str] = set()

        while True:
            prefix = self._continuation_prefix(answer)
            chunk, hit_limit, _generated_count = self._generate_once(
                messages,
                max_context_chars=max_context_chars,
                max_new_tokens=chunk_tokens,
                max_images=max_images,
                assistant_prefix=prefix,
            )
            if not chunk:
                break

            chunk = self._trim_overlap(answer, chunk)
            normalized = self._normalized_chunk(chunk)
            if not normalized:
                break

            # Content-level runaway protection, not a length limit. If the model
            # loops and emits the same continuation again, stop rather than
            # generating forever.
            if normalized in seen_chunks:
                break
            seen_chunks.add(normalized)

            previous = answer
            answer += chunk
            if answer == previous:
                break

            if not hit_limit:
                break

        # Base models sometimes emit reasoning tags despite direct-answer
        # instructions. Keep tagged reasoning out of the returned answer.
        answer = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", answer, flags=re.I | re.S)
        answer = re.sub(r"<think\b[^>]*>.*$", "", answer, flags=re.I | re.S).strip()
        if not answer:
            return "(The model returned an empty response.)"
        return answer

    @staticmethod
    def _needs_scope_review(query: str, retrieved_context: str) -> bool:
        """Review only narrow scope questions with multiple cited passages."""
        scoped_question = bool(_SCOPE_REVIEW_RE.search(query))
        numbered_spec_question = bool(
            _NUMBERED_SPEC_RE.search(query) and _SPEC_VALUE_RE.search(query)
        )
        if (query_requests_complete_list(query) and not numbered_spec_question) or not (
            scoped_question or numbered_spec_question
        ):
            return False
        return len(re.findall(r"\[Source\s+\d+:", retrieved_context)) >= 2

    def answer(self, messages: list[dict]) -> str:
        if not self.loaded:
            raise RuntimeError("Talker model is not loaded")

        with self._lock:
            recent_user_queries = [
                (message.get("content") or "").strip()
                for message in messages[-MAX_HISTORY_MESSAGES:]
                if message.get("role") == "user" and message.get("content")
            ]
            latest_user_query = recent_user_queries[-1] if recent_user_queries else ""
            query = retrieval_query(recent_user_queries)[-4_000:]
            prior_user_query = recent_user_queries[-2] if len(recent_user_queries) > 1 else ""
            curated = self.retriever.curated_context(
                latest_user_query,
                context_query=prior_user_query,
            )
            if curated and len(curated) > RETRIEVAL_CONTEXT_CHARS:
                curated = None
            retrieved_context = self.retriever.context(
                query, max_chars=RETRIEVAL_CONTEXT_CHARS - (len(curated) + 2 if curated else 0),
            )
            if curated:
                retrieved_context = curated + "\n\n" + retrieved_context
            if is_design_request(latest_user_query):
                checked_plan = source_checked_circle_plan(query, retrieved_context)
                if checked_plan:
                    return checked_plan
            prompt_messages = [dict(message) for message in messages]
            if is_design_request(latest_user_query):
                latest = next(message for message in reversed(prompt_messages)
                              if message.get("role") == "user")
                latest["content"] = (latest.get("content") or "") + "\n\n" + ENGINEERING_INSTRUCTION
            if retrieved_context:
                latest_user = next(
                    (
                        message
                        for message in reversed(prompt_messages)
                        if message.get("role") == "user"
                    ),
                    None,
                )
                if latest_user is not None:
                    if query_requests_complete_list(latest_user.get("content") or ""):
                        latest_user["content"] = (
                            (latest_user.get("content") or "").rstrip()
                            + "\n\nFor this request, include every directly matching source row or callout "
                            "within the user's requested scope, with each item's own values and limits. "
                            "Do not give only a representative example, infer identifiers from sequence or "
                            "proximity, or calculate totals unless asked. Preserve the requested output format."
                        )
                    latest_user["attachments"] = list(
                        latest_user.get("attachments", [])
                    ) + [{
                        "filename": "Retrieved corpus passages",
                        "parsed_text": retrieved_context,
                        "images": [],
                    }]
                    # Preserve user-supplied image priority. Otherwise give the
                    # VLM the retrieved original pages, not just flattened text.
                    has_user_images = any(
                        attachment.get("images")
                        for message in messages[-MAX_HISTORY_MESSAGES:]
                        for attachment in message.get("attachments", [])
                    )
                    if not has_user_images:
                        visual_context = getattr(self.retriever, "visual_context", None)
                        if visual_context is not None:
                            latest_user["attachments"].extend(visual_context(query, limit=MAX_IMAGES))
            try:
                def generate_answer(max_context_chars, chunk_tokens, max_images):
                    draft = self._answer_with_auto_continue(
                        prompt_messages,
                        max_context_chars=max_context_chars,
                        chunk_tokens=chunk_tokens,
                        max_images=max_images,
                    )
                    if is_design_request(latest_user_query) or not retrieved_context or not self._needs_scope_review(query, retrieved_context):
                        return draft
                    review_messages = [dict(message) for message in prompt_messages]
                    review_instruction = (
                        "Recheck the draft against the attached original source passages for the "
                        "requested field. If multiple pages, steps, or versions report different "
                        "values, list each exact value with its source scope; do not choose one as "
                        "canonical. If passages agree, keep the shared value. If a field is not "
                        "explicit for one scope, say it is not stated. Remove unsupported pairings. "
                        "Preserve the user's language and output format, and cite the supplied labels."
                    )
                    if _NUMBERED_SPEC_RE.search(query) and _SPEC_VALUE_RE.search(query):
                        review_instruction += (
                            " For this numbered specification item, distinguish its nominal value "
                            "from tolerance or limits and include both when the source states them."
                        )
                    review_messages.extend([
                        {"role": "assistant", "content": draft},
                        {"role": "user", "content": review_instruction},
                    ])
                    return self._answer_with_auto_continue(
                        review_messages,
                        max_context_chars=max_context_chars,
                        chunk_tokens=chunk_tokens,
                        max_images=max_images,
                    )

                return generate_answer(MAX_CONTEXT_CHARS, GENERATION_CHUNK_TOKENS, MAX_IMAGES)
            except self.torch.cuda.OutOfMemoryError:
                gc.collect()
                self.torch.cuda.empty_cache()
                return generate_answer(7_000, GENERATION_FALLBACK_CHUNK_TOKENS, 1)
