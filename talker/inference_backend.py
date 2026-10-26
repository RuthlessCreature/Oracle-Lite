"""OpenAI-compatible inference backends used by Talker and Oracle Viewer.

The application owns prompts, evidence retrieval and validation. This module
owns only the wire protocol, endpoint health check and backend identity so a
local Strata or vLLM server can be swapped without changing those workflows.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceResult:
    text: str
    truncated: bool
    generated_tokens: int


class OpenAICompatibleBackend:
    def __init__(self, base_url: str, model: str, *, label: str,
                 api_key: str = ""):
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.model = str(model or "").strip()
        self.label = str(label or "openai-compatible").strip()
        self.api_key = str(api_key or "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def identity(self) -> dict:
        return {"inference_backend": self.label, "model_alias": self.model,
                "endpoint": self.base_url}

    def _request(self, path: str, *, method: str = "GET", payload: dict | None = None,
                 timeout: float = 10):
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base_url + path, data=data,
                                         headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}

    def ready(self, *, timeout: float = 5) -> bool:
        if not self.configured:
            return False
        # vLLM, Strata and compatible servers expose /health. A few providers
        # expose only /v1/models, so use that as a protocol-level fallback.
        try:
            self._request("/health", timeout=timeout)
            return True
        except (OSError, ValueError, urllib.error.URLError, TimeoutError):
            try:
                payload = self._request("/v1/models", timeout=timeout)
            except (OSError, ValueError, urllib.error.URLError, TimeoutError):
                return False
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            return any(isinstance(row, dict) and row.get("id") == self.model
                       for row in rows)

    def generate(self, context: str, image_paths: list[str], *, max_tokens: int,
                 temperature: float, system_text: str,
                 assistant_prefix: str = "") -> InferenceResult:
        if assistant_prefix:
            context += ("\n\nContinue the prior answer from this continuation context. "
                        "Do not repeat it:\n" + assistant_prefix)
        content: list[dict] = [{"type": "text", "text": context}]
        for image_path in image_paths:
            path = Path(image_path)
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{encoded}"}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if self.label == "strata-native":
            payload["repeat_penalty"] = 1.05
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            # vLLM follows the OpenAI parameter spelling. Qwen templates accept
            # this switch; keeping it in extra_body avoids changing prompts.
            payload["repetition_penalty"] = 1.05
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        result = self._request("/v1/chat/completions", method="POST",
                               payload=payload, timeout=600)
        choice = result["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content") or message.get("reasoning_content") or ""
        usage = result.get("usage") or {}
        return InferenceResult(str(text), choice.get("finish_reason") == "length",
                               int(usage.get("completion_tokens") or 0))
