from __future__ import annotations

import json
import time
from typing import Any

import requests

from .config import AppConfig


MINIMAX_M3_MODEL = "MiniMax-M3"
MINIMAX_CHAT_URL = "https://api.minimaxi.com/v1/text/chatcompletion_v2"

JANITOR_SYSTEM_PROMPT = """You are Oracle-Lite Data Janitor.
Your job is limited to low-risk data chores: classification, metadata extraction,
format cleanup, title/section recovery, and paraphrasing of questions when the
answer is already fixed by source data.

Rules:
1. Never invent or supplement domain facts.
2. Never change numerical values, units, part numbers, alarm codes, versions, dates,
   or causal claims from the supplied source.
3. If the source is insufficient, return an explicit insufficient_source flag.
4. When JSON is requested, return valid JSON only.
"""


class MiniMaxError(RuntimeError):
    pass


class MiniMaxM3Client:
    """Thin Token-Plan-compatible client for low-risk Data Janitor tasks."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        timeout_seconds: int = 120,
        max_retries: int = 4,
    ):
        self.api_key = cfg.minimax_api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def complete(self, prompt: str, *, system_prompt: str = JANITOR_SYSTEM_PROMPT) -> str:
        payload = {
            "model": MINIMAX_M3_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    MINIMAX_CHAT_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise MiniMaxError(
                        f"retryable MiniMax HTTP {response.status_code}: {response.text[:500]}"
                    )
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise MiniMaxError(f"MiniMax response has no choices: {body}")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise MiniMaxError(f"MiniMax response has no text content: {body}")
                return content.strip()
            except (requests.RequestException, ValueError, MiniMaxError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))

        raise MiniMaxError(f"MiniMax request failed after retries: {last_error}")

    def json_task(self, prompt: str) -> dict[str, Any]:
        text = self.complete(
            prompt
            + "\n\nReturn one JSON object only. Do not wrap it in Markdown fences."
        )
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MiniMaxError(f"MiniMax returned invalid JSON: {text[:1000]}") from exc
        if not isinstance(value, dict):
            raise MiniMaxError("MiniMax JSON task must return one object")
        return value
