"""Thin LLM client wrapping Hugging Face Inference Providers (OpenAI-compatible
chat completions) with structured-JSON extraction + retry.

The four agents (§3) never write or execute code — they only produce
structured JSON that this client validates against a pydantic schema
before handing it back. All agent modules depend on the `LLMClient`
protocol rather than this concrete class, so tests can inject a fake.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, TypeVar

from huggingface_hub import InferenceClient
from pydantic import BaseModel, ValidationError

from .config import LLMConfig, LLM_CONFIG

T = TypeVar("T", bound=BaseModel)


class LLMNotConfiguredError(RuntimeError):
    pass


class LLMJSONError(RuntimeError):
    pass


class LLMClient(Protocol):
    def complete_json(
        self, system_prompt: str, user_prompt: str, schema: type[T], *, max_retries: int = 2
    ) -> T: ...


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise LLMJSONError(f"Could not find valid JSON in model output: {text[:200]!r}")


class HFLLMClient:
    """Concrete LLMClient backed by huggingface_hub's InferenceClient."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLM_CONFIG
        if not self.config.api_key:
            raise LLMNotConfiguredError(
                "HF_TOKEN is not set. Export HF_TOKEN=<your Hugging Face access token> "
                "before running agents that call the LLM."
            )
        self._client = InferenceClient(token=self.config.api_key, provider=self.config.provider)

    def complete_json(
        self, system_prompt: str, user_prompt: str, schema: type[T], *, max_retries: int = 2
    ) -> T:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            content = response.choices[0].message.content or ""
            try:
                data = _extract_json(content)
                return schema.model_validate(data)
            except (LLMJSONError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That response was invalid: {exc}. "
                            "Reply with ONLY valid JSON matching the required schema — "
                            "no prose, no markdown code fences."
                        ),
                    }
                )

        raise LLMJSONError(f"Failed to get valid JSON after {max_retries + 1} attempts: {last_error}")


_default_client: HFLLMClient | None = None


def get_default_client() -> HFLLMClient:
    global _default_client
    if _default_client is None:
        _default_client = HFLLMClient()
    return _default_client
