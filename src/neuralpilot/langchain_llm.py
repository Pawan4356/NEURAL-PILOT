"""LangChain implementation of the `LLMClient` protocol (Phase 2).

Agents keep calling `complete_json(system, user, schema)`; only the
transport changes. The provider is chosen by NEURALPILOT_LLM_SOURCE:
Hugging Face Inference Providers (default) or NVIDIA NIM.
Each call:

1. tries the provider's structured output (`with_structured_output`),
   unless that has already failed for this provider or is switched off;
2. otherwise sends a plain chat request and validates the reply with the
   same JSON-repair loop as the original client.

Transport failures (429, 5xx, timeouts, dropped connections) are retried
with exponential backoff and jitter. They are kept separate from JSON
repair: a rate limit is not the model's fault and must not use up the
attempts the model gets to fix its JSON.

Because calls go through a LangChain chat model, they appear as LLM spans
in LangSmith whenever tracing is on (see `observability.py`).
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Callable, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from . import config
from .config import LLMConfig, NvidiaConfig
from .llm_client import LLMJSONError, LLMNotConfiguredError, _extract_json

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

# 529 ("overloaded") is not a standard code but is used by some providers;
# it is retried like a rate limit.
_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_TRANSIENT_ERROR_NAMES = {
    "TimeoutError",
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "TimeoutException",
    "ConnectError",
    "RemoteProtocolError",
    "APIConnectionError",  # OpenAI-style SDKs; APITimeoutError subclasses it
}


class LLMTransportError(RuntimeError):
    """The provider kept failing at the transport level after all retries."""


def build_chat_model(llm_config: LLMConfig) -> BaseChatModel:
    """Chat model over Hugging Face Inference Providers, configured from the
    existing HF_TOKEN / HF_MODEL / HF_PROVIDER settings."""
    if not llm_config.api_key:
        raise LLMNotConfiguredError(
            "HF_TOKEN is not set. Export HF_TOKEN=<your Hugging Face access token> "
            "before running agents that call the LLM."
        )
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id=llm_config.model,
        provider=llm_config.provider,
        huggingfacehub_api_token=llm_config.api_key,
        task="conversational",
        max_new_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
        timeout=int(llm_config.timeout),
    )
    return ChatHuggingFace(
        llm=endpoint,
        model_id=llm_config.model,
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
    )


def build_nvidia_chat_model(nvidia_config: NvidiaConfig) -> BaseChatModel:
    """Chat model over NVIDIA NIM (OpenAI-compatible), configured from the
    NVIDIA_API_KEY / NVIDIA_MODEL / NVIDIA_BASE_URL settings."""
    if not nvidia_config.api_key:
        raise LLMNotConfiguredError(
            "NVIDIA_API_KEY is not set. Export NVIDIA_API_KEY=<your NVIDIA API key> "
            "before running agents that call NVIDIA."
        )
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    kwargs: dict[str, Any] = {}
    if nvidia_config.base_url:
        kwargs["base_url"] = nvidia_config.base_url
    return ChatNVIDIA(
        model=nvidia_config.model,
        api_key=nvidia_config.api_key,
        max_completion_tokens=nvidia_config.max_tokens,
        temperature=nvidia_config.temperature,
        timeout=nvidia_config.timeout,
        **kwargs,
    )


def build_default_chat_model() -> BaseChatModel:
    """The chat model for the provider chosen by NEURALPILOT_LLM_SOURCE."""
    if config.LLM_SOURCE == "nvidia":
        return build_nvidia_chat_model(config.NVIDIA_CONFIG)
    return build_chat_model(config.LLM_CONFIG)


def status_code_of(exc: BaseException) -> int | None:
    """HTTP status carried by an exception from huggingface_hub, httpx or requests."""
    code = getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def is_transient(exc: BaseException) -> bool:
    """Worth retrying: rate limits, server errors, timeouts, dropped connections."""
    code = status_code_of(exc)
    if code is not None:
        return code in _TRANSIENT_STATUS_CODES
    return any(cls.__name__ in _TRANSIENT_ERROR_NAMES for cls in type(exc).__mro__)


def call_with_backoff(
    fn: Callable[[], R],
    *,
    max_attempts: int,
    base_delay_s: float = config.LLM_BACKOFF_BASE_S,
    max_delay_s: float = config.LLM_BACKOFF_MAX_S,
    sleep: Callable[[float], None] = time.sleep,
) -> R:
    """Run `fn`, retrying transient failures with exponential backoff + jitter.
    Non-transient errors are raised immediately."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_transient(exc):
                raise
            if attempt == max_attempts:
                raise LLMTransportError(
                    f"LLM provider still failing after {max_attempts} attempts: {exc}"
                ) from exc
            delay = min(max_delay_s, base_delay_s * 2 ** (attempt - 1))
            delay += random.uniform(0, delay / 2)
            logger.warning("LLM transport error (%s); retry %s/%s in %.1fs", exc, attempt, max_attempts - 1, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def _is_output_error(exc: BaseException) -> bool:
    """The model answered but the answer did not fit the schema. Such errors
    are about this one reply, not about the provider's capabilities."""
    from langchain_core.exceptions import OutputParserException

    return isinstance(exc, (OutputParserException, ValidationError, json.JSONDecodeError, LLMJSONError))


class LangChainLLMClient:
    """`LLMClient` backed by a LangChain chat model."""

    def __init__(
        self,
        *,
        chat_model: BaseChatModel | None = None,
        structured_output: str | None = None,
        structured_method: str | None = None,
        max_transport_attempts: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.chat_model = chat_model if chat_model is not None else build_default_chat_model()
        self.structured_output = structured_output or config.LLM_STRUCTURED_OUTPUT
        self.structured_method = structured_method or config.LLM_STRUCTURED_METHOD
        self.max_transport_attempts = max_transport_attempts or config.LLM_TRANSPORT_RETRIES
        self._sleep = sleep
        # None = not tried yet; False = the provider rejected it, stop trying.
        self._structured_supported: bool | None = None if self.structured_output != "off" else False

    def complete_json(
        self, system_prompt: str, user_prompt: str, schema: type[T], *, max_retries: int = 2
    ) -> T:
        messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
        if self._structured_supported is not False:
            result = self._try_structured(messages, schema)
            if result is not None:
                return result
        return self._complete_with_repair(messages, schema, max_retries)

    def _structured_kwargs(self) -> dict[str, str]:
        """`method` is only passed when a provider needs to be told. NVIDIA NIM
        chooses it from the schema and warns when it is passed one."""
        if self.structured_method in ("", "none", "auto"):
            return {}
        return {"method": self.structured_method}

    def _invoke(self, runnable: Any, messages: list[BaseMessage]) -> Any:
        return call_with_backoff(
            lambda: runnable.invoke(messages),
            max_attempts=self.max_transport_attempts,
            sleep=self._sleep,
        )

    def _try_structured(self, messages: list[BaseMessage], schema: type[T]) -> T | None:
        """Provider-native structured output. Returns None when the caller
        should fall back to the plain-text repair loop."""
        try:
            # Passed as JSON Schema, not the Pydantic class: ChatHuggingFace
            # rejects Pydantic classes for function calling. The reply is
            # validated against the Pydantic model below either way.
            structured = self.chat_model.with_structured_output(
                schema.model_json_schema(), **self._structured_kwargs()
            )
            result = self._invoke(structured, messages)
            if result is None:
                raise LLMJSONError("structured output returned nothing (no tool call)")
            parsed = result if isinstance(result, schema) else schema.model_validate(result)
            self._structured_supported = True
            return parsed
        except LLMTransportError:
            raise
        except Exception as exc:
            if self.structured_output == "on":
                raise
            if _is_output_error(exc):
                logger.info("Structured output did not match %s (%s); using JSON repair", schema.__name__, exc)
            else:
                logger.info("Provider does not support structured output (%s); using JSON repair from now on", exc)
                self._structured_supported = False
            return None

    def _complete_with_repair(self, messages: list[BaseMessage], schema: type[T], max_retries: int) -> T:
        """Plain chat completion; on invalid JSON, show the model its reply
        and the error, and ask again (same policy as `HFLLMClient`)."""
        messages = list(messages)
        last_error: Exception | None = None
        for _ in range(max_retries + 1):
            reply = self._invoke(self.chat_model, messages)
            content = reply.content if isinstance(reply.content, str) else str(reply.content)
            try:
                return schema.model_validate(_extract_json(content))
            except (LLMJSONError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                messages.append(AIMessage(content))
                messages.append(
                    HumanMessage(
                        f"That response was invalid: {exc}. "
                        "Reply with ONLY valid JSON matching the required schema — "
                        "no prose, no markdown code fences."
                    )
                )
        raise LLMJSONError(f"Failed to get valid JSON after {max_retries + 1} attempts: {last_error}")
