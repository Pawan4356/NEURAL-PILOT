"""LangSmith tracing with redaction (Phase 2).

Tracing is off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set
in `.env`. When it is on, one NeuralPilot run produces one trace: LangGraph
records each graph node as a span, LangChain records each LLM call, and
the agent functions add a named span of their own (`traced`).

Every payload passes through `redact_payload` before it leaves the
process. Prompts never contain dataset rows to begin with; redaction is
the second line of defence:
- fields named like `df`, `rows`, `sample` or `api_key` are replaced;
- DataFrames, Series and arrays are replaced by a placeholder;
- the values of secret environment variables (HF_TOKEN, ...) are masked
  wherever they appear in a string.

Deferred to later phases: privacy-mode column aliasing and LangSmith
feedback scores (`constraints_passed`, holdout metric) on the root run.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import Any, Callable, Iterator, Mapping

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "neuralpilot-dev"
REDACTED = "[redacted]"
SENSITIVE_KEYS = {
    "df",
    "dataframe",
    "rows",
    "sample",
    "samples",
    "api_key",
    "token",
    "hf_token",
    "huggingfacehub_api_token",
    "authorization",
    "password",
    "secret",
    # One-hot feature names ("color_teal") carry category values.
    "shap_summary",
}
SECRET_ENV_VARS = ("HF_TOKEN", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "TAVILY_API_KEY")
_MIN_SECRET_LENGTH = 8  # shorter values would mask ordinary words

_client = None


def tracing_enabled() -> bool:
    flag = os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or ""
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def tracing_status() -> str:
    """One line for the startup log; never includes the key itself."""
    if not tracing_enabled():
        return "LangSmith tracing: off"
    configured = "yes" if os.environ.get("LANGSMITH_API_KEY") else "no"
    return f"LangSmith tracing: on (project={project_name()}, api key configured: {configured})"


def project_name() -> str:
    return os.environ.get("LANGSMITH_PROJECT", DEFAULT_PROJECT)


def _secret_values() -> list[str]:
    values = (os.environ.get(name) for name in SECRET_ENV_VARS)
    return [v for v in values if v and len(v) >= _MIN_SECRET_LENGTH]


def _placeholder_for(value: Any) -> str | None:
    """Tabular data never goes into a trace, whatever key it sits under."""
    type_name = type(value).__name__
    if type_name in {"DataFrame", "Series", "ndarray"}:
        return f"[{type_name} redacted]"
    return None


def redact(value: Any, secrets: list[str] | None = None) -> Any:
    """Recursively copy `value` with sensitive fields and secrets masked."""
    secrets = _secret_values() if secrets is None else secrets
    placeholder = _placeholder_for(value)
    if placeholder is not None:
        return placeholder
    if isinstance(value, Mapping):
        return {
            key: REDACTED if str(key).lower() in SENSITIVE_KEYS else redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret in value:
                value = value.replace(secret, REDACTED)
        return value
    return value


def redact_payload(payload: dict) -> dict:
    """Hook for LangSmith's `hide_inputs` / `hide_outputs` and `process_inputs`."""
    return redact(payload)


def redaction_hooks() -> dict:
    """LangSmith `Client` arguments that pass every payload through `redact_payload`."""
    return {"hide_inputs": redact_payload, "hide_outputs": redact_payload}


def get_tracing_client():
    """A LangSmith client that redacts every input and output it sends."""
    global _client
    if _client is None:
        from langsmith import Client

        _client = Client(**redaction_hooks())
    return _client


@contextlib.contextmanager
def run_tracing(client=None) -> Iterator[None]:
    """Route all spans created inside this block through the redacting
    client. A no-op when tracing is off and no client is passed (tests pass
    a capturing client)."""
    if client is None and not tracing_enabled():
        yield
        return
    from langsmith import tracing_context

    with tracing_context(client=client or get_tracing_client(), enabled=True, project_name=project_name()):
        yield


def traced(
    name: str,
    *,
    run_type: str = "chain",
    tags: list[str] | None = None,
    metadata_fn: Callable[..., dict] | None = None,
):
    """Give a function its own named span. `metadata_fn(*args, **kwargs)`
    adds call-specific metadata (e.g. the iteration number). Cheap when
    tracing is off: langsmith skips span creation entirely."""
    from langsmith import get_current_run_tree, traceable

    def decorator(fn):
        @functools.wraps(fn)
        def with_metadata(*args, **kwargs):
            if metadata_fn is not None:
                run_tree = get_current_run_tree()
                if run_tree is not None:
                    try:
                        run_tree.add_metadata(metadata_fn(*args, **kwargs))
                    except Exception:  # metadata must never break the call
                        logger.debug("could not attach span metadata for %s", name, exc_info=True)
            return fn(*args, **kwargs)

        return traceable(run_type=run_type, name=name, tags=tags, process_inputs=redact_payload)(with_metadata)

    return decorator
