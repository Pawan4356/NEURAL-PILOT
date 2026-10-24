"""Phase 2: the LangChain LLM adapter (structured output, JSON repair,
transport backoff) and backend selection. No network: fake chat models."""

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from neuralpilot import config, llm_client
from neuralpilot.agents.requirement_agent import understand_requirement
from neuralpilot.langchain_llm import (
    LangChainLLMClient,
    LLMTransportError,
    call_with_backoff,
    is_transient,
)
from neuralpilot.llm_client import LLMJSONError, LLMNotConfiguredError
from neuralpilot.meta_features import extract_meta_features
from neuralpilot.schemas import ReflectionOutput, RequirementSpec

VALID_REFLECTION = '{"weakest_block": "model_family", "rationale": "underfits"}'


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(r) for r in replies]))


class HTTPStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class StructuredFake(GenericFakeChatModel):
    """A fake chat model whose provider supports structured output."""

    structured_reply: dict = {}

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _messages: self.structured_reply)


class FlakyModel(GenericFakeChatModel):
    """Fails with the given errors, then answers normally."""

    errors: list = []

    def invoke(self, *args, **kwargs):
        if self.errors:
            raise self.errors.pop(0)
        return super().invoke(*args, **kwargs)


def client_for(model, **kwargs) -> LangChainLLMClient:
    kwargs.setdefault("sleep", lambda s: None)
    return LangChainLLMClient(chat_model=model, **kwargs)


def test_provider_structured_output_is_used_and_validated():
    model = StructuredFake(messages=iter([]), structured_reply={"weakest_block": "preprocessing", "rationale": "r"})
    result = client_for(model).complete_json("sys", "user", ReflectionOutput)
    assert result == ReflectionOutput(weakest_block="preprocessing", rationale="r")


def test_falls_back_to_json_repair_when_provider_lacks_structured_output():
    # GenericFakeChatModel has no tool calling, like some HF provider routes.
    client = client_for(fake_model(f"```json\n{VALID_REFLECTION}\n```", VALID_REFLECTION))
    assert client.complete_json("sys", "user", ReflectionOutput).weakest_block == "model_family"
    assert client._structured_supported is False  # not retried on the next call
    assert client.complete_json("sys", "user", ReflectionOutput).rationale == "underfits"


def test_schema_mismatch_from_structured_output_falls_back_for_that_call_only():
    model = StructuredFake(messages=iter([AIMessage(VALID_REFLECTION)]), structured_reply={"weakest_block": "nonsense"})
    client = client_for(model)
    assert client.complete_json("sys", "user", ReflectionOutput).weakest_block == "model_family"
    assert client._structured_supported is None  # still worth trying next time


def test_json_repair_loop_shows_the_model_its_error():
    model = fake_model("not json at all", VALID_REFLECTION)
    seen = []
    original = model.invoke

    def recording_invoke(messages, *args, **kwargs):
        seen.append([m.content for m in messages])
        return original(messages, *args, **kwargs)

    object.__setattr__(model, "invoke", recording_invoke)
    result = client_for(model, structured_output="off").complete_json("sys", "user", ReflectionOutput)
    assert result.weakest_block == "model_family"
    assert len(seen) == 2
    assert "That response was invalid" in seen[1][-1]


def test_json_repair_gives_up_after_max_retries():
    client = client_for(fake_model("nope", "still nope", "never"), structured_output="off")
    with pytest.raises(LLMJSONError):
        client.complete_json("sys", "user", ReflectionOutput, max_retries=2)


def test_rate_limit_is_retried_with_backoff():
    sleeps = []
    model = FlakyModel(messages=iter([AIMessage(VALID_REFLECTION)]), errors=[HTTPStatusError(429), HTTPStatusError(503)])
    client = LangChainLLMClient(chat_model=model, structured_output="off", sleep=sleeps.append)
    assert client.complete_json("sys", "user", ReflectionOutput).weakest_block == "model_family"
    assert len(sleeps) == 2
    assert sleeps[1] >= sleeps[0] * 0.66  # exponential, give or take jitter


def test_transport_retries_are_bounded():
    sleeps = []
    with pytest.raises(LLMTransportError):
        call_with_backoff(lambda: (_ for _ in ()).throw(HTTPStatusError(429)), max_attempts=3, sleep=sleeps.append)
    assert len(sleeps) == 2


def test_client_errors_are_not_retried():
    sleeps = []
    with pytest.raises(HTTPStatusError):
        call_with_backoff(lambda: (_ for _ in ()).throw(HTTPStatusError(401)), max_attempts=3, sleep=sleeps.append)
    assert sleeps == []


def test_transient_classification():
    assert is_transient(HTTPStatusError(429))
    assert is_transient(HTTPStatusError(502))
    assert is_transient(TimeoutError())
    assert not is_transient(HTTPStatusError(400))
    assert not is_transient(ValueError("bad"))


def test_agent_runs_unchanged_on_the_langchain_backend():
    reply = (
        '{"target_column": "y", "target_confidence": 0.9, "problem_type": "classification", '
        '"constraint_weights": {"accuracy": 0.7, "latency": 0.15, "interpretability": 0.15}, '
        '"weight_confidence": 0.9, "assumptions": []}'
    )
    df = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})
    spec = understand_requirement("predict y", extract_meta_features(df), client=client_for(fake_model(reply)))
    assert isinstance(spec, RequirementSpec) and spec.target_column == "y"


def test_backend_selection(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(config.LLM_CONFIG, "api_key", None)
    for backend in ("langchain", "hf"):
        llm_client._default_client = None
        monkeypatch.setattr(config, "LLM_BACKEND", backend)
        with pytest.raises(LLMNotConfiguredError):
            llm_client.get_default_client()
    llm_client._default_client = None
    monkeypatch.setattr(config, "LLM_BACKEND", "bogus")
    with pytest.raises(ValueError, match="NEURALPILOT_LLM_BACKEND"):
        llm_client.get_default_client()


def test_source_selection_builds_the_matching_chat_model(monkeypatch):
    from neuralpilot.langchain_llm import build_default_chat_model

    monkeypatch.setattr(config, "LLM_SOURCE", "huggingface")
    monkeypatch.setattr(config.LLM_CONFIG, "api_key", "hf_dummy_token_123456")
    assert type(build_default_chat_model()).__name__ == "ChatHuggingFace"

    monkeypatch.setattr(config, "LLM_SOURCE", "claude")
    monkeypatch.setattr(config.CLAUDE_CONFIG, "api_key", "sk-ant-dummy-123456")
    model = build_default_chat_model()
    assert type(model).__name__ == "ChatAnthropic"
    assert model.model == config.CLAUDE_CONFIG.model


def test_claude_source_without_api_key_is_not_configured(monkeypatch):
    from neuralpilot.langchain_llm import build_default_chat_model

    monkeypatch.setattr(config, "LLM_SOURCE", "claude")
    monkeypatch.setattr(config.CLAUDE_CONFIG, "api_key", None)
    with pytest.raises(LLMNotConfiguredError, match="ANTHROPIC_API_KEY"):
        build_default_chat_model()


def test_source_selection_rejects_unknown_and_legacy_combinations(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "langchain")
    monkeypatch.setattr(config, "LLM_SOURCE", "openai")
    llm_client._default_client = None
    with pytest.raises(ValueError, match="NEURALPILOT_LLM_SOURCE"):
        llm_client.get_default_client()

    monkeypatch.setattr(config, "LLM_SOURCE", "claude")
    monkeypatch.setattr(config, "LLM_BACKEND", "hf")
    monkeypatch.setattr(config.CLAUDE_CONFIG, "api_key", "sk-ant-dummy-123456")
    llm_client._default_client = None
    with pytest.raises(ValueError, match="only supports NEURALPILOT_LLM_SOURCE=huggingface"):
        llm_client.get_default_client()
    llm_client._default_client = None


def test_claude_overload_and_connection_errors_are_transient():
    class APIConnectionError(Exception):
        pass

    assert is_transient(HTTPStatusError(529))  # Anthropic "overloaded"
    assert is_transient(APIConnectionError("dropped"))
    assert not is_transient(HTTPStatusError(401))  # bad key: fail fast


class StubInference:
    """Stands in for huggingface_hub.InferenceClient.chat_completion."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def chat_completion(self, **params):
        self.requests.append(params)
        return self.responses.pop(0)


def _hf_client(stub, monkeypatch, **kwargs):
    from neuralpilot.config import LLMConfig
    from neuralpilot.langchain_llm import build_chat_model

    model = build_chat_model(LLMConfig(api_key="hf_dummy_token_123456"))
    monkeypatch.setattr(model.llm, "client", stub)
    return LangChainLLMClient(chat_model=model, sleep=lambda s: None, **kwargs)


def test_real_chat_huggingface_tool_call_round_trip(monkeypatch):
    """The real ChatHuggingFace sends a tool definition and the tool call it
    gets back is validated into the schema."""
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "ReflectionOutput", "arguments": VALID_REFLECTION},
    }
    stub = StubInference({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}]})
    client = _hf_client(stub, monkeypatch)

    result = client.complete_json("sys", "user", ReflectionOutput)
    assert result == ReflectionOutput(weakest_block="model_family", rationale="underfits")
    [request] = stub.requests
    assert request["tools"][0]["function"]["name"] == "ReflectionOutput"
    assert request["model"] == config.LLM_CONFIG.model


def test_real_chat_huggingface_without_tool_call_falls_back_to_json(monkeypatch):
    """A provider that ignores the tool and answers in text still works."""
    text_reply = {"choices": [{"message": {"role": "assistant", "content": VALID_REFLECTION}}]}
    stub = StubInference(text_reply, text_reply)
    client = _hf_client(stub, monkeypatch)
    assert client.complete_json("sys", "user", ReflectionOutput).weakest_block == "model_family"
    assert len(stub.requests) == 2 and "tools" not in stub.requests[1]
