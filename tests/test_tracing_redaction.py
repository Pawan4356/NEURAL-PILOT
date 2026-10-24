"""Phase 2: what leaves the process when LangSmith tracing is on.

A whole run is traced into a capturing LangSmith client (nothing is sent
over the network). Redaction happens inside the real `Client.create_run` /
`update_run`; the capture hooks sit after it, where the HTTP send would be.
"""

import json

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

from neuralpilot import observability
from neuralpilot.langchain_llm import LangChainLLMClient
from neuralpilot.orchestrator import create_run, run_to_completion
from neuralpilot.repository import ExperimentRepository

SECRET_TOKEN = "hf_TESTSECRET_do_not_leak_123456"
CATEGORY_VALUES = ["zqteal", "zqamber", "zqviolet"]


class CapturingClient(Client):
    def __init__(self):
        super().__init__(
            api_key="test-key", api_url="http://127.0.0.1:9", auto_batch_tracing=False, **observability.redaction_hooks()
        )
        self.payloads: list[dict] = []

    def _create_run(self, run_create, **kwargs):
        self.payloads.append(run_create)

    def _update_run(self, run_update, **kwargs):
        self.payloads.append(run_update)


def _llm_replies():
    requirement = {
        "target_column": "y",
        "target_confidence": 0.95,
        "problem_type": "classification",
        # Interpretability above default, confidently stated: SHAP is computed,
        # so one-hot feature names (which contain category values) exist.
        "constraint_weights": {"accuracy": 0.4, "latency": 0.1, "interpretability": 0.5},
        "weight_confidence": 0.95,
        "assumptions": [],
    }
    plan = {
        "preprocessing": [
            {"category": "encoder", "component": "OneHotEncoder", "params": {}, "rationale": "r"},
            {"category": "scaler", "component": "StandardScaler", "params": {}, "rationale": "r"},
        ],
        "feature_engineering": [],
        "model_family": "LinearModel",
        "tuning": {"component": "NoTuning", "n_trials": 0, "rationale": "r"},
        "rationale": "r",
    }
    reflection = {"weakest_block": "model_family", "rationale": "r"}
    decision = {"decision": "accept", "rationale": "good enough"}
    return [AIMessage(json.dumps(r)) for r in (requirement, plan, reflection, decision)]


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "color": rng.choice(CATEGORY_VALUES, size=n),
            "y": rng.choice([0, 1], size=n),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_redact_masks_fields_frames_and_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET_TOKEN)
    payload = {
        "df": "anything",
        "nested": {"api_key": "k", "note": f"token is {SECRET_TOKEN}"},
        "frame": pd.DataFrame({"a": [1]}),
        "items": [{"rows": [1, 2]}, "plain"],
        "shap_summary": {"color_zqteal": 0.3},
    }
    redacted = observability.redact_payload(payload)
    assert redacted["df"] == observability.REDACTED
    assert redacted["nested"]["api_key"] == observability.REDACTED
    assert SECRET_TOKEN not in redacted["nested"]["note"]
    assert redacted["frame"] == "[DataFrame redacted]"
    assert redacted["items"] == [{"rows": observability.REDACTED}, "plain"]
    assert redacted["shap_summary"] == observability.REDACTED


def test_traced_run_sends_no_cell_values_or_secrets(tmp_path, dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET_TOKEN)
    capture = CapturingClient()
    llm = LangChainLLMClient(chat_model=GenericFakeChatModel(messages=iter(_llm_replies())), sleep=lambda s: None)
    repo = ExperimentRepository(db_path=tmp_path / "exp.sqlite3")

    state = create_run(dataset, "data.csv", "predict y, and it must be explainable")
    monkeypatch.setattr(observability, "get_tracing_client", lambda: capture)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    try:
        state = run_to_completion(state, repo=repo, llm=llm)
        wait_for_all_tracers()
    finally:
        repo.close()

    assert state.status == "finalized", state.error
    assert state.final_report["shap_summary"]  # the data-derived names exist ...
    sent = json.dumps(capture.payloads, default=str)
    for value in CATEGORY_VALUES:
        assert value not in sent  # ... but never reach the trace
    assert SECRET_TOKEN not in sent

    names = {p.get("name") for p in capture.payloads}
    assert {"neuralpilot_run", "requirement_agent", "planning_agent", "reflection_agent", "decision_agent"} <= names
    assert {"understand", "plan", "execute", "finalize"} <= names  # graph nodes are spans
    llm_spans = [p for p in capture.payloads if p.get("run_type") == "llm"]
    assert len(llm_spans) >= 4
    assert all(span["extra"]["metadata"]["run_id"] == state.run_id for span in llm_spans)
    root = next(p for p in capture.payloads if p.get("name") == "neuralpilot_run")
    assert root["extra"]["metadata"]["run_id"] == state.run_id
