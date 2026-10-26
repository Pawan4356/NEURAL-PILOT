import os

import pytest

# The suite must not pick up the developer's .env: it decides the LLM source,
# API keys and budgets. Set before `neuralpilot.config` is imported below.
os.environ.setdefault("NEURALPILOT_DOTENV", "off")

from neuralpilot import config, llm_client  # noqa: E402

assert not config.DOTENV_LOADED, "tests must run without .env; NEURALPILOT_DOTENV was overridden"


@pytest.fixture(autouse=True)
def isolated_run_storage(tmp_path, monkeypatch):
    """Graph checkpoints, models and MLflow stores go to a per-test directory,
    never runs/. MLflow is off unless a test turns it on for its own store."""
    monkeypatch.setattr(config, "GRAPH_DB_PATH", tmp_path / "graph.sqlite3")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "experiments.sqlite3")
    monkeypatch.setattr(config, "ACCOUNTS_DB_PATH", tmp_path / "accounts.sqlite3")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MLFLOW_ENABLED", False)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", str(tmp_path / "mlartifacts"))
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", tmp_path / "knowledge")  # no documentation index in tests
    monkeypatch.setattr(config, "LLM_SOURCE", "huggingface")
    monkeypatch.setattr(config, "LLM_BACKEND", "langchain")
    yield
    llm_client._default_client = None
