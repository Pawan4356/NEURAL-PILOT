import pytest

from neuralpilot import config, llm_client


@pytest.fixture(autouse=True)
def isolated_run_storage(tmp_path, monkeypatch):
    """Graph checkpoints, models and MLflow stores go to a per-test directory,
    never runs/. MLflow is off unless a test turns it on for its own store."""
    monkeypatch.setattr(config, "GRAPH_DB_PATH", tmp_path / "graph.sqlite3")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MLFLOW_ENABLED", False)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", str(tmp_path / "mlartifacts"))
    yield
    llm_client._default_client = None
