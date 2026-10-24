import pytest

from neuralpilot import config, llm_client


@pytest.fixture(autouse=True)
def isolated_run_storage(tmp_path, monkeypatch):
    """Graph checkpoints and model files go to a per-test directory, never runs/."""
    monkeypatch.setattr(config, "GRAPH_DB_PATH", tmp_path / "graph.sqlite3")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    yield
    llm_client._default_client = None
