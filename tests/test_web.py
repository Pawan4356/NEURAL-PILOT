from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from neuralpilot import config
from neuralpilot.web import app as web_app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    # Keep runs from starting real background work during these tests.
    monkeypatch.setattr(web_app, "_run_in_background", lambda run_id: None)
    web_app.RUNS.clear()
    yield TestClient(web_app.app)
    web_app.RUNS.clear()


def test_unreadable_csv_returns_actionable_400_and_is_not_kept(client, tmp_path):
    with (FIXTURES / "junk_header.csv").open("rb") as f:
        response = client.post("/api/runs", files={"file": ("junk.csv", f, "text/csv")}, data={"goal": "predict 3"})
    assert response.status_code == 400
    assert "header row" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_semicolon_csv_is_accepted(client):
    with (FIXTURES / "semicolon_decimal_comma.csv").open("rb") as f:
        response = client.post("/api/runs", files={"file": ("s.csv", f, "text/csv")}, data={"goal": "predict grade"})
    assert response.status_code == 200
    state = web_app.RUNS[response.json()["run_id"]]
    assert state.df.shape[1] == 5


def test_invalid_clarification_returns_400_and_keeps_waiting(client, tmp_path):
    from neuralpilot.orchestrator import RunState
    from neuralpilot.schemas import ConstraintWeights, RequirementSpec

    df = pd.DataFrame({"a": range(20), "b": [0, 1] * 10})
    state = RunState(run_id="r1", dataset_path=tmp_path / "x.csv", dataset_name="x.csv", user_goal="g", df=df)
    state.status = "awaiting_clarification"
    state.requirement = RequirementSpec(
        target_column="guess", target_confidence=0.2, problem_type="classification",
        constraint_weights=ConstraintWeights(), weight_confidence=0.9,
    )
    web_app.RUNS["r1"] = state

    response = client.post("/api/runs/r1/clarify", data={"answer": "nope"})
    assert response.status_code == 400
    assert "Available columns: a, b" in response.json()["detail"]
    assert state.status == "awaiting_clarification"

    response = client.post("/api/runs/r1/clarify", data={"answer": "B"})
    assert response.status_code == 200
    assert state.requirement.target_column == "b"


def test_cancel_sets_the_event(client, tmp_path):
    from neuralpilot.orchestrator import RunState

    state = RunState(
        run_id="r2", dataset_path=tmp_path / "x.csv", dataset_name="x.csv", user_goal="g",
        df=pd.DataFrame({"a": [1]}),
    )
    web_app.RUNS["r2"] = state
    assert client.post("/api/runs/r2/cancel").status_code == 200
    assert state.cancel_event.is_set()
