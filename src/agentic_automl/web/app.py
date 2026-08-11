"""Minimal web UI (§7, §9) — usable by non-technical users.

One page: upload a dataset + describe the goal in plain English, answer
the single clarifying question if asked, watch iterations progress, see
the final report. Runs execute in a background thread; the frontend polls
for status.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import config
from ..orchestrator import RunState, apply_clarification, create_run, run_to_completion

app = FastAPI(title="Agentic AutoML")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

RUNS: dict[str, RunState] = {}
_RUNS_LOCK = threading.Lock()


def _run_in_background(run_id: str) -> None:
    state = RUNS[run_id]
    run_to_completion(state)
    with _RUNS_LOCK:
        RUNS[run_id] = state


def _serialize(state: RunState) -> dict:
    return {
        "run_id": state.run_id,
        "dataset_name": state.dataset_name,
        "status": state.status,
        "clarification_question": state.clarification_question,
        "iteration": state.iteration,
        "max_iterations": config.MAX_ITERATIONS,
        "iteration_logs": [
            {
                "iteration": log.iteration,
                "plan_summary": log.plan_summary,
                "composite_score": round(log.composite_score, 4),
                "weakest_block": log.weakest_block,
                "decision": log.decision,
                "decision_rationale": log.decision_rationale,
            }
            for log in state.iteration_logs
        ],
        "final_report": state.final_report,
        "error": state.error,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


@app.post("/api/runs")
async def create_run_endpoint(file: UploadFile, goal: str = Form(...)) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    if not goal.strip():
        raise HTTPException(400, "Please describe what you want to predict.")

    dataset_path = config.DATA_DIR / f"{uuid.uuid4()}_{file.filename}"
    with dataset_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        state = create_run(dataset_path, file.filename, goal)
    except Exception as exc:
        raise HTTPException(400, f"Could not read dataset: {exc}") from exc

    with _RUNS_LOCK:
        RUNS[state.run_id] = state

    thread = threading.Thread(target=_run_in_background, args=(state.run_id,), daemon=True)
    thread.start()

    return JSONResponse(_serialize(state))


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(404, "Run not found.")
    return JSONResponse(_serialize(state))


@app.post("/api/runs/{run_id}/clarify")
def clarify_run(run_id: str, answer: str = Form(...)) -> JSONResponse:
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(404, "Run not found.")
    if state.status != "awaiting_clarification":
        raise HTTPException(400, f"Run is not awaiting clarification (status: {state.status}).")

    apply_clarification(state, answer)
    thread = threading.Thread(target=_run_in_background, args=(run_id,), daemon=True)
    thread.start()

    return JSONResponse(_serialize(state))


@app.get("/api/runs/{run_id}/model")
def download_model(run_id: str) -> FileResponse:
    state = RUNS.get(run_id)
    if state is None or not state.model_path:
        raise HTTPException(404, "Model not available.")
    return FileResponse(state.model_path, filename=f"{state.dataset_name}_model.joblib")
