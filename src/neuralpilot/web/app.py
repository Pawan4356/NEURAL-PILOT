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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import config
from ..agents.requirement_agent import InvalidClarification
from ..engine import DatasetLoadError
from ..orchestrator import RunState, apply_clarification, create_run, run_to_completion

app = FastAPI(title="Neural Pilot")

_STATIC_DIR = Path(__file__).parent / "static"
_LOGO_PATH = Path(__file__).resolve().parents[3] / "media" / "Logo.svg"
_FAVICON_PATH = Path(__file__).resolve().parents[3] / "media" / "favicon.svg"
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
        "current_task": state.current_task,
        "activity_log": state.activity_log,
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
                "fallbacks": log.fallbacks,
                "verification_warnings": log.verification_warnings,
                "signature": log.signature,
                "plan_source": log.plan_source,
                "primary_metric": log.primary_metric,
                "cv_mean": round(log.cv_mean, 4),
                "cv_std": round(log.cv_std, 4),
                "latency_ms_per_1k": round(log.latency_ms_per_1k, 3),
                "decision_forced": log.decision_forced,
            }
            for log in state.iteration_logs
        ],
        "final_report": state.final_report,
        "error": state.error,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((_STATIC_DIR / "index.html").read_text())


@app.get("/logo.svg")
def logo(theme: str = None) -> Response:
    content = _LOGO_PATH.read_text()
    if theme == "dark":
        content = content.replace("#121726", "#FFFFFF")
    elif theme == "light":
        # Force light mode by replacing the dark mode white color with navy blue
        # so the media query doesn't override it if the OS is in dark mode
        content = content.replace("fill: #FFFFFF", "fill: #121726")
    return Response(content, media_type="image/svg+xml")


@app.get("/favicon.svg")
def favicon() -> Response:
    return Response(_FAVICON_PATH.read_text(), media_type="image/svg+xml")


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
    except (DatasetLoadError, Exception) as exc:  # DatasetLoadError carries an actionable message
        dataset_path.unlink(missing_ok=True)
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

    try:
        apply_clarification(state, answer)
    except InvalidClarification as exc:
        # The run stays awaiting clarification; the user can answer again.
        raise HTTPException(400, str(exc)) from exc
    thread = threading.Thread(target=_run_in_background, args=(run_id,), daemon=True)
    thread.start()

    return JSONResponse(_serialize(state))


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> JSONResponse:
    """Stop after the current trial; the best attempt so far is still saved."""
    state = RUNS.get(run_id)
    if state is None:
        raise HTTPException(404, "Run not found.")
    if state.status != "running":
        raise HTTPException(400, f"Run is not running (status: {state.status}).")
    state.cancel_event.set()
    return JSONResponse(_serialize(state))


@app.get("/api/runs/{run_id}/model")
def download_model(run_id: str) -> FileResponse:
    state = RUNS.get(run_id)
    if state is None or not state.model_path:
        raise HTTPException(404, "Model not available.")
    return FileResponse(state.model_path, filename=f"{state.dataset_name}_model.joblib")
