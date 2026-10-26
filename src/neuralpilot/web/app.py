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

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import config
from ..accounts import AccountStore, OwnedRun, User
from ..agents.requirement_agent import InvalidClarification
from ..engine import DatasetLoadError
from ..orchestrator import RunState, apply_clarification, create_run, run_to_completion
from . import history
from .auth import current_user, get_accounts, router as auth_router

app = FastAPI(title="Neural Pilot")
app.include_router(auth_router)

_STATIC_DIR = Path(__file__).parent / "static"
_LOGO_PATH = Path(__file__).resolve().parents[3] / "media" / "Logo.svg"
_FAVICON_PATH = Path(__file__).resolve().parents[3] / "media" / "favicon.svg"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Active runs only. A finished run is evicted here (its DataFrame is large) and
# served from the experiment store from then on.
RUNS: dict[str, RunState] = {}
_RUNS_LOCK = threading.Lock()


def _run_in_background(run_id: str) -> None:
    state = RUNS[run_id]
    run_to_completion(state)
    if state.status in history.FINISHED_STATUSES:
        with history.experiment_store() as repo:
            repo.save_activity(run_id, state.activity_log)
    with _RUNS_LOCK:
        if state.status in history.FINISHED_STATUSES:
            RUNS.pop(run_id, None)
        else:
            RUNS[run_id] = state  # paused for clarification: keep it so the answer can resume it


def _live_state(run_id: str) -> RunState | None:
    with _RUNS_LOCK:
        return RUNS.get(run_id)


def _owned_run(run_id: str, user: User, accounts: AccountStore) -> OwnedRun:
    """The run, if this user owns it. Other users' runs get the same 404 as
    unknown ones, so the API does not reveal which run IDs exist."""
    owned = accounts.owned_run(run_id, user.id)
    if owned is None:
        raise HTTPException(404, "Run not found.")
    return owned


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


@app.get("/api/config")
def runtime_config() -> JSONResponse:
    """Settings the page shows before a run exists (LLM source, split, budgets)."""
    key = config.NVIDIA_CONFIG.api_key if config.LLM_SOURCE == "nvidia" else config.LLM_CONFIG.api_key
    return JSONResponse(
        {
            "llm_source": config.LLM_SOURCE,
            "llm_model": config.active_llm_model(),
            "llm_key_configured": bool(key),
            "max_iterations": config.MAX_ITERATIONS,
            "run_budget_s": config.RUN_BUDGET_S,
            "seed": config.SEED,
            "holdout_fraction": config.HOLDOUT_FRACTION,
            "cv_folds": config.CV_FOLDS,
        }
    )


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


@app.get("/api/runs")
def list_runs(user: User = Depends(current_user), accounts: AccountStore = Depends(get_accounts)) -> JSONResponse:
    """This user's runs, newest first. Only their own runs are ever listed."""
    rows = []
    with history.experiment_store() as repo:
        for owned in accounts.runs_for_user(user.id):
            state = _live_state(owned.run_id)
            live_status = state.status if state else None
            record = repo.get_run(owned.run_id) if live_status is None else None
            rows.append(history.summarize(owned, live_status, record))
    return JSONResponse(rows)


@app.post("/api/runs")
async def create_run_endpoint(
    file: UploadFile,
    goal: str = Form(...),
    user: User = Depends(current_user),
    accounts: AccountStore = Depends(get_accounts),
) -> JSONResponse:
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

    # Ownership is recorded before the run starts, so a run interrupted by a restart
    # still shows up in its owner's list.
    accounts.record_run(state.run_id, user.id, file.filename, goal)
    with _RUNS_LOCK:
        RUNS[state.run_id] = state

    thread = threading.Thread(target=_run_in_background, args=(state.run_id,), daemon=True)
    thread.start()

    return JSONResponse(_serialize(state))


@app.get("/api/runs/{run_id}")
def get_run(
    run_id: str, user: User = Depends(current_user), accounts: AccountStore = Depends(get_accounts)
) -> JSONResponse:
    owned = _owned_run(run_id, user, accounts)
    state = _live_state(run_id)
    if state is not None:
        return JSONResponse(_serialize(state))
    with history.experiment_store() as repo:
        record = repo.get_run(run_id)
    if record is None:
        return JSONResponse(history.interrupted_view(owned))
    return JSONResponse(history.serialize_record(record))


@app.post("/api/runs/{run_id}/clarify")
def clarify_run(
    run_id: str,
    answer: str = Form(...),
    user: User = Depends(current_user),
    accounts: AccountStore = Depends(get_accounts),
) -> JSONResponse:
    _owned_run(run_id, user, accounts)
    state = _live_state(run_id)
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
def cancel_run(
    run_id: str, user: User = Depends(current_user), accounts: AccountStore = Depends(get_accounts)
) -> JSONResponse:
    """Stop after the current trial; the best attempt so far is still saved."""
    _owned_run(run_id, user, accounts)
    state = _live_state(run_id)
    if state is None:
        raise HTTPException(404, "Run not found.")
    if state.status != "running":
        raise HTTPException(400, f"Run is not running (status: {state.status}).")
    state.cancel_event.set()
    return JSONResponse(_serialize(state))


@app.get("/api/runs/{run_id}/model")
def download_model(
    run_id: str, user: User = Depends(current_user), accounts: AccountStore = Depends(get_accounts)
) -> FileResponse:
    owned = _owned_run(run_id, user, accounts)
    state = _live_state(run_id)
    if state is not None:
        model_path, dataset_name = state.model_path, state.dataset_name
    else:
        with history.experiment_store() as repo:
            record = repo.get_run(run_id)
        model_path, dataset_name = (record.model_path, record.dataset_name) if record else (None, owned.dataset_name)
    if not model_path or not Path(model_path).exists():
        raise HTTPException(404, "Model not available.")
    return FileResponse(model_path, filename=f"{dataset_name}_model.joblib")
