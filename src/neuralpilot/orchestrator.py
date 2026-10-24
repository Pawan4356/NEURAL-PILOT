"""Run façade over the LangGraph orchestration graph (§6, Phase 3).

The web app and tests talk to three functions:

    create_run(path, name, goal)   load the dataset, create a RunState
    run_to_completion(state)       advance the graph until it finishes or
                                   pauses for the clarifying question
    apply_clarification(state, a)  validate the answer; the next
                                   run_to_completion resumes the graph

The control flow itself lives in `graph/` (nodes + routing):

    understand -> [clarify] -> prepare -> retrieve_knowledge
      -> start_iteration -> plan -> resolve_plan -> execute -> reflect -> decide
      -> (refine/replan: start_iteration | accept/stop: finalize -> persist)

`RunState` is a projection of graph state for the UI: it is refreshed
after every node, plus live progress messages from inside long nodes.
The graph state itself is checkpointed in SQLite (thread_id = run_id).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from langgraph.types import Command

from . import config, observability, provenance, reporting
from .agents.requirement_agent import apply_clarification_answer
from .engine import load_dataset
from .engine.tracking import make_tracker
from .graph import RunContext, compiled_graph, resources
from .graph.routing import recursion_limit
from .llm_client import LLMClient
from .meta_features import extract_meta_features
from .repository import ExperimentRepository
from .schemas import AttemptSummary, BestAttempt, DecisionOutput, IterationLog, RequirementSpec

logger = logging.getLogger(__name__)

RunStatus = Literal["running", "awaiting_clarification", "finalized", "failed"]


@dataclass
class RunState:
    """What the UI shows about one run. Read-only for callers, except for
    `cancel_event` and `budget_s`/`max_iterations` before the run starts."""

    run_id: str
    dataset_path: Path
    dataset_name: str
    user_goal: str
    df: pd.DataFrame
    status: RunStatus = "running"
    current_task: str = "Starting run"
    activity_log: list[dict[str, str]] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    clarification_question: str | None = None
    requirement: RequirementSpec | None = None
    iteration: int = 0
    iteration_logs: list[IterationLog] = field(default_factory=list)
    attempts: list[AttemptSummary] = field(default_factory=list)
    best: BestAttempt | None = None
    dataset_fingerprint: str | None = None
    load_report: dict = field(default_factory=dict)
    budget_s: float = config.RUN_BUDGET_S
    max_iterations: int = config.MAX_ITERATIONS
    cancel_event: threading.Event = field(default_factory=threading.Event)
    holdout_evaluations: int = 0
    model_path: str | None = None
    final_metrics: dict | None = None
    final_decision: DecisionOutput | None = None
    final_report: dict | None = None
    error: str | None = None
    pending_answer: str | None = None  # validated clarification, sent on resume


def create_run(dataset_path: Path, dataset_name: str, user_goal: str) -> RunState:
    """Load the dataset (raises engine.DatasetLoadError with an actionable
    message on unreadable files) and create a run."""
    loaded = load_dataset(dataset_path)
    resources.remember_dataframe(loaded.fingerprint, loaded.df)
    return RunState(
        run_id=str(uuid.uuid4()),
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        user_goal=user_goal,
        df=loaded.df,
        dataset_fingerprint=loaded.fingerprint,
        load_report=loaded.report,
    )


def apply_clarification(state: RunState, answer: str) -> RunState:
    """Validate the user's answer now, so a bad answer gets an immediate
    error. Raises InvalidClarification (state unchanged) if `answer` is not
    a column. The graph receives the matched column name on resume."""
    if state.requirement is None:
        raise ValueError("Cannot apply clarification before requirement understanding has run")
    state.requirement = apply_clarification_answer(state.requirement, answer, state.df)
    state.pending_answer = state.requirement.target_column
    state.status = "running"
    state.current_task = "Resuming after target column clarification"
    state.clarification_question = None
    return state


def run_to_completion(
    state: RunState, repo: ExperimentRepository | None = None, *, llm: LLMClient | None = None
) -> RunState:
    """Advance the run until it finalizes, fails, or needs the one allowed
    clarifying question (§7). Call again after apply_clarification() to
    resume; calling it on a finished run only refreshes the projection."""
    owns_repo = repo is None
    repo = repo or ExperimentRepository()
    graph = compiled_graph()
    graph_config = _graph_config(state)
    context = RunContext(
        repo=repo,
        cancel_event=state.cancel_event,
        on_progress=functools.partial(_on_progress, state),
        llm=llm,
        run_id=state.run_id,
        tracker=make_tracker(_mlflow_parent(graph, graph_config)),
    )
    try:
        should_run, graph_input = _next_input(graph, graph_config, state)
        if should_run:
            with observability.run_tracing():
                # durability="sync": a node's result is on disk before the next
                # node starts, so a crash never loses a finished attempt.
                for values in graph.stream(
                    graph_input, graph_config, context=context, stream_mode="values", durability="sync"
                ):
                    _project(values, state)
        _sync_pause(graph, graph_config, state)
    except Exception as exc:  # the graph could not route to `fail` itself
        _fail_outside_graph(state, exc, repo, graph, graph_config)
    finally:
        if owns_repo:
            repo.close()
    return state


# --- graph plumbing ----------------------------------------------------------


def _mlflow_parent(graph, graph_config: dict) -> str | None:
    """The MLflow parent run of an earlier invocation of this run, if any,
    so a resumed run keeps logging into the same parent."""
    values = graph.get_state(graph_config).values
    return values.get("mlflow_run_id") if values else None


def _graph_config(state: RunState) -> dict:
    return {
        "configurable": {"thread_id": state.run_id},
        "recursion_limit": recursion_limit(state.max_iterations),
        "run_name": "neuralpilot_run",
        "tags": ["neuralpilot"],
        "metadata": {
            "run_id": state.run_id,
            "dataset_fingerprint": state.dataset_fingerprint,
            "code_version": provenance.code_version(),
            "llm_backend": config.LLM_BACKEND,
            "llm_source": config.LLM_SOURCE,
            "llm_model": config.active_llm_model(),
        },
    }


def _initial_input(state: RunState) -> dict:
    return {
        "run_id": state.run_id,
        "dataset_path": str(state.dataset_path),
        "dataset_name": state.dataset_name,
        "dataset_fingerprint": state.dataset_fingerprint,
        "load_report": state.load_report,
        "user_goal": state.user_goal,
        "seed": config.SEED,
        "budget_s": state.budget_s,
        "max_iterations": state.max_iterations,
        "status": "running",
        "iteration": 0,
        "clarification_rounds": 0,
        "holdout_evaluations": 0,
    }


def _next_input(graph, graph_config: dict, state: RunState) -> tuple[bool, object]:
    """(should_run, input) for this call:
    - new run: the initial state;
    - paused with an answer waiting: Command(resume=answer);
    - stopped mid-way (crash, restart): None, i.e. continue from the last checkpoint;
    - finished, or paused with no answer yet: nothing to run."""
    snapshot = graph.get_state(graph_config)
    if not snapshot.values:
        return True, _initial_input(state)
    _project(snapshot.values, state)
    if snapshot.interrupts:
        if state.pending_answer is None:
            return False, None
        answer, state.pending_answer = state.pending_answer, None
        state.status = "running"
        return True, Command(resume=answer)
    return bool(snapshot.next), None


def _sync_pause(graph, graph_config: dict, state: RunState) -> None:
    """If the graph is paused at the clarifying question, show it."""
    snapshot = graph.get_state(graph_config)
    if not snapshot.interrupts:
        return
    payload = snapshot.interrupts[0].value
    state.status = "awaiting_clarification"
    state.clarification_question = payload.get("question") if isinstance(payload, dict) else str(payload)
    state.current_task = "Waiting for target column clarification"


def _project(values: dict, state: RunState) -> None:
    """Copy the UI-relevant parts of graph state onto the RunState."""
    state.status = values.get("status", state.status)
    state.requirement = values.get("requirement") or state.requirement
    state.clarification_question = values.get("clarification_question")
    state.iteration = values.get("iteration", 0)
    state.iteration_logs = list(values.get("iteration_logs") or [])
    state.attempts = list(values.get("attempts") or [])
    state.best = values.get("best")
    state.final_decision = values.get("decision")
    state.holdout_evaluations = values.get("holdout_evaluations", 0)
    state.model_path = values.get("model_path")
    state.final_metrics = values.get("final_metrics")
    state.final_report = values.get("final_report")
    state.error = values.get("error")
    if state.status == "failed":
        state.current_task = "Run failed"


def _on_progress(state: RunState, task: str, tag: str) -> None:
    state.current_task = task
    elapsed = time.monotonic() - state.started_at
    state.activity_log.append(
        {"time": f"{elapsed // 60:02.0f}:{elapsed % 60:04.1f}", "tag": tag, "message": task}
    )


def _fail_outside_graph(state: RunState, exc: Exception, repo: ExperimentRepository, graph, graph_config) -> None:
    """Last resort for errors the graph could not route to its `fail` node
    (e.g. the recursion limit). The run is still persisted as failed."""
    logger.exception("run=%s failed outside the graph", state.run_id)
    state.status = "failed"
    state.current_task = "Run failed"
    state.error = str(exc) or type(exc).__name__
    try:
        values = dict(graph.get_state(graph_config).values)
        values.update(run_id=state.run_id, dataset_name=state.dataset_name, error=state.error)
        target = state.requirement.target_column if state.requirement else None
        meta = extract_meta_features(state.df, target if target in state.df.columns else None).to_dict()
        repo.save_run(reporting.failed_record(values, meta))
    except Exception:
        logger.exception("run=%s could not persist failure record", state.run_id)
