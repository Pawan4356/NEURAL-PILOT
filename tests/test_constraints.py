"""Phase 5: constraint model, checks, feasibility-first selection and the
run-level validation report (Deliverable 9.2-9.4)."""

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from neuralpilot.graph.nodes import _keep_searching_while_infeasible
from neuralpilot.orchestrator import create_run, run_to_completion
from neuralpilot.repository import ExperimentRepository
from neuralpilot.schemas import (
    AttemptResult,
    AttemptSummary,
    ConstraintResult,
    ConstraintWeights,
    Constraint,
    DecisionOutput,
    MetricSummary,
    PlanSpec,
    PreprocessingStep,
    ReflectionOutput,
    RequirementSpec,
    ResourceUsage,
    SplitSpec,
    TuningStrategy,
)
from neuralpilot.selection import is_better
from neuralpilot.validation import check_constraints, is_feasible, pareto_front, requires_glass_box, summarize_run
from neuralpilot.validation.report import closest_per_violation


def _attempt(model="LinearModel", metric="f1_weighted", value=0.9, std=0.01, latency=10.0, train_s=1.0, model_bytes=1_000_000):
    return AttemptResult(
        model_family=model,
        primary_metric=metric,
        cv={metric: MetricSummary(mean=value, std=std, folds=[value] * 3)},
        split=SplitSpec(strategy="stratified"),
        resources=ResourceUsage(train_s=train_s, hpo_s=0.0, latency_ms_per_1k=latency, model_bytes=model_bytes),
    )


def _summary(signature, cv_mean, *, feasible=True, latency=10.0, interp=1.0, results=()):
    return AttemptSummary(
        signature=signature,
        iteration=1,
        model_family="LinearModel",
        preprocessing=[],
        feature_engineering=[],
        tuning="NoTuning(0)",
        primary_metric="f1_weighted",
        cv_mean=cv_mean,
        cv_std=0.01,
        composite_score=0.5,
        latency_ms_per_1k=latency,
        interp_norm=interp,
        feasible=feasible,
        constraint_results=list(results),
    )


def _result(name, passed, measured, threshold, kind="hard"):
    return ConstraintResult(name=name, kind=kind, threshold=threshold, measured=measured, passed=passed, explanation="")


# --- the constraint model ----------------------------------------------------


def test_threshold_must_match_the_constraint_kind():
    with pytest.raises(ValidationError):
        Constraint(name="interpretability", threshold=5)
    with pytest.raises(ValidationError):
        Constraint(name="max_latency_ms_per_1k", threshold="fast")


def test_requirement_drops_unknown_constraints_and_notes_it():
    spec = RequirementSpec(
        target_column="y",
        target_confidence=0.9,
        problem_type="classification",
        constraint_weights=ConstraintWeights(),
        weight_confidence=0.9,
        constraints=[
            {"name": "max_latency_ms_per_1k", "kind": "hard", "threshold": 50},
            {"name": "be_nice", "kind": "hard", "threshold": 1},
        ],
    )
    assert [c.name for c in spec.constraints] == ["max_latency_ms_per_1k"]
    assert any("ignored 1 unsupported constraint" in note for note in spec.assumptions)


@pytest.mark.parametrize(
    "constraint, attempt_kwargs, passes",
    [
        (Constraint(name="max_latency_ms_per_1k", threshold=50), {"latency": 40}, True),
        (Constraint(name="max_latency_ms_per_1k", threshold=50), {"latency": 80}, False),
        (Constraint(name="max_train_seconds", threshold=5), {"train_s": 4}, True),
        (Constraint(name="max_train_seconds", threshold=5), {"train_s": 9}, False),
        (Constraint(name="max_model_mb", threshold=2), {"model_bytes": 1_000_000}, True),
        (Constraint(name="max_model_mb", threshold=2), {"model_bytes": 5_000_000}, False),
        (Constraint(name="max_cv_std", threshold=0.02), {"std": 0.01}, True),
        (Constraint(name="max_cv_std", threshold=0.02), {"std": 0.05}, False),
        (Constraint(name="min_primary_metric", threshold=0.85), {"value": 0.9}, True),
        (Constraint(name="min_primary_metric", threshold=0.85), {"value": 0.8}, False),
    ],
)
def test_each_numeric_constraint_passes_and_fails(constraint, attempt_kwargs, passes):
    [result] = check_constraints([constraint], _attempt(**attempt_kwargs), interp_class="linear")
    assert result.passed is passes
    assert result.explanation


def test_minimum_metric_is_a_maximum_for_error_metrics():
    rmse_limit = Constraint(name="min_primary_metric", threshold=0.5)
    [ok] = check_constraints([rmse_limit], _attempt(metric="rmse", value=0.4), interp_class="linear")
    [bad] = check_constraints([rmse_limit], _attempt(metric="rmse", value=0.7), interp_class="linear")
    assert ok.passed and not bad.passed


def test_latency_explanation_states_the_measured_overshoot():
    [result] = check_constraints(
        [Constraint(name="max_latency_ms_per_1k", threshold=50)], _attempt(latency=82.0), interp_class="linear"
    )
    assert result.explanation == "latency 82.0 ms/1k > 50.0 ms/1k limit (+64%)"


def test_glass_box_is_decided_by_model_class():
    glass = Constraint(name="interpretability", threshold="glass_box")
    assert check_constraints([glass], _attempt(), interp_class="linear")[0].passed
    assert check_constraints([glass], _attempt(), interp_class="tree")[0].passed
    assert not check_constraints([glass], _attempt(), interp_class="xgboost")[0].passed
    assert requires_glass_box([glass])


def test_soft_constraints_are_reported_but_do_not_make_an_attempt_infeasible():
    results = check_constraints(
        [
            Constraint(name="max_latency_ms_per_1k", kind="soft", threshold=1),
            Constraint(name="max_latency_ms_per_1k", kind="hard", threshold=1000),
        ],
        _attempt(latency=40),
        interp_class="linear",
    )
    assert [r.passed for r in results] == [False, True]
    assert is_feasible(results)


# --- selection and the run-level report -------------------------------------


def test_a_feasible_attempt_beats_a_better_infeasible_one():
    infeasible_best = _summary("a", 0.95, feasible=False)
    feasible = _summary("b", 0.80, feasible=True)
    assert is_better(feasible, infeasible_best, n_folds=3)
    assert not is_better(infeasible_best, feasible, n_folds=3)


def test_among_feasible_attempts_the_metric_still_decides():
    assert is_better(_summary("b", 0.9), _summary("a", 0.8), n_folds=3)


def test_closest_attempt_is_reported_when_nothing_is_feasible():
    far = _summary("far", 0.9, feasible=False, results=[_result("max_latency_ms_per_1k", False, 120, 50)])
    near = _summary("near", 0.85, feasible=False, results=[_result("max_latency_ms_per_1k", False, 60, 50)])
    closest = closest_per_violation(far, [far, near])
    assert closest["max_latency_ms_per_1k"]["signature"] == "near"
    summary = summarize_run([far, near], far)
    assert summary["best_feasible"] is False and summary["feasible_count"] == 0


def test_pareto_front_keeps_attempts_nobody_beats_on_every_objective():
    a = _summary("a", 0.90, latency=10, interp=1.0)
    b = _summary("b", 0.80, latency=50, interp=0.5)  # worse than a on everything
    c = _summary("c", 0.95, latency=60, interp=0.4)  # best metric, but slower and less interpretable
    assert pareto_front([a, b, c]) == ["a", "c"]


def test_the_guard_keeps_refining_while_the_best_attempt_is_infeasible():
    infeasible = SimpleNamespace(
        validation=SimpleNamespace(feasible=False, constraint_results=[_result("max_latency_ms_per_1k", False, 90, 50)])
    )
    state = {"best": infeasible, "max_iterations": 5, "budget_s": 900, "budget_started_at": None}
    accept = DecisionOutput(decision="accept", rationale="looks good")

    overridden = _keep_searching_while_infeasible(accept, {**state, "budget_started_at": time.time()}, iteration=1)
    assert overridden.decision == "refine" and overridden.forced
    assert "max_latency_ms_per_1k" in overridden.rationale

    last_iteration = _keep_searching_while_infeasible(accept, {**state, "max_iterations": 1}, iteration=1)
    assert last_iteration.decision == "accept"

    feasible = SimpleNamespace(validation=SimpleNamespace(feasible=True, constraint_results=[]))
    assert _keep_searching_while_infeasible(accept, {**state, "best": feasible}, iteration=1) is accept


# --- end to end: an impossible limit ends in an honest, unacceptable report --


class ScriptedClient:
    def __init__(self, responses_by_schema):
        self.responses_by_schema = responses_by_schema
        self.indices = {k: 0 for k in responses_by_schema}

    def complete_json(self, system_prompt, user_prompt, schema, *, max_retries=2):
        responses = self.responses_by_schema[schema]
        idx = min(self.indices[schema], len(responses) - 1)
        self.indices[schema] += 1
        return responses[idx]


def test_impossible_hard_constraint_finishes_with_an_unacceptable_report(tmp_path):
    rng = np.random.default_rng(3)
    x1, x2 = rng.normal(size=300), rng.normal(size=300)
    y = (x1 + x2 > 0).astype(int)
    path = tmp_path / "data.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "y": y}).to_csv(path, index=False)

    plan = PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        model_family="LinearModel",
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
    )
    requirement = RequirementSpec(
        target_column="y",
        target_confidence=0.95,
        problem_type="classification",
        constraint_weights=ConstraintWeights(accuracy=1.0, latency=0.0, interpretability=0.0),
        weight_confidence=0.95,
        constraints=[Constraint(name="max_latency_ms_per_1k", kind="hard", threshold=1e-6)],
    )
    client = ScriptedClient(
        {
            RequirementSpec: [requirement],
            PlanSpec: [plan],
            ReflectionOutput: [ReflectionOutput(weakest_block="model_family", rationale="r")],
            DecisionOutput: [DecisionOutput(decision="accept", rationale="fine")],
        }
    )
    repo = ExperimentRepository(db_path=tmp_path / "exp.sqlite3")
    try:
        run = create_run(path, "data.csv", "predict y")
        run.max_iterations = 2
        state = run_to_completion(run, repo=repo, llm=client)
    finally:
        repo.close()

    assert state.status == "finalized", state.error
    report = state.final_report
    assert report["acceptable"] is False
    failing = [r for r in report["constraint_table"] if not r["passed"]]
    assert [r["name"] for r in failing] == ["max_latency_ms_per_1k"]
    assert report["final_decision"] == "stop" and report["final_decision_forced"] is True  # iteration cap
    assert state.iteration_logs[0].decision == "refine" and state.iteration_logs[0].decision_forced
