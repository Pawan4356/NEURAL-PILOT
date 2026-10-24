"""Phase 4: candidate generation (Deliverable 8.3)."""

from neuralpilot import config
from neuralpilot.candidates import build_candidates
from neuralpilot.engine import plan_signature
from neuralpilot.plan_resolution import PlanProfile, resolve
from neuralpilot.schemas import PlanSpec, PreprocessingStep, TuningStrategy


def _plan(model="LinearModel"):
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        model_family=model,
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
    )


def _profile(n_rows=1_000, **overrides):
    return PlanProfile(problem_type="classification", n_rows=n_rows, categorical_cardinality=5, **overrides)


def _build(plan, profile, **kwargs):
    planner = resolve(plan, profile).plan
    return build_candidates(planner, profile, iteration=1, tried=kwargs.pop("tried", set()), **kwargs)


def test_planner_plan_comes_first_and_a_baseline_is_added():
    candidates, _ = _build(_plan("RandomForest"), _profile())
    assert candidates[0].source == "planner"
    assert candidates[0].plan.model_family == "RandomForest"
    sources = [c.source for c in candidates]
    assert "baseline" in sources
    baseline = next(c for c in candidates if c.source == "baseline")
    assert baseline.plan.model_family == "LinearModel"


def test_baseline_moves_on_when_the_planner_already_chose_linear_model():
    candidates, _ = _build(_plan("LinearModel"), _profile())
    baseline = next(c for c in candidates if c.source == "baseline")
    assert baseline.plan.model_family == "DecisionTree"


def test_large_data_adds_histogram_boosting_and_small_data_adds_extra_trees():
    large, _ = _build(_plan("LinearModel"), _profile(n_rows=config.HIST_GB_MIN_ROWS + 1))
    assert any(c.plan.model_family == "HistGradientBoosting" for c in large if c.source == "profile")

    small, _ = _build(_plan("LinearModel"), _profile(n_rows=config.SMALL_DATA_ROWS))
    assert any(c.plan.model_family == "ExtraTrees" for c in small if c.source == "profile")


def test_the_candidate_cap_is_respected_and_the_planner_is_never_dropped():
    candidates, _ = _build(_plan("LinearModel"), _profile(n_rows=100), max_candidates=2)
    assert len(candidates) == 2
    assert candidates[0].source == "planner"


def test_signatures_already_executed_are_never_proposed_again():
    profile = _profile()
    planner = resolve(_plan("RandomForest"), profile).plan
    baseline_signature = plan_signature(
        planner.model_copy(update={"model_family": "LinearModel"}), categorical_cardinality=5
    )
    candidates, _ = build_candidates(
        planner, profile, iteration=2, tried={baseline_signature}
    )
    assert all(c.plan.model_family != "LinearModel" for c in candidates if c.source != "planner")
    signatures = [plan_signature(c.plan, categorical_cardinality=5) for c in candidates]
    assert len(set(signatures)) == len(signatures)


def test_glass_box_requirement_keeps_black_box_families_out_of_the_candidates():
    candidates, _ = _build(_plan("GradientBoosting"), _profile(glass_box=True))
    for candidate in candidates:
        assert candidate.plan.model_family in ("LinearModel", "DecisionTree")


def test_a_similar_past_plan_is_evaluated_as_a_warm_start_only():
    # XGBoost is not proposed by any other rule, so the warm start is the one that adds it.
    memory_plan = _plan("XGBoost")
    candidates, _ = _build(_plan("LinearModel"), _profile(), memory_plans=[memory_plan], max_candidates=5)
    memory = [c for c in candidates if c.source == "memory"]
    assert memory and memory[0].plan.model_family == "XGBoost"
    assert memory[0].plan.rationale == "warm start from a similar past experiment"


def test_substitutions_made_for_challengers_are_reported():
    planner = resolve(_plan("LinearModel"), _profile()).plan
    memory_plan = _plan("MysteryNet")
    _, substitutions = build_candidates(
        planner, _profile(), iteration=1, tried=set(), memory_plans=[memory_plan], max_candidates=5
    )
    assert any(s.requested == "MysteryNet" and s.source == "memory" for s in substitutions)
