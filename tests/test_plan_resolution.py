"""Phase 4: plan resolution (Deliverable 8.2). Pure logic, no data needed."""

from dataclasses import replace

from neuralpilot.plan_resolution import PlanProfile, PlanRejection, ResolvedPlan, resolve
from neuralpilot.registry import REGISTRY
from neuralpilot.schemas import PlanSpec, PreprocessingStep, TuningStrategy


def _plan(model="LinearModel", encoder="OneHotEncoder", scaler="StandardScaler", params=None):
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component=encoder, params=params or {}),
            PreprocessingStep(category="scaler", component=scaler),
        ],
        model_family=model,
        tuning=TuningStrategy(component="NoTuning", n_trials=0),
    )


def _profile(**overrides):
    fields = dict(problem_type="classification", n_rows=1_000, categorical_cardinality=5)
    fields.update(overrides)
    return PlanProfile(**fields)


def test_exact_name_and_alias_need_no_substitution():
    result = resolve(_plan(model="LogisticRegression", encoder="OHE"), _profile())
    assert isinstance(result, ResolvedPlan)
    assert result.plan.model_family == "LinearModel"
    assert result.plan.preprocessing[0].component == "OneHotEncoder"
    assert result.substitutions == []


def test_unknown_model_uses_the_registry_default_and_records_it():
    result = resolve(_plan(model="MysteryNet"), _profile(), iteration=2)
    assert result.plan.model_family == "RandomForest"
    [record] = result.substitutions
    assert (record.requested, record.reason, record.alternative_used) == ("MysteryNet", "not_in_registry", "RandomForest")
    assert record.iteration == 2 and record.semantic_change is False


def test_over_budget_model_is_replaced_by_a_same_group_alternative():
    result = resolve(_plan(model="GradientBoosting"), _profile(n_rows=300_000))
    assert result.plan.model_family == "HistGradientBoosting"
    [record] = result.substitutions
    assert record.reason == "over_budget" and record.semantic_change is False


def test_glass_box_requirement_replaces_a_black_box_model():
    result = resolve(_plan(model="XGBoost"), _profile(glass_box=True))
    assert result.plan.model_family in ("LinearModel", "DecisionTree")
    [record] = result.substitutions
    assert record.reason == "violates_constraint"
    assert record.semantic_change is True


def test_substituted_component_drops_the_params_written_for_the_old_one():
    result = resolve(_plan(encoder="MysteryEncoder", params={"min_frequency": 3}), _profile(categorical_cardinality=50))
    assert result.plan.preprocessing[0].component == "OneHotEncoder"
    assert result.plan.preprocessing[0].params == {}


def test_task_incompatible_model_is_substituted_with_one_that_supports_the_task(monkeypatch):
    table = REGISTRY["model_family"]
    monkeypatch.setitem(table, "LinearModel", replace(table["LinearModel"], supports=("regression",)))
    result = resolve(_plan(model="LinearModel"), _profile(problem_type="classification"))
    assert result.plan.model_family == "DecisionTree"
    assert result.substitutions[0].reason == "incompatible_task"


def test_plan_is_rejected_with_an_actionable_message_when_nothing_can_run_the_task(monkeypatch):
    table = REGISTRY["model_family"]
    for name, spec in list(table.items()):
        monkeypatch.setitem(table, name, replace(spec, supports=("regression",)))
    result = resolve(_plan(model="LinearModel"), _profile(problem_type="classification"))
    assert isinstance(result, PlanRejection)
    assert "no usable alternative" in result.reason
    assert "Choose one of" in result.reason


def test_encoder_swap_across_groups_is_refused_for_linear_models(monkeypatch):
    """One-hot cannot become ordinal/target encoding for a linear model: that
    changes what the model sees. Only tree models may swap across groups."""
    table = REGISTRY["encoder"]
    monkeypatch.setitem(table, "OneHotEncoder", replace(table["OneHotEncoder"], supports=("regression",)))
    rejected = resolve(_plan(model="LinearModel", encoder="OneHotEncoder"), _profile())
    assert isinstance(rejected, PlanRejection)

    accepted = resolve(_plan(model="GradientBoosting", encoder="OneHotEncoder"), _profile(n_rows=1_000))
    assert isinstance(accepted, ResolvedPlan)
    assert accepted.plan.preprocessing[0].component in ("OrdinalEncoder", "TargetEncoder")
    assert accepted.substitutions[0].semantic_change is True


def test_prior_evidence_breaks_ties_between_alternatives():
    plain = resolve(_plan(model="GradientBoosting"), _profile(n_rows=300_000))
    informed = resolve(
        _plan(model="GradientBoosting"),
        _profile(n_rows=300_000),
        prior_evidence={"XGBoost": 0.99, "HistGradientBoosting": 0.10},
    )
    assert plain.plan.model_family == "HistGradientBoosting"
    assert informed.plan.model_family == "XGBoost"
