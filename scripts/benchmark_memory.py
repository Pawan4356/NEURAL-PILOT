"""Memory-on versus memory-off benchmark (Phase 7, Deliverable 7 acceptance).

The question: when a similar dataset's tuned hyperparameters seed the search,
does the target reach nearly the same CV score with fewer HPO trials?

For each seed the script measures three things with the real engine:
1. the source dataset tuned with --trials-off trials (this is the "past run"
   whose best parameters are stored in memory);
2. the target tuned from scratch with --trials-off trials (memory off);
3. the target tuned with --trials-on trials, starting from the source's best
   parameters (memory on).

It prints the CV metric of (2) and (3), their ratio, and the trials and seconds
each one used. Nothing is hard-coded: the numbers are what this run measured.

    python scripts/benchmark_memory.py --source a.csv --source-target Churn \
        --target b.csv --target-target Churn --family GradientBoosting --seeds 1 2 3
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from neuralpilot.engine import AttemptBudget, load_dataset, prepare_dataset, run_attempt
from neuralpilot.schemas import ConstraintWeights, PlanSpec, PreprocessingStep, RequirementSpec, TuningStrategy


def make_requirement(target: str, problem_type: str) -> RequirementSpec:
    return RequirementSpec(
        target_column=target,
        target_confidence=1.0,
        problem_type=problem_type,
        constraint_weights=ConstraintWeights(accuracy=1.0, latency=0.0, interpretability=0.0),
        weight_confidence=1.0,
    )


def make_plan(family: str, trials: int) -> PlanSpec:
    return PlanSpec(
        preprocessing=[
            PreprocessingStep(category="encoder", component="OneHotEncoder"),
            PreprocessingStep(category="scaler", component="StandardScaler"),
        ],
        model_family=family,
        tuning=TuningStrategy(component="OptunaTPE", n_trials=trials),
    )


def tune(df, requirement, family: str, seed: int, trials: int, warm_start: dict | None = None):
    prepared = prepare_dataset(df, requirement, seed=seed)
    started = time.perf_counter()
    outcome = run_attempt(
        make_plan(family, trials),
        prepared,
        budget=AttemptBudget(max_trials=trials, hpo_timeout_s=None),
        warm_start_params=warm_start,
    )
    attempt = outcome.result.attempt
    return {
        "cv": attempt.cv_score,
        "trials": attempt.n_trials_completed,
        "seconds": time.perf_counter() - started,
        "best_params": dict(attempt.best_params),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark memory-on against memory-off warm starts.")
    parser.add_argument("--source", type=Path, required=True, help="CSV of the similar past dataset")
    parser.add_argument("--source-target", required=True)
    parser.add_argument("--target", type=Path, required=True, help="CSV of the dataset being tuned now")
    parser.add_argument("--target-target", required=True)
    parser.add_argument("--problem-type", default="classification", choices=["classification", "regression"])
    parser.add_argument("--family", default="GradientBoosting")
    parser.add_argument("--trials-off", type=int, default=20, help="trials without memory (and for the source run)")
    parser.add_argument("--trials-on", type=int, default=5, help="trials with memory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args(argv)

    source_df = load_dataset(args.source).df
    target_df = load_dataset(args.target).df
    source_req = make_requirement(args.source_target, args.problem_type)
    target_req = make_requirement(args.target_target, args.problem_type)

    ratios, rows = [], []
    for seed in args.seeds:
        past = tune(source_df, source_req, args.family, seed, args.trials_off)
        off = tune(target_df, target_req, args.family, seed, args.trials_off)
        on = tune(target_df, target_req, args.family, seed, args.trials_on, warm_start=past["best_params"])
        ratio = on["cv"] / off["cv"] if off["cv"] else float("nan")
        ratios.append(ratio)
        rows.append((seed, off, on, ratio))
        print(
            f"seed={seed} memory_off cv={off['cv']:.4f} trials={off['trials']} {off['seconds']:.1f}s | "
            f"memory_on cv={on['cv']:.4f} trials={on['trials']} {on['seconds']:.1f}s | ratio={ratio:.4f}"
        )

    print(f"mean ratio (memory_on / memory_off CV) = {statistics.mean(ratios):.4f} over {len(ratios)} seed(s)")
    print(
        f"trials: memory_off={args.trials_off} memory_on={args.trials_on}; "
        f"mean seconds off={statistics.mean(r[1]['seconds'] for r in rows):.1f} "
        f"on={statistics.mean(r[2]['seconds'] for r in rows):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
