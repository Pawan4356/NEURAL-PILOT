"""Copy finished NeuralPilot runs into results/ in the shared comparison shape.

Reads runs/experiments.sqlite3 in read-only mode; NeuralPilot is not touched.

    python scripts/export_neuralpilot.py telco_churn --latest 3
    python scripts/export_neuralpilot.py telco_churn --run-id <run_id>

--latest picks the newest finished runs whose dataset file name matches the
dataset's path in datasets.json.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from common import PROJECT_ROOT, RESULTS_DIR, get_dataset

DB_PATH = PROJECT_ROOT / "runs" / "experiments.sqlite3"


def wall_time_s(activity_json: str | None) -> float | None:
    """Last activity timestamp ('mm:ss.s' or 'h:mm:ss.s') as seconds."""
    entries = json.loads(activity_json or "[]")
    if not entries:
        return None
    seconds = 0.0
    for part in entries[-1]["time"].split(":"):
        seconds = seconds * 60 + float(part)
    return round(seconds, 1)


def to_shared_shape(report: dict, wall_s: float | None) -> dict:
    plan = report.get("best_plan") or {}
    steps = {p.get("category"): p.get("component") for p in plan.get("preprocessing") or []}
    tuning = plan.get("tuning") or {}
    return {
        "system": "NeuralPilot",
        "run_id": report["run_id"],
        "dataset_name": report["dataset_name"],
        "target_column": report["target_column"],
        "problem_type": report["problem_type"],
        "primary_metric": report["primary_metric"],
        "model_family": report["model_family"],
        "best_hyperparameters": report.get("best_hyperparameters") or {},
        "best_plan": {
            "encoder": steps.get("encoder", ""),
            "scaler": steps.get("scaler", ""),
            "feature_engineering": [f.get("name") for f in plan.get("feature_engineering") or []],
            "tuning": tuning.get("component", ""),
        },
        "cv_metrics": report["cv_metrics"],
        "holdout_metrics": report["holdout_metrics"],
        "latency_ms_per_1k": report["latency_ms_per_1k"],
        "constraint_table": [
            {k: row.get(k) for k in ("name", "kind", "threshold", "measured", "passed")}
            for row in report.get("constraint_table") or []
        ],
        "best_iteration": report["best_iteration"],
        "run_history": [
            {k: h.get(k) for k in (
                "iteration", "plan_summary", "primary_metric", "cv_mean", "cv_std",
                "latency_ms_per_1k", "decision", "decision_rationale",
            )}
            for h in report.get("run_history") or []
        ],
        "assumptions": report.get("assumptions") or [],
        "final_decision_rationale": report.get("final_decision_rationale", ""),
        "wall_time_s": wall_s,
        "artifacts": {
            "model_path": report.get("model_path"),
            "mlflow_run_id": report.get("mlflow_run_id"),
            "shap": bool(report.get("shap_summary")),
        },
        "failed_tool_calls": 0,
        "manual_interventions": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id")
    group.add_argument("--latest", type=int)
    args = parser.parse_args()

    entry = get_dataset(args.dataset)
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    if args.run_id:
        rows = conn.execute("SELECT * FROM runs WHERE run_id = ?", (args.run_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM runs WHERE status = 'finalized' AND dataset_name = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (Path(entry["path"]).name, args.latest),
        ).fetchall()
    if not rows:
        raise SystemExit("No matching finished NeuralPilot runs found.")

    out_dir = RESULTS_DIR / entry["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if not row["final_report_json"]:
            print(f"skip {row['run_id']}: status {row['status']}, no final report")
            continue
        result = to_shared_shape(json.loads(row["final_report_json"]), wall_time_s(row["activity_log_json"]))
        path = out_dir / f"neuralpilot_{row['run_id'][:8]}.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"saved {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
