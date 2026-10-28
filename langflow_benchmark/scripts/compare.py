"""Build the side-by-side table from results/<dataset>/*.json.

    python scripts/compare.py              # every dataset
    python scripts/compare.py telco_churn

Writes results/comparison.md and prints it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

from common import RESULTS_DIR

SYSTEMS = ("NeuralPilot", "Claude", "Langflow")


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _primary_holdout(r: dict) -> float | None:
    metric = r.get("primary_metric")
    holdout = r.get("holdout_metrics") or {}
    return _num(holdout.get(metric)) if metric in holdout else None


def _primary_cv(r: dict) -> tuple[float | None, float | None]:
    cv = (r.get("cv_metrics") or {}).get(r.get("primary_metric")) or {}
    return _num(cv.get("mean")), _num(cv.get("std"))


def _hard_constraints(r: dict) -> tuple[int, int]:
    rows = [c for c in r.get("constraint_table") or [] if c.get("kind", "hard") == "hard"]
    return sum(1 for c in rows if c.get("passed") is True), len(rows)


def _mean_sd(values: list[float], digits: int = 4) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def _column(runs: list[dict]) -> dict[str, str]:
    if not runs:
        return {}
    ok = [r for r in runs if r.get("status", "ok") == "ok"]
    holdout = [_primary_holdout(r) for r in ok]
    cv = [_primary_cv(r) for r in ok]
    met = [_hard_constraints(r) for r in ok]
    metrics = sorted({r.get("primary_metric") for r in ok if r.get("primary_metric")})
    families = sorted({str(r.get("model_family")) for r in ok if r.get("model_family")})
    artifacts = [r.get("artifacts") or {} for r in ok]

    def share(key: str) -> str:
        n = sum(1 for a in artifacts if a.get(key))
        return "Yes" if artifacts and n == len(artifacts) else ("No" if n == 0 else f"{n}/{len(artifacts)} runs")

    spread = [v for v in holdout if v is not None]
    return {
        "Runs (ok / total)": f"{len(ok)} / {len(runs)}",
        "Primary metric": ", ".join(metrics) or "-",
        "Model family chosen": ", ".join(families) or "-",
        "Holdout primary metric": _mean_sd(holdout),
        "CV mean ± std (avg over runs)": (
            f"{_mean_sd([m for m, _ in cv])} (fold std {_mean_sd([s for _, s in cv], 3)})" if cv else "-"
        ),
        "Latency (ms per 1k rows)": _mean_sd([_num(r.get("latency_ms_per_1k")) for r in ok], 1),
        "Hard constraints met": ", ".join(f"{p}/{t}" for p, t in met) or "-",
        "Iterations used": ", ".join(str(len(r.get("run_history") or [])) for r in ok) or "-",
        "Wall time (s)": _mean_sd([_num(r.get("wall_time_s")) for r in runs], 0),
        "Spread of holdout metric across runs": (
            f"{max(spread) - min(spread):.4f} (max - min)" if len(spread) > 1 else "needs 2+ runs"
        ),
        "Failed runs / failed tool calls / manual fixes": (
            f"{len(runs) - len(ok)} / {sum(r.get('failed_tool_calls', 0) or 0 for r in runs)}"
            f" / {sum(r.get('manual_interventions', 0) or 0 for r in runs)}"
        ),
        "Saved model / MLflow record / SHAP": f"{share('model_path')} / {share('mlflow_run_id')} / {share('shap')}",
    }


def build_table(dataset: str) -> str:
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((RESULTS_DIR / dataset).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        runs[result.get("system", path.stem.split("_")[0])].append(result)
    columns = {s: _column(runs.get(s, [])) for s in SYSTEMS}
    rows = next((list(c) for c in columns.values() if c), [])
    lines = [f"## {dataset}", "", "| Criterion | " + " | ".join(SYSTEMS) + " |", "|---" * (len(SYSTEMS) + 1) + "|"]
    for row in rows:
        lines.append(f"| {row} | " + " | ".join(columns[s].get(row, "no runs") for s in SYSTEMS) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?")
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir())
    if not datasets:
        raise SystemExit("No results yet.")
    report = "# Langflow vs Claude vs NeuralPilot\n\n" + "\n\n".join(build_table(d) for d in datasets) + "\n"
    out = RESULTS_DIR / "comparison.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
