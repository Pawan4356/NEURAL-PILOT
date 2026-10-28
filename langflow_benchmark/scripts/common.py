"""Shared settings and helpers for the Langflow vs NeuralPilot comparison.

Nothing here imports NeuralPilot. The settings are copied from
src/neuralpilot/config.py so the comparison stays fair, and the loading and
split code mirrors engine/loader.py + engine/splits.py so all systems score
the same holdout rows.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BENCH_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
WORK_DIR = BENCH_DIR / "work"
DATASETS_FILE = BENCH_DIR / "datasets.json"
INSTRUCTIONS_FILE = BENCH_DIR / "protocol" / "agent_instructions.md"

# Copied from NeuralPilot's config.py
SETTINGS = {
    "holdout_fraction": 0.2,
    "seed": 42,
    "cv_folds": 3,
    "max_iterations": 5,
    "budget_s": 900,
    "imbalance_minority_share": 0.10,
    "latency_sample_rows": 1000,
    "latency_repeats": 3,
    "numeric_coercion_min_share": 0.95,
}

_DELIMITERS = ",;\t|"
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

# Fields every system's final JSON must have (same shape as NeuralPilot's report)
REQUIRED_FIELDS = (
    "system", "dataset_name", "target_column", "problem_type", "primary_metric",
    "model_family", "cv_metrics", "holdout_metrics", "latency_ms_per_1k",
    "constraint_table", "best_iteration", "run_history",
)


def load_datasets() -> list[dict]:
    return json.loads(DATASETS_FILE.read_text(encoding="utf-8"))


def get_dataset(name: str) -> dict:
    for entry in load_datasets():
        if entry["name"] == name:
            entry = dict(entry)
            entry["abs_path"] = str((BENCH_DIR / entry["path"]).resolve())
            return entry
    known = ", ".join(d["name"] for d in load_datasets())
    raise SystemExit(f"Unknown dataset '{name}'. Known: {known}")


def _coerce_numeric(series: pd.Series, decimal: str) -> pd.Series | None:
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return None
    cleaned = non_null.str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
    if decimal == ",":
        cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        cleaned = cleaned.str.replace(",", "", regex=False)
    matches = cleaned.str.match(_NUMBER_RE)
    if matches.mean() < SETTINGS["numeric_coercion_min_share"]:
        return None
    converted = pd.to_numeric(cleaned.where(matches), errors="coerce")
    full = pd.Series(float("nan"), index=series.index, dtype="float64")
    full.loc[converted.index] = converted
    return full


def load_like_neuralpilot(path: str | Path, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Load a CSV the way NeuralPilot does and drop rows with no target.
    Returns the frame (its index is the row ID) and plain-word notes."""
    text = Path(path).read_bytes().decode("utf-8", errors="replace")
    try:
        delimiter = csv.Sniffer().sniff(text[:65536], delimiters=_DELIMITERS).delimiter
    except csv.Error:
        delimiter = ","
    decimal = "," if delimiter == ";" else "."
    df = pd.read_csv(io.StringIO(text), sep=delimiter, decimal=decimal, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    empty = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
    df = df.drop(columns=empty).dropna(how="all")

    notes = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        numeric = _coerce_numeric(df[col], decimal)
        if numeric is not None:
            df[col] = numeric
            notes.append(f"text column '{col}' converted to numbers")
    df = df.reset_index(drop=True)

    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    before = len(df)
    df = df.dropna(subset=[target])
    if len(df) < before:
        notes.append(f"{before - len(df)} rows with a missing target were dropped")
    return df, notes


def holdout_split(df: pd.DataFrame, target: str, problem_type: str) -> tuple[pd.Index, pd.Index]:
    y = df[target]
    stratify = None
    if problem_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2:
        stratify = y
    train_idx, test_idx = train_test_split(
        df.index, test_size=SETTINGS["holdout_fraction"], random_state=SETTINGS["seed"], stratify=stratify
    )
    return pd.Index(train_idx), pd.Index(test_idx)


def extract_json(text: str) -> dict:
    """Return the last JSON object in an agent's reply (fenced or bare)."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for block in reversed(fenced):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    # No fenced block: scan backwards for the last balanced {...} that parses.
    decoder = json.JSONDecoder()
    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "system" in obj:
            return obj
    raise ValueError("No JSON result block found in the reply")


def missing_fields(result: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if f not in result]


def save_result(dataset: str, system: str, result: dict, raw_text: str | None = None) -> Path:
    """Write results/<dataset>/<system>_runN.json (and the raw reply, if any)."""
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = system.lower()
    n = 1 + sum(1 for _ in out_dir.glob(f"{stem}_run*.json"))
    path = out_dir / f"{stem}_run{n}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if raw_text is not None:
        raw_dir = out_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / f"{stem}_run{n}.txt").write_text(raw_text, encoding="utf-8")
    return path
