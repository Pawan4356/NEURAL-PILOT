"""Dataset loader (engine component 1).

Turns an uploaded CSV into a DataFrame without silently garbling it:
- sniffs the delimiter (`,` `;` tab `|`) instead of assuming a comma
- tries a short list of encodings instead of failing on non-UTF-8 files
- coerces text columns that are really numbers ("1,000", "3.5") to numeric
- rejects files whose shape or header make no sense, with a message the
  user can act on

Every decision is recorded in `LoadedDataset.report` so the final report
can say exactly how the file was read.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .. import config

_SNIFF_BYTES = 64 * 1024
_DELIMITERS = [",", ";", "\t", "|"]
_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


class DatasetLoadError(ValueError):
    """Raised with a message that tells the user how to fix their file."""


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    fingerprint: str  # SHA-256 of the file bytes + loader options
    report: dict = field(default_factory=dict)


def _decode(raw: bytes) -> tuple[str, str]:
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    # latin-1 decodes any byte sequence, so this is unreachable in practice.
    raise DatasetLoadError("Could not decode the file as UTF-8, CP-1252 or Latin-1 text.")


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITERS)).delimiter
    except csv.Error:
        pass
    # Sniffer gives up on some valid files (e.g. quoted fields everywhere);
    # pick the candidate that splits the header into the most fields while
    # giving the same field count on the next lines.
    lines = [line for line in sample.splitlines() if line.strip()][:20]
    best, best_fields = ",", 1
    for delimiter in _DELIMITERS:
        counts = {len(row) for row in csv.reader(lines, delimiter=delimiter)}
        if len(counts) == 1:
            n_fields = counts.pop()
            if n_fields > best_fields:
                best, best_fields = delimiter, n_fields
    return best


def _coerce_numeric(series: pd.Series, decimal: str) -> pd.Series | None:
    """Return a numeric version of a text column, or None if it isn't one."""
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return None
    cleaned = non_null.str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
    if decimal == ",":
        cleaned = cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        cleaned = cleaned.str.replace(",", "", regex=False)
    matches = cleaned.str.match(_NUMBER_RE)
    if matches.mean() < config.NUMERIC_COERCION_MIN_SHARE:
        return None
    converted = pd.to_numeric(cleaned.where(matches), errors="coerce")
    full = pd.Series(float("nan"), index=series.index, dtype="float64")
    full.loc[converted.index] = converted
    return full


def _looks_like_junk_header(columns: pd.Index) -> bool:
    names = [str(c).strip() for c in columns]
    return names == [str(i) for i in range(len(names))]


def load_dataset(
    path: Path | str,
    *,
    max_rows: int = config.MAX_ROWS,
    max_cols: int = config.MAX_COLS,
) -> LoadedDataset:
    path = Path(path)
    raw = path.read_bytes()
    if not raw.strip():
        raise DatasetLoadError("The uploaded file is empty.")

    text, encoding = _decode(raw)
    delimiter = _sniff_delimiter(text[:_SNIFF_BYTES])
    # A ';'-separated file almost always uses ',' as the decimal mark.
    decimal = "," if delimiter == ";" else "."

    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, decimal=decimal, low_memory=False)
    except (pd.errors.ParserError, ValueError) as exc:
        raise DatasetLoadError(f"Could not parse the file as CSV (delimiter {delimiter!r}): {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    # Drop fully empty trailing columns/rows that spreadsheet exports add.
    unnamed_empty = [c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()]
    df = df.drop(columns=unnamed_empty).dropna(how="all")

    if df.shape[1] <= 1:
        raise DatasetLoadError(
            "The file was read as a single column. Make sure it is a CSV with a header row and "
            "values separated by commas, semicolons, tabs or '|'."
        )
    if _looks_like_junk_header(df.columns):
        raise DatasetLoadError(
            "The header row looks like column numbers (0, 1, 2, ...) rather than names. "
            "Add a header row with a descriptive name for each column."
        )
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise DatasetLoadError(f"Column names must be unique; duplicated: {', '.join(dupes)}.")
    if len(df) > max_rows:
        raise DatasetLoadError(f"The file has {len(df):,} rows; the limit is {max_rows:,}.")
    if df.shape[1] > max_cols:
        raise DatasetLoadError(f"The file has {df.shape[1]} columns; the limit is {max_cols}.")
    if len(df) < 10:
        raise DatasetLoadError(f"The file has only {len(df)} data rows; at least 10 are needed.")

    coerced: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        numeric = _coerce_numeric(series, decimal)
        if numeric is not None:
            df[col] = numeric
            coerced.append(col)

    df = df.reset_index(drop=True)
    options = f"sep={delimiter!r};encoding={encoding};decimal={decimal!r}"
    fingerprint = hashlib.sha256(raw + options.encode()).hexdigest()
    report = {
        "delimiter": delimiter,
        "encoding": encoding,
        "decimal": decimal,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "numeric_coerced_columns": coerced,
        "dropped_empty_columns": unnamed_empty,
    }
    return LoadedDataset(df=df, fingerprint=fingerprint, report=report)
