"""Central configuration for Neural Pilot.

All tunables that the spec treats as fixed defaults (§3 weight defaults,
plateau thresholds, iteration caps, etc.) live here so every module reads
from one place instead of hardcoding literals in multiple files.

Importing this module loads the project's `.env` first, so settings there
apply without sourcing it by hand. Variables already set in the real
environment always win, so `NEURALPILOT_LLM_SOURCE=nvidia uv run ...`
still overrides the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
DB_PATH = RUNS_DIR / "experiments.sqlite3"
# Users, sessions and run ownership. Kept apart from the experiment store so
# that guidance (which reads every finished run) never touches personal data.
ACCOUNTS_DB_PATH = RUNS_DIR / "accounts.sqlite3"
SESSION_TTL_S = 7 * 24 * 3600
SESSION_COOKIE_SECURE = os.environ.get("NEURALPILOT_COOKIE_SECURE", "0") == "1"
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv(path: Path = DOTENV_PATH) -> list[str]:
    """Put `KEY=VALUE` lines from `path` into os.environ and return the names
    that were set. Existing environment variables are never overwritten, and a
    missing file is not an error. Set NEURALPILOT_DOTENV=off to skip the file."""
    if os.environ.get("NEURALPILOT_DOTENV", "on").strip().lower() in ("off", "0", "false"):
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    loaded = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


DOTENV_LOADED = load_dotenv()

DATA_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# --- Requirement Understanding Agent defaults (§3.1) ---
DEFAULT_CONSTRAINT_WEIGHTS = {
    "accuracy": 0.7, 
    "latency": 0.15, # We calculate Normalized latency score...
    "interpretability": 0.15
}
TARGET_CONFIDENCE_THRESHOLD = 0.7 #  What to predict
WEIGHT_CONFIDENCE_THRESHOLD = 0.7 #  What to prioritize

# --- Plan verification thresholds (§4.1) ---
CLASS_IMBALANCE_WARN_RATIO = 8.0
FEATURE_TO_ROW_WARN_RATIO = 0.1  # n_features > n_rows / 10
TARGET_MISSING_BLOCK_PCT = 70.0

# --- Multi-objective validation (§4.4) ---
# These are configured scores (domain/design assumption), not values calculated from the dataset.
# We CAN use dynamic values for interpretability as well but it would be too tediuos...
INTERPRETABILITY_TABLE = {
    "linear": 1.0, # Coefficients are relatively easy to inspect
    "tree": 1.0, # Decision rules can be visualized
    "random_forest": 0.6, # Many trees make the overall decision harder to explain
    "gbm": 0.6, # Ensemble of sequential trees
    "xgboost": 0.4, # Powerful ensemble; less directly interpretable
    "catboost": 0.4, # Similar ensemble complexity
    "deep": 0.1, # Many parameters/layers; difficult to directly explain
    "hist_gbm": 0.6, # Same family as GradientBoosting, binned features
    "extra_trees": 0.6, # Randomised forest; same explanation burden as random_forest
}
# Interpretability classes that count as "glass box" for a hard constraint.
GLASS_BOX_CLASSES = ("linear", "tree")

# --- Plateau detection (§4.5) ---
PLATEAU_SCORE_DELTA_PCT = 2.0
PLATEAU_LOOKBACK_ITERS = 2



def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


# --- Orchestration loop (§6) ---
MAX_ITERATIONS = _env_int("NEURALPILOT_MAX_ITERATIONS", 5)
SEED = _env_int("NEURALPILOT_SEED", 42)
RUN_BUDGET_S = _env_float("NEURALPILOT_RUN_BUDGET_S", 900.0)  # wall-clock budget per run
# Range the user may pick for a run's time limit on the new-run form.
RUN_BUDGET_MIN_S = 15 * 60.0
RUN_BUDGET_MAX_S = 120 * 60.0

# --- Dataset loading limits ---
MAX_ROWS = _env_int("NEURALPILOT_MAX_ROWS", 2_000_000)
MAX_COLS = _env_int("NEURALPILOT_MAX_COLS", 500)
NUMERIC_COERCION_MIN_SHARE = 0.95  # share of non-null values that must parse as numbers

# --- Column profiling and leakage (audit G2) ---
ID_UNIQUE_RATIO = 0.95
LEAKAGE_ACTION = os.environ.get("NEURALPILOT_LEAKAGE_ACTION", "drop")  # "drop" or "warn"
LEAKAGE_SCORE_THRESHOLD = 0.98  # single-feature CV AUC / R² / accuracy
LEAKAGE_ACCURACY_MARGIN = 0.3  # accuracy must also beat the base rate by this much
LEAKAGE_SAMPLE_ROWS = 20_000

# --- Evaluation protocol (audit G6-G8) ---
HOLDOUT_FRACTION = 0.2
CV_FOLDS = 3
IMBALANCE_MINORITY_SHARE = 0.10  # below this, use class_weight + a macro metric
# Fixed reference so latency_norm is comparable across iterations and runs.
LATENCY_REFERENCE_MS_PER_1K = _env_float("NEURALPILOT_LATENCY_REFERENCE_MS_PER_1K", 100.0)
LATENCY_SAMPLE_ROWS = 1_000
LATENCY_REPEATS = 3

# --- Compute budgets (audit G9) ---
HPO_MAX_TRIALS = _env_int("NEURALPILOT_HPO_MAX_TRIALS", 50)
HPO_TIMEOUT_S = _env_float("NEURALPILOT_HPO_TIMEOUT_S", 300.0)
HPO_MAX_ROWS = _env_int("NEURALPILOT_HPO_MAX_ROWS", 50_000)
POLYNOMIAL_MAX_INPUT_WIDTH = 50  # PolynomialFeatures only below this encoded width
POLYNOMIAL_MAX_DEGREE = 2

# --- Knowledge layer (§5) ---
RETRIEVAL_TOP_K = 3 # Top similar models (already trained)
MIN_SIMILARITY = _env_float("NEURALPILOT_MIN_SIMILARITY", 0.5)

# --- Orchestration graph (LangGraph, Phase 3) ---
GRAPH_DB_PATH = RUNS_DIR / "graph.sqlite3"  # checkpoints; thread_id = run_id
MODELS_DIR = RUNS_DIR / "models"
MAX_PLAN_ATTEMPTS = 3  # plan -> verify -> dedupe tries per iteration
MAX_CLARIFICATION_ROUNDS = 2

# --- Candidate plans per iteration (Phase 4, Deliverable 8.3-8.4) ---
# The planner's plan plus up to MAX_CANDIDATES - 1 deterministic challengers.
# The iteration's HPO trial cap and timeout are shared between them.
MAX_CANDIDATES = _env_int("NEURALPILOT_MAX_CANDIDATES", 3)
MIN_TRIALS_PER_CANDIDATE = 5
# Profile rules for generated challengers.
HIST_GB_MIN_ROWS = 50_000  # large data: histogram-based boosting is much faster
SMALL_DATA_ROWS = 5_000  # small data: randomised forests are a cheap, strong check
# Share of a candidate's budget needed before a challenger is started.
CHALLENGER_MIN_BUDGET_S = 30.0

# --- Experiment tracking (Phase 6, Deliverable 9.1) ---
MLFLOW_ENABLED = os.environ.get("NEURALPILOT_MLFLOW", "on").strip().lower() in ("on", "1", "true")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{RUNS_DIR / 'mlflow.db'}")
MLFLOW_ARTIFACT_ROOT = os.environ.get("NEURALPILOT_MLFLOW_ARTIFACTS", str(RUNS_DIR / "mlartifacts"))
MLFLOW_EXPERIMENT = "neuralpilot"

# --- Documentation index (Phase 8, Deliverable 7) ---
# Built offline by `python -m neuralpilot.knowledge.ingest`. Until it exists,
# the planner gets no passages and runs exactly as before.
KNOWLEDGE_DIR = RUNS_DIR / "knowledge"
KNOWLEDGE_SOURCES_DIR = PROJECT_ROOT / "knowledge_sources"
RAG_ENABLED = os.environ.get("NEURALPILOT_RAG", "on").strip().lower() in ("on", "1", "true")
RAG_MAX_PASSAGES = 6  # passages shown to the planner
RAG_FETCH_N = 24  # nearest chunks read before the task and registry filters

# --- LLM transport (Phase 2) ---
# "langchain" routes calls through a LangChain chat model; "hf" is the
# original huggingface_hub client, kept as a rollback.
LLM_BACKEND = os.environ.get("NEURALPILOT_LLM_BACKEND", "langchain")
# "auto" tries the provider's structured output first and falls back to the
# JSON-repair loop; "on" never falls back; "off" always uses the repair loop.
LLM_STRUCTURED_OUTPUT = os.environ.get("NEURALPILOT_LLM_STRUCTURED_OUTPUT", "auto")
# How the provider is asked for structure: "function_calling" (tool call),
# "json_schema" (response_format), or "none" to let the provider decide.
# Which one a given HF provider honours varies, so Hugging Face is told
# explicitly; NVIDIA NIM picks the method itself from the schema and warns
# when it is told which one to use, so its default is "none".
LLM_STRUCTURED_METHOD = os.environ.get("NEURALPILOT_LLM_STRUCTURED_METHOD", "").strip().lower()
LLM_TRANSPORT_RETRIES = _env_int("NEURALPILOT_LLM_TRANSPORT_RETRIES", 3)
LLM_BACKOFF_BASE_S = 1.0
LLM_BACKOFF_MAX_S = 20.0


# --- LLM source: which provider the agents call ---
# "huggingface" (default) uses HF Inference Providers; "nvidia" uses NVIDIA NIM
# (build.nvidia.com by default, or a self-hosted NIM via NVIDIA_BASE_URL).
LLM_SOURCE = os.environ.get("NEURALPILOT_LLM_SOURCE", "huggingface").strip().lower()
LLM_SOURCES = ("huggingface", "nvidia")

# Default structured-output method per source, used when
# NEURALPILOT_LLM_STRUCTURED_METHOD is unset.
_DEFAULT_STRUCTURED_METHOD = {"huggingface": "function_calling", "nvidia": "none"}
if not LLM_STRUCTURED_METHOD:
    LLM_STRUCTURED_METHOD = _DEFAULT_STRUCTURED_METHOD.get(LLM_SOURCE, "none")


# --- LLM client (Hugging Face Inference Providers, OpenAI-compatible) ---
@dataclass
class LLMConfig:
    api_key: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    model: str = field(
        default_factory=lambda: os.environ.get(
            "HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"
        )
    )
    provider: str = field(default_factory=lambda: os.environ.get("HF_PROVIDER", "auto"))
    max_tokens: int = 1500
    temperature: float = 0.2
    timeout: float = 90.0


LLM_CONFIG = LLMConfig()


# --- LLM client (NVIDIA NIM, OpenAI-compatible), used when NEURALPILOT_LLM_SOURCE=nvidia ---
@dataclass
class NvidiaConfig:
    api_key: str | None = field(default_factory=lambda: os.environ.get("NVIDIA_API_KEY"))
    model: str = field(
        default_factory=lambda: os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    )
    # None uses the hosted endpoint (https://integrate.api.nvidia.com/v1);
    # set it to reach a self-hosted NIM, e.g. http://localhost:8000/v1.
    base_url: str | None = field(default_factory=lambda: os.environ.get("NVIDIA_BASE_URL") or None)
    max_tokens: int = 1500
    temperature: float = 0.2
    timeout: float = 90.0


NVIDIA_CONFIG = NvidiaConfig()


def active_llm_model() -> str:
    """Model name of the source selected by NEURALPILOT_LLM_SOURCE."""
    return NVIDIA_CONFIG.model if LLM_SOURCE == "nvidia" else LLM_CONFIG.model
