"""Central configuration for Neural Pilot.

All tunables that the spec treats as fixed defaults (§3 weight defaults,
plateau thresholds, iteration caps, etc.) live here so every module reads
from one place instead of hardcoding literals in multiple files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
DB_PATH = RUNS_DIR / "experiments.sqlite3"

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
}

# --- Plateau detection (§4.5) ---
PLATEAU_SCORE_DELTA_PCT = 2.0
PLATEAU_LOOKBACK_ITERS = 2

# --- Orchestration loop (§6) ---
MAX_ITERATIONS = int(os.environ.get("NEURALPILOT_MAX_ITERATIONS", "5"))

# --- Knowledge layer (§5) ---
RETRIEVAL_TOP_K = 3 # Top similar models (already trained)

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
