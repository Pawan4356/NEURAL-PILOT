# Neural Pilot Implementation Spec

**Status:** Implemented baseline, kept as the technical reference for the current code
**Scope:** This document defines exactly what to build. Anything not listed under the project scope is explicitly deferred — do not implement it, do not scaffold for it beyond the noted interface seams.

---

## 1. Guiding Principle

Agents think. The Execution Engine works.

The LLM never writes or executes code. It only selects and parametrizes trusted, pre-built modules (scikit-learn, XGBoost, CatBoost, Optuna, SHAP). All execution is deterministic, sandboxed, and logged.

---

## 2. Scope Boundary

**In scope:**
- Single primary strategy per planning pass (no parallel A/B/C strategy generation)
- Weighted-sum multi-objective scoring (accuracy / latency / interpretability)
- Linear closed loop: plan → verify → execute → validate → reflect → decide → (refine/replan/stop)
- SQLite/JSON experiment log with nearest-neighbor meta-feature retrieval (no vector DB)
- Deterministic plan verification (rule-based, not LLM-judged)
- Deterministic component fallback (category-based lookup, not similarity search)
- One clarifying question max, only when target-column confidence is low
- Minimal FastAPI web UI for upload, clarification, progress polling, final report, and model download

**Explicitly deferred to V2+ (do not build now):**
- Multi-strategy (A/B/C) parallel planning
- Real vector database / embeddings-based retrieval
- Web search integration
- Cross-iteration ensembling
- Pareto-frontier multi-objective optimization
- Full Retrieval-Augmented Planning against papers/docs

---

## 3. Agents (Reasoning Layer)

Exactly four LLM agents. Use these names consistently everywhere (code, prompts, logs, UI):

1. **Requirement Understanding Agent**
   Input: raw user prompt + dataset meta-features (column names, dtypes, row/col count — never raw rows)
   Output (structured JSON):
   - `target_column` (string) + `confidence` (0–1)
   - `problem_type` (classification/regression, inferred from target dtype/cardinality)
   - `constraint_weights`: `{accuracy: float, latency: float, interpretability: float}` (default `0.7/0.15/0.15` unless explicit priority language detected)
   - `weight_confidence` (0–1)

   **Clarification rule:** if `target_column confidence < 0.7`, agent must ask the user one clarifying question via the app (see §7) before proceeding. If `weight_confidence < 0.7`, do NOT ask — silently fall back to default weights and note the assumption in the final report.

2. **Planning Agent**
   Input: requirement spec + dataset meta-features + top-k similar past experiments (from experiment log, §6)
   Output: one structured pipeline plan — preprocessing steps, feature engineering steps, candidate model family/families, tuning strategy — each with a one-line rationale.
   On refine (loop iteration >1): receives Reflection Agent's target block and revises only that block, keeping the rest of the plan fixed.

3. **Reflection Agent**
   Input: validation report (accuracy/latency/interpretability scores + per-block diagnostics from Execution Engine)
   Output: `weakest_block` (string, e.g. `"feature_engineering"`, `"model_family"`, `"hyperparameters"`) + rationale.

4. **Decision Agent**
   Input: current composite score, score history (all prior iterations), weakest_block, iteration count, plateau flag
   Output: one of `accept | refine | replan | stop` + rationale.
   **Deterministic overrides the agent must respect (hardcode these as guardrails, not just prompt instructions):**
   - If `iteration_count >= max_iterations` → force `stop`
   - If plateau condition met (see §5.5) → force `stop`
   - Otherwise the agent chooses freely among `accept/refine/replan`

---

## 4. Execution Layer (Deterministic, No LLM)

### 4.1 Plan Verification (runs before every execution)
Hardcoded checks, all must pass or the plan is rejected and sent back to Planning Agent with the failure reason. The orchestrator retries planning/verification up to three times before failing the run.
- `target_column` exists in dataset
- Target has ≥2 unique values (classification) or is numeric (regression)
- Class imbalance ratio flagged (warn, not block) if >10:1
- `n_features > n_rows / 10` → warning logged, not blocking
- Missing-value % on target column — block if >80%
- Every component named in the plan exists in the Component Registry (§4.3); if not, trigger fallback

### 4.2 Execution Engine
Runs preprocessing → feature engineering → hyperparameter tuning → final training → evaluation exactly as verified. Logs per-block diagnostics (inputs, outputs, timing, errors) into the run state/final report, then persists a summary to the experiment log. Wraps scikit-learn/XGBoost/CatBoost/Optuna/SHAP as callable modules — agents select and parametrize, engine executes.

### 4.3 Component Registry + Fallback
Registry: components tagged by category (`encoder`, `scaler`, `model_family`, `tuner`).
Fallback rule (deterministic decision tree per category, not similarity search):
- `encoder`: unsupported → if categorical cardinality ≤ 10, use `OrdinalEncoder`; else `OneHotEncoder`
- `scaler`: unsupported → default to `StandardScaler`
- `model_family`: unsupported → default to registry name `RandomForest`, which builds `RandomForestClassifier` or `RandomForestRegressor` by task type
- `tuner`: unsupported → default to registry name `OptunaRandom`, an Optuna random sampler with fixed trial budget
Every fallback is logged as `{requested: X, used: Y, reason: "unsupported"}` — the run never fails on this basis.

Implemented registry names:
- `encoder`: `OneHotEncoder`, `OrdinalEncoder`
- `scaler`: `MinMaxScaler`, `RobustScaler`, `StandardScaler`
- `model_family`: `CatBoost`, `DecisionTree`, `GradientBoosting`, `LinearModel`, `RandomForest`, `XGBoost`
- `tuner`: `NoTuning`, `OptunaRandom`, `OptunaTPE`

Feature-engineering steps are handled by a small execution-engine registry, not the component fallback registry. Supported names are `PolynomialFeatures`, `SelectKBest`, and `PCA`; unknown feature-engineering steps are skipped with a diagnostic warning rather than failing the run.

### 4.4 Multi-Objective Validation
```
score = w_acc * accuracy_norm + w_lat * (1 - latency_norm) + w_interp * interp_norm
```
- `accuracy_norm`: task metric normalized 0–1 (accuracy/F1 for classification, normalized R²/RMSE for regression)
- `latency_norm`: inference time normalized against a rolling max seen this run
- `interp_norm`: lookup table by model class — `{linear: 1.0, tree: 1.0, random_forest: 0.6, gbm: 0.6, xgboost: 0.4, catboost: 0.4, deep: 0.1}`
- Weights come from Requirement Understanding Agent output (§3.1)
- If interpretability requested explicitly, generate SHAP summary regardless of score — SHAP generation is not gated by the score itself
- Classification metrics are `accuracy` and weighted `f1`; regression metrics are `r2` and `rmse`.

### 4.5 Plateau Detection (deterministic)
Plateau = TRUE if either:
- Composite score improvement < 2% over the last 2 consecutive iterations, OR
- Same `weakest_block` identified in 2 consecutive Reflection Agent outputs with no net score gain
When plateau = TRUE, force Decision Agent output to `stop`.

---

## 5. Knowledge Layer

- **Experiment repository:** SQLite (or JSON file, pick SQLite for query simplicity) storing per-run: dataset meta-feature vector, plan used, scores, weakest_block history, final decision.
- **Retrieval method:** nearest-neighbor (cosine similarity) over a small hand-built meta-feature vector: `[n_rows, n_cols, n_classes, missing_pct, categorical_ratio]`. No embeddings, no vector DB library.
- **Interface seam for V2:** wrap retrieval behind a `retrieve_similar_experiments(meta_features, k)` function so swapping in pgvector/embeddings later doesn't change any agent-facing contract.
- **Artifacts:** uploaded datasets are stored in `data/`, the SQLite log is `runs/experiments.sqlite3`, and accepted/stopped runs save a final scikit-learn pipeline at `runs/models/<run_id>.joblib`.

---

## 6. End-to-End Loop

```
1. User uploads dataset + plain-English goal
2. Requirement Understanding Agent → spec (target, problem_type, weights)
   → if target confidence low: ask ONE clarifying question, else proceed
3. Planning Agent → plan (using top-k similar past experiments)
4. Plan Verification (deterministic) → pass/fail
   → fail: back to step 3 with failure reason, up to 3 verification attempts
5. Execution Engine runs plan → metrics + per-block diagnostics + model artifact candidate
6. Multi-Objective Validation → composite score
7. Reflection Agent → weakest_block
8. Decision Agent → accept | refine | replan | stop
   (plateau/max-iteration guardrails can force "stop" regardless of agent output)
9. If refine/replan: loop to step 3 with new context
   If accept/stop: finalize
10. Final output: saved model, metrics, SHAP explanation (if applicable), assumptions made, run history summary
```

---

## 7. User-Facing Clarification

Only one interaction point beyond initial upload: a single clarifying question if target-column confidence is low. No other mid-run prompts. All other uncertainty (constraint weights, component fallback) is resolved silently with defaults and surfaced in the final report as "Assumptions Made."

---

## 8. Build Order (suggested for Claude CLI)

1. Component Registry + fallback logic (pure Python, no LLM) — foundation everything else calls
2. Plan Verification rules (pure Python)
3. Execution Engine wrapping sklearn/XGBoost/CatBoost/Optuna/SHAP with logging
4. Multi-Objective Validation scoring function
5. Plateau detection function
6. Experiment repository (SQLite) + nearest-neighbor retrieval
7. Four LLM agents (structured JSON I/O, start with Requirement Understanding Agent since it's the entry point)
8. Orchestration loop wiring 1–7 together per §6
9. Minimal UI/CLI for upload + clarifying question + final report display

Each numbered item should be a separate, testable module before wiring the loop together.
