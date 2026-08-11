# Agentic AutoML — V1

An implementation of the [V1 spec](./AgenticAutoML-V1-Spec.md): four LLM agents that *think*
(select and parametrize trusted components) and a deterministic Execution Engine that *works*
(runs scikit-learn/XGBoost/CatBoost/Optuna/SHAP). The LLM never writes or executes code.

```
Requirement Understanding Agent → Planning Agent → [verify] → Execution Engine
        → Multi-Objective Validation → Reflection Agent → Decision Agent → (refine/replan/stop)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The four agents call an LLM through [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers).
Set an access token before starting a run:

```bash
export HF_TOKEN=hf_...
# optional overrides:
export HF_MODEL="meta-llama/Llama-3.3-70B-Instruct"   # default
export HF_PROVIDER="auto"                              # default
```

## Run

```bash
uv run agentic-automl
```

Then open http://127.0.0.1:8000 — upload a CSV, describe what to predict in plain English,
answer the one clarifying question if asked, and watch the run finalize into a report
(metrics, SHAP feature importance if interpretability was requested, assumptions made, run
history, and a downloadable trained model `.joblib`).

Env vars: `AUTOML_HOST` (default `127.0.0.1`), `AUTOML_PORT` (default `8000`),
`AUTOML_MAX_ITERATIONS` (default `5`).

## Tests

```bash
uv run pytest
```

All deterministic layers (registry, verification, scoring, plateau, repository, execution)
and the four agents are unit-tested; the orchestration loop is integration-tested end-to-end
with a scripted fake LLM client (no network/API key needed for the test suite).

## Project layout

```
src/agentic_automl/
  config.py         tunables (default weights, thresholds, plateau/iteration limits)
  meta_features.py  dataset meta-feature extraction (shared by agents + retrieval)
  registry.py        §4.3 Component Registry + deterministic fallback
  verification.py     §4.1 Plan Verification
  execution.py         §4.2 Execution Engine (sklearn/XGBoost/CatBoost/Optuna/SHAP)
  scoring.py            §4.4 Multi-Objective Validation
  plateau.py             §4.5 Plateau Detection
  repository.py           §5 SQLite experiment log + nearest-neighbor retrieval
  llm_client.py            Hugging Face chat-completions client, structured JSON + retry
  schemas.py               shared pydantic contracts (agent I/O, execution results)
  agents/                  the four LLM agents (§3)
  orchestrator.py          §6 end-to-end loop
  web/                     §7/§9 minimal web UI
```

## V1 scope

See [§2 of the spec](./AgenticAutoML-V1-Spec.md#2-v1-scope-boundary) for what's in vs.
explicitly deferred (multi-strategy A/B/C planning, real vector DB retrieval, web search,
cross-iteration ensembling, Pareto-frontier optimization, RAG over papers/docs).
