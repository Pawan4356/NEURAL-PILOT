# Neural Pilot

Neural Pilot is a machine-learning workflow where four LLM agents plan and reason, while a
deterministic Execution Engine performs the actual work. The LLM selects and parametrizes
trusted components from scikit-learn, XGBoost, CatBoost, Optuna, and SHAP; it never writes or
executes code.

The application accepts a CSV dataset and a plain-English prediction goal, then runs this loop:

![Agentic Planning Execution Loop](<media/Agentic Planning Execution Loop.png>)

The complete architecture, agent contracts, deterministic rules, data flow, and deferred scope
are documented in [NeuralPilot-Spec.md](./NeuralPilot-Spec.md). This README is the practical
guide for understanding the product and running it locally.

## What the product does

1. You upload a CSV and describe what you want to predict.
2. The Requirement Understanding Agent identifies the target, task type, and objective weights.
3. The Planning Agent proposes one trusted pipeline using dataset metadata and similar past runs.
4. Deterministic verification checks the plan before execution and applies logged fallbacks when needed.
5. The Execution Engine preprocesses data, trains and tunes a model, validates it, and records each step.
6. Reflection and Decision agents either accept the result, refine it, replan it, or stop it.
7. The UI reports the saved model artifact, metrics, assumptions, run history, and optional SHAP explanations.

Only one clarification can be asked during a run, and only when the target-column confidence is low.

## Prerequisites

Install [uv](https://docs.astral.sh/uv/) and use Python supported by the project configuration.
The four agents call an LLM through [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers),
so a Hugging Face access token is required for interactive runs. The test suite does not need a
network connection or an API key.

## Commands

All common setup, startup, debugging, and verification commands are collected here:

```bash
# First-time setup: run from the project directory.
uv sync

# Configure the LLM. Loading .env is convenient when the project has one.
set -a
source .env
set +a

# Or configure the variables directly instead of loading .env.
export HF_TOKEN=hf_...
export HF_MODEL="meta-llama/Llama-3.3-70B-Instruct"  # default
export HF_PROVIDER="auto"                              # default

# Normal startup.
uv run neuralpilot
# Open http://127.0.0.1:8000, upload a CSV, and describe the prediction goal.

# Detailed terminal progress logging.
NEURALPILOT_LOG_LEVEL=DEBUG uv run neuralpilot

# Start on another port when 8000 is already in use.
NEURALPILOT_PORT=8001 uv run neuralpilot
# Open http://127.0.0.1:8001

# Quick smoke run with one orchestration iteration.
NEURALPILOT_MAX_ITERATIONS=1 NEURALPILOT_LOG_LEVEL=INFO uv run neuralpilot

# Run the complete test suite. No network or HF_TOKEN is required.
uv run pytest
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | None | Hugging Face access token; required for agent calls |
| `HF_MODEL` | `meta-llama/Llama-3.3-70B-Instruct` | Hugging Face model |
| `HF_PROVIDER` | `auto` | Hugging Face inference provider |
| `NEURALPILOT_HOST` | `127.0.0.1` | Web server host |
| `NEURALPILOT_PORT` | `8000` | Web server port |
| `NEURALPILOT_MAX_ITERATIONS` | `5` | Maximum refinement iterations |
| `NEURALPILOT_LOG_LEVEL` | `INFO` | Logging level, such as `INFO` or `DEBUG` |

The web page shows the current task while a run is active. The terminal logs each phase,
iteration, verification retry, and tuning trial. A complete plan, training, validation,
reflection, and decision cycle advances the iteration counter.

Generated files are written under the project root: uploaded CSVs go to `data/`, the
SQLite experiment log is `runs/experiments.sqlite3`, and final model artifacts are saved
as `runs/models/<run_id>.joblib`.

## Project layout

```
src/neuralpilot/
  config.py         tunables (default weights, thresholds, plateau/iteration limits)
  meta_features.py  dataset meta-feature extraction (shared by agents + retrieval)
  registry.py       component registry and deterministic fallback
  verification.py   deterministic plan verification
  execution.py      execution engine for preprocessing, training, tuning, and SHAP
  scoring.py        weighted multi-objective validation
  plateau.py        deterministic plateau detection
  repository.py     SQLite experiment log and nearest-neighbor retrieval
  llm_client.py     Hugging Face chat-completions client with structured JSON and retry
  schemas.py        shared Pydantic contracts for agent I/O and execution results
  agents/           the four LLM agents
  orchestrator.py   end-to-end planning and execution loop
  web/              minimal upload, clarification, progress, and report UI
```

## Reading path for the code

If you are learning the project structure, read it in the same order the run executes:

1. `schemas.py`, `config.py`, and `meta_features.py` define the contracts, defaults, and dataset summaries.
2. `agents/requirement_agent.py`, `planning_agent.py`, `reflection_agent.py`, and `decision_agent.py` define the four LLM-only reasoning steps.
3. `registry.py`, `verification.py`, `execution.py`, `scoring.py`, and `plateau.py` implement deterministic behavior.
4. `repository.py` stores and retrieves previous experiments.
5. `orchestrator.py` wires the full loop together.
6. `web/app.py` and `web/static/index.html` expose the upload, clarification, progress, final report, and model download UI.

## Scope and technical reference

Read [NeuralPilot-Spec.md](./NeuralPilot-Spec.md) to learn the complete project structure and
internal workings: the four agent contracts, execution boundaries, verification rules, fallback
behavior, scoring, plateau detection, experiment retrieval, orchestration flow, and build order.
It also records what is explicitly deferred, including parallel A/B/C planning, a real vector
database, web search, cross-iteration ensembling, Pareto-frontier optimization, and full RAG.
