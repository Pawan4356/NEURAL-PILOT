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
The four agents call an LLM through [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers)
by default, or through the [Anthropic Claude API](https://docs.anthropic.com) when `NEURALPILOT_LLM_SOURCE=claude`.
A Hugging Face token (or an Anthropic API key for Claude) is required for interactive runs. The test suite does not need a
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

# Or switch the source to Claude. The agents then use ANTHROPIC_API_KEY instead of HF_TOKEN.
export NEURALPILOT_LLM_SOURCE=claude                   # default: huggingface
export ANTHROPIC_API_KEY=sk-ant-...
export CLAUDE_MODEL="claude-sonnet-5-5"                # default

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
| `NEURALPILOT_LLM_SOURCE` | `huggingface` | Provider the agents call: `huggingface` or `claude` |
| `ANTHROPIC_API_KEY` | None | Anthropic API key; required when `NEURALPILOT_LLM_SOURCE=claude` |
| `CLAUDE_MODEL` | `claude-haiku-5-5` | Claude model used when `NEURALPILOT_LLM_SOURCE=claude` |
| `NEURALPILOT_LLM_BACKEND` | `langchain` | `langchain` (LangChain chat model) or `hf` (original client, rollback only; Hugging Face only) |
| `NEURALPILOT_LLM_STRUCTURED_OUTPUT` | `auto` | `auto` tries the provider's structured output and falls back to JSON repair; `on` or `off` |
| `NEURALPILOT_LLM_STRUCTURED_METHOD` | `function_calling` | How structure is requested: `function_calling` or `json_schema` |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | `false` / None / `neuralpilot-dev` | Optional LangSmith tracing; payloads are redacted before they are sent |
| `NEURALPILOT_HOST` | `127.0.0.1` | Web server host |
| `NEURALPILOT_PORT` | `8000` | Web server port |
| `NEURALPILOT_MAX_ITERATIONS` | `5` | Maximum refinement iterations |
| `NEURALPILOT_LOG_LEVEL` | `INFO` | Logging level, such as `INFO` or `DEBUG` |
| `NEURALPILOT_RUN_BUDGET_S` | `900` | Wall-clock budget per run; the best attempt so far is kept when it runs out |
| `NEURALPILOT_SEED` | `42` | Seed for splits, samplers and estimators |
| `NEURALPILOT_HPO_MAX_TRIALS` | `50` | Cap on Optuna trials per attempt (larger plan values are clamped) |
| `NEURALPILOT_HPO_TIMEOUT_S` | `300` | Optuna timeout per attempt |
| `NEURALPILOT_HPO_MAX_ROWS` | `50000` | Training rows used for tuning/CV on large data; the final fit uses all of them |
| `NEURALPILOT_MAX_ROWS` / `NEURALPILOT_MAX_COLS` | `2000000` / `500` | Upload size limits |
| `NEURALPILOT_LEAKAGE_ACTION` | `drop` | `drop` excludes columns that predict the target on their own; `warn` only reports them |
| `NEURALPILOT_MIN_SIMILARITY` | `0.5` | Past experiments less similar than this are not shown to the planner |
| `NEURALPILOT_MAX_CANDIDATES` | `3` | Plans measured per iteration: the planner's plan plus deterministic challengers (set `1` for planner-only) |
| `NEURALPILOT_MLFLOW` | `on` | Log every attempt and the final model to MLflow (`off` disables it) |
| `MLFLOW_TRACKING_URI` | `sqlite:///runs/mlflow.db` | MLflow tracking store; view with `mlflow ui --backend-store-uri <uri>` |
| `NEURALPILOT_MLFLOW_ARTIFACTS` | `runs/mlartifacts` | Artifact root for MLflow |

### Evaluation protocol

Each run makes one stratified (classification) or random (regression) holdout split, or a
time/group split when the goal asks for one. Every plan is measured by cross-validation on
the training part only, using a single primary metric (`f1_weighted`, `f1_macro` for
imbalanced targets, or `r2`) for tuning, selection and reporting. The best attempt across
iterations is finalized and scored once on the holdout. Identifier, constant, raw timestamp
and free-text columns, and columns that predict the target almost perfectly on their own, are
excluded from the features and listed in the report's assumptions. Plans already tried in a
run are not repeated. `POST /api/runs/{run_id}/cancel` stops a run after the current trial and
keeps its best attempt.

Each iteration measures the planner's plan and up to two deterministic challengers: an
interpretable baseline, a family suited to the data size, and the plan of a similar past run
(as a warm start only). Components the registry cannot use are replaced by a ranked
alternative, and every substitution is recorded in the report. The user's hard constraints
(for example a latency limit, or "must be interpretable") are checked on measured values; a
feasible attempt always beats an infeasible one, and the report shows the constraint table, the
Pareto front and, when nothing is feasible, the closest attempt for each violated limit.

The web page shows the current task while a run is active. The terminal logs each phase,
iteration, verification retry, and tuning trial. A complete plan, training, validation,
reflection, and decision cycle advances the iteration counter.

### GPU training

GPU mode is automatic by default when calling `run_pipeline` or `engine.run_attempt`: if `nvidia-smi` reports an
available NVIDIA GPU, XGBoost and CatBoost training and tuning use CUDA. Scikit-learn model
families continue to use CPU because they do not provide CUDA implementations. Callers can
override this with `use_gpu=False` or request it explicitly with `use_gpu=True`; an explicit
request falls back to CPU and records the reason in the run fallbacks when no GPU is available.
The selected device is recorded in the model-family diagnostics.

### Orchestration and tracing

The run loop is a LangGraph `StateGraph` (`graph/`). Each step is a node; routing between
them is a set of plain functions in `graph/routing.py`. Graph state is checkpointed after
every node in `runs/graph.sqlite3` (one thread per run id), which is what lets the run pause
for the clarifying question and continue from the last finished step after a crash. LLM
calls go through a LangChain chat model with transport retries and a JSON-repair fallback.
With `LANGSMITH_TRACING=true`, each run is one LangSmith trace with a span per graph node,
agent and LLM call; field names like `df`/`rows`/`api_key`, tabular data, SHAP feature names
and secret values are masked before anything is sent.

With `NEURALPILOT_MLFLOW=on` (default), each run is one MLflow parent run with its holdout
metrics and final model, and each candidate attempt is a child run with its parameters and
cross-validated metrics (`cv_*`). Tracking errors are logged and never fail a run.

Generated files are written under the project root: uploaded CSVs go to `data/`, the
SQLite experiment log is `runs/experiments.sqlite3`, graph checkpoints are in
`runs/graph.sqlite3`, final model artifacts are saved as `runs/models/<run_id>.joblib`, and
MLflow's store and artifacts are in `runs/mlflow.db` and `runs/mlartifacts/`.

## Project layout

```
src/neuralpilot/
  config.py         tunables (default weights, thresholds, plateau/iteration limits)
  meta_features.py  dataset meta-feature extraction (shared by agents + retrieval)
  registry.py       component registry and deterministic fallback
  verification.py   deterministic plan verification
  engine/           execution engine: loader, splits, preprocessing, feature_engineering,
                    model_factory, hpo, training, evaluation, explainability, tracking, runner
  execution.py      run_pipeline(): one-call compatibility wrapper over engine/
  profiling.py      column roles (id/constant/datetime/text) and single-feature leakage scan
  provenance.py     code version and library versions recorded with each run
  scoring.py        weighted multi-objective validation
  plateau.py        deterministic plateau detection
  repository.py     SQLite experiment log and nearest-neighbor retrieval
  llm_client.py     LLMClient protocol, original Hugging Face client, default-client selection
  langchain_llm.py  LangChain client (Hugging Face or Claude): structured output, JSON repair, transport backoff
  observability.py  LangSmith tracing setup and payload redaction
  selection.py      best-attempt rule and deterministic next-untried-plan fallback
  reporting.py      final report and experiment records built from graph state
  schemas.py        shared Pydantic contracts for agent I/O and execution results
  agents/           the four LLM agents
  graph/            LangGraph orchestration: state, nodes, routing, build (checkpointer)
  orchestrator.py   façade over the graph: create_run, run_to_completion, apply_clarification
  web/              minimal upload, clarification, progress, and report UI
```

## Reading path for the code

If you are learning the project structure, read it in the same order the run executes:

1. `schemas.py`, `config.py`, and `meta_features.py` define the contracts, defaults, and dataset summaries.
2. `agents/requirement_agent.py`, `planning_agent.py`, `reflection_agent.py`, and `decision_agent.py` define the four LLM-only reasoning steps.
3. `registry.py`, `verification.py`, `profiling.py`, `engine/`, `scoring.py`, and `plateau.py` implement deterministic behavior (`engine/runner.py` shows how the engine pieces fit together).
4. `repository.py` stores and retrieves previous experiments.
5. `orchestrator.py` is the entry point for a run; `graph/routing.py` shows the whole control flow at a glance and `graph/nodes.py` has one function per step.
6. `web/app.py` and `web/static/index.html` expose the upload, clarification, progress, final report, and model download UI.

## Scope and technical reference

Read [NeuralPilot-Spec.md](./NeuralPilot-Spec.md) to learn the complete project structure and
internal workings: the four agent contracts, execution boundaries, verification rules, fallback
behavior, scoring, plateau detection, experiment retrieval, orchestration flow, and build order.
It also records what is explicitly deferred, including parallel A/B/C planning, a real vector
database, web search, cross-iteration ensembling, Pareto-frontier optimization, and full RAG.
