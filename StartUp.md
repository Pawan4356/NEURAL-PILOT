# Neural Pilot Startup

## Install

From the project directory, install the dependencies with:

```bash
uv sync
```

The four agents call an LLM through Hugging Face Inference Providers. Set a valid Hugging
Face access token before starting a run. The project `.env` contains the same settings and
can be loaded into the current shell:

```bash
set -a
source .env
set +a
```

Or configure the LLM variables directly:

```bash
export HF_TOKEN=hf_...
export HF_MODEL="meta-llama/Llama-3.3-70B-Instruct"  # default
export HF_PROVIDER="auto"                              # default
```

## Start

```bash
uv run neuralpilot
```

Open http://127.0.0.1:8000, upload a CSV, and describe what you want to predict in plain
English. Answer the target-column question if the system asks one. The run then produces
metrics, optional SHAP feature importance, assumptions, run history, and a downloadable
trained model.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | None | Hugging Face access token; required for agent calls |
| `HF_MODEL` | `meta-llama/Llama-3.3-70B-Instruct` | Hugging Face model |
| `HF_PROVIDER` | `auto` | Hugging Face inference provider |
| `NEURALPILOT_HOST` | `127.0.0.1` | Web server host |
| `NEURALPILOT_PORT` | `8000` | Web server port |
| `NEURALPILOT_MAX_ITERATIONS` | `5` | Maximum refinement iterations |
| `NEURALPILOT_LOG_LEVEL` | `INFO` | Logging level, such as `INFO` or `DEBUG` |

## Progress Logging

The web page shows the current task while a run is active. The terminal logs each phase,
iteration, verification retry, and hyperparameter-tuning trial. For detailed logs:

```bash
NEURALPILOT_LOG_LEVEL=DEBUG uv run neuralpilot
```

During tuning, progress appears as:

```text
Hyperparameter tuning: trial 1/25
Hyperparameter tuning: trial 2/25
Final training of XGBoost
Generating SHAP feature importance
```

The iteration counter is updated after a complete plan, training, validation, reflection,
and decision cycle. Trial-level task messages show progress while that cycle is running.

## Troubleshooting

If port `8000` is already in use, stop the existing server with `Ctrl+C`, or start on another
port:

```bash
NEURALPILOT_PORT=8001 uv run neuralpilot
```

Then open http://127.0.0.1:8001.

For a quick test run with one orchestration iteration:

```bash
NEURALPILOT_MAX_ITERATIONS=1 NEURALPILOT_LOG_LEVEL=INFO uv run neuralpilot
```

## Tests

The test suite does not require a network connection or Hugging Face API key:

```bash
uv run pytest
```