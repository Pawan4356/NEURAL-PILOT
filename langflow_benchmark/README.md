# Langflow vs NeuralPilot benchmark

This folder tests a Langflow agent (and optionally Claude) against NeuralPilot
under **the same rules**: same dataset, same request text, same test rows,
same metric, same number of attempts and the same LLM. Each system's final
answer is saved in the same JSON shape, so `compare.py` can put the numbers
side by side.

It is kept apart from NeuralPilot on purpose:

- Langflow runs from its own Python 3.12 environment, `langflow_benchmark/.venv`.
- Langflow keeps its database and settings in `langflow_benchmark/.langflow`.
- Nothing here changes NeuralPilot's code. The scripts only **read**
  NeuralPilot's `.env` (for the NVIDIA key and model) and
  `runs/experiments.sqlite3` (to export finished runs).

Every command is given for **Windows (PowerShell)** and **Ubuntu 24.04
(bash)**. The only difference is where Python lives:

| | Windows | Ubuntu |
|---|---|---|
| This folder's Python (Langflow env) | `.venv\Scripts\python.exe` | `.venv/bin/python` |
| NeuralPilot's Python (repo root env) | `..\.venv\Scripts\python.exe` | `../.venv/bin/python` |

---

## Part 1 - One-time setup (about 10 minutes)

### 1.1 Prerequisites

- [ ] The NeuralPilot repo is cloned and NeuralPilot itself works
      (`uv sync` in the repo root, then `uv run neuralpilot` opens
      http://127.0.0.1:8000). See the main [README](../README.md).
- [ ] `uv` is installed (`uv --version`). If not:

      Windows:
      ```powershell
      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
      ```
      Ubuntu (then open a new terminal):
      ```bash
      curl -LsSf https://astral.sh/uv/install.sh | sh
      ```
- [ ] NeuralPilot's `.env` (repo root) has these set:
      ```
      NEURALPILOT_LLM_SOURCE=nvidia
      NVIDIA_API_KEY=nvapi-...
      NVIDIA_MODEL=openai/gpt-oss-20b
      ```
      Langflow uses the **same** key and model, so both systems run on the
      same LLM. Don't use `meta/llama-3.3-70b-instruct`: NVIDIA retired it on
      2026-08-26 and it now returns "410 Gone".

### 1.2 Install Langflow into this folder

From the **repo root**:

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File langflow_benchmark\setup.ps1
```
Ubuntu:
```bash
bash langflow_benchmark/setup.sh
```

This creates `langflow_benchmark/.venv` with the pinned versions in
[requirements.txt](requirements.txt) (Langflow 1.13.0, scikit-learn, XGBoost,
CatBoost, pandas). uv downloads Python 3.12 itself if needed. Keep the
versions pinned: the instructions given to the agent depend on how Langflow
1.13's Python tool behaves.

### 1.3 Start Langflow

Open a **separate terminal** at the repo root and leave it running for the
whole session:

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File langflow_benchmark\start_langflow.ps1
```
Ubuntu:
```bash
bash langflow_benchmark/start_langflow.sh
```

The first start takes 2-4 minutes. It is ready when http://127.0.0.1:7860
opens in a browser. On Ubuntu without a desktop (SSH), use
`LANGFLOW_HOST=0.0.0.0 bash langflow_benchmark/start_langflow.sh` to reach it
from another machine, or just use the scripts, which don't need the browser.

From here on, run every command from inside the folder:

```bash
cd langflow_benchmark
```

---

## Part 2 - Running one test (repeat for each dataset)

Use **3 datasets**: one balanced classification, one imbalanced
classification (smallest class under 10% of rows) and one regression. One
dataset proves very little. `telco_churn` (balanced classification) is
already set up in `datasets.json`.

Plan about **2-3 hours per dataset**: 3 NeuralPilot runs plus 3 Langflow
runs. A Langflow run usually takes 20-40 minutes.

### Step 1 - Register the dataset and the request

Open [datasets.json](datasets.json) and add or update an entry:

```json
{
  "name": "telco_churn",
  "path": "../archive/WA_Fn-UseC_-Telco-Customer-Churn.csv",
  "target": "Churn",
  "problem_type": "classification",
  "request": "Predict which customers will churn. Accuracy must be at least 0.80 and the model must be explainable."
}
```

- `name`: short id, used for folder names.
- `path`: relative to this folder, always with forward slashes (works on
  both systems). Put new CSVs in `../archive/`, because `../data/` is
  git-ignored.
- `problem_type`: `classification` or `regression`.
- `request`: **write it once and copy it word for word** into NeuralPilot in
  Step 3. Include any limits ("must be explainable", "under 50 ms per 1,000
  rows", "accuracy at least 0.8"). The scripts refuse to run while it still
  says `PASTE HERE`.

### Step 2 - Check that every system scores the same test rows

Windows:
```powershell
..\.venv\Scripts\python.exe scripts\check_split.py telco_churn --neuralpilot
```
Ubuntu:
```bash
../.venv/bin/python scripts/check_split.py telco_churn --neuralpilot
```

This uses NeuralPilot's environment because it calls NeuralPilot's own loader
to compare. It must end with:

```
test rows   : 1409
first 5 IDs : [437, 2280, 2235, 4460, 3761]
MATCH: identical holdout rows
```

**Write down the test-row count and the first 5 IDs.** Every Langflow (and
Claude) result shows the same two values in its `split_check` field; if they
differ, that run is not comparable.

### Step 3 - Run NeuralPilot 3 times

1. Start NeuralPilot in another terminal (repo root): `uv run neuralpilot`
2. Open http://127.0.0.1:8000, upload the **same CSV**, paste the **exact
   request** from `datasets.json`, keep the defaults (5 iterations, 15-minute
   time limit) and start the run.
3. Wait until it finishes. Do this **3 times** (3 separate runs).

Then copy those 3 runs into `results/`:

Windows:
```powershell
.venv\Scripts\python.exe scripts\export_neuralpilot.py telco_churn --latest 3
```
Ubuntu:
```bash
.venv/bin/python scripts/export_neuralpilot.py telco_churn --latest 3
```

`--latest 3` takes the 3 newest finished runs on a file with that name. If you
made other runs on the same file in between, export each run by id instead
(the run id is shown in NeuralPilot's run history), using
`--run-id <run_id>` in place of `--latest 3`.

### Step 4 - Build the Langflow flow for this dataset

Windows:
```powershell
.venv\Scripts\python.exe scripts\build_flow.py telco_churn
```
Ubuntu:
```bash
.venv/bin/python scripts/build_flow.py telco_churn
```

This creates (or replaces) the flow **"NeuralPilot Benchmark"** in Langflow:
Chat Input → Agent (with a Python Interpreter tool) → Chat Output. It fills
the agent instructions in for this dataset, points the agent at the same
NVIDIA model as NeuralPilot, and creates a Langflow API key in
`.langflow/api_key.txt`.

Re-run this step **every time you switch datasets**: the instructions contain
the dataset's file path and target.

Optional check: open the link it prints. The Agent's *Language Model* should
show your `NVIDIA_MODEL` with no "Not available" badge.

### Step 5 - Run the Langflow agent 3 times

Windows:
```powershell
.venv\Scripts\python.exe scripts\run_flow.py telco_churn --runs 3
```
Ubuntu:
```bash
.venv/bin/python scripts/run_flow.py telco_churn --runs 3
```

Each run starts a fresh chat session, sends the request from `datasets.json`
and waits (up to 30 minutes per run). After each run it prints a line like:

```
  ok: XGBoost holdout f1_weighted=0.79  wall 1240.5 s, tool calls 22 (6 failed)  -> langflow_run1.json
```

- `ok`: the agent returned a complete JSON report.
- `incomplete`: JSON returned but required fields are missing.
- `failed`: no usable JSON (crash, timeout, or the agent gave up). Count it
  as a failed run; don't re-run to replace it.

The agent's full last message is saved in `results/<dataset>/raw/`, so you can
check what it did.

**Don't step in during these runs.** If you have to (for example, you ran it in
the Playground and had to tell the agent to continue), record it as a manual
intervention (see the note at the end of Part 2).

Tip for Ubuntu over SSH: start it inside `tmux` or with `nohup ... &` so a
dropped connection doesn't stop the runs.

### Step 6 (optional) - Claude, 3 times

Windows:
```powershell
.venv\Scripts\python.exe scripts\render_prompt.py telco_churn --claude
```
Ubuntu:
```bash
.venv/bin/python scripts/render_prompt.py telco_churn --claude
```

1. Open a **new** chat at claude.ai, attach the CSV and paste the printed
   prompt.
2. Time it from sending until the final answer.
3. Copy Claude's whole final message into a text file, e.g. `reply1.txt`.
4. Save it (example: 410 seconds, no manual help):

   Windows:
   ```powershell
   .venv\Scripts\python.exe scripts\add_result.py telco_churn Claude reply1.txt --wall-time 410 --manual 0
   ```
   Ubuntu:
   ```bash
   .venv/bin/python scripts/add_result.py telco_churn Claude reply1.txt --wall-time 410 --manual 0
   ```
5. Repeat with a new chat each time (3 runs).

### Step 7 - Build the comparison table

Windows:
```powershell
.venv\Scripts\python.exe scripts\compare.py
```
Ubuntu:
```bash
.venv/bin/python scripts/compare.py
```

This prints the table and writes `results/comparison.md`, with one section per
dataset and one column per system:

| Row | Meaning |
|---|---|
| Runs (ok / total) | runs that returned a complete report |
| Holdout primary metric | mean ± std over runs, scored once on the 20% test rows |
| CV mean ± std | 3-fold CV on training rows, averaged over runs |
| Latency | ms to predict 1,000 rows |
| Hard constraints met | passed/total per run (e.g. accuracy limit, explainability) |
| Iterations used, Wall time | effort per run |
| Spread across runs | max - min of the holdout metric: lower = more consistent |
| Failed runs / failed tool calls / manual fixes | reliability |
| Saved model / MLflow record / SHAP | what each system leaves behind |

### Note: if you use the Langflow Playground instead of `run_flow.py`

You can also open the flow at http://127.0.0.1:7860, click **Playground** and
paste the request yourself. Then:

1. Time the run yourself.
2. Count the failed tool calls (red or error outputs in the agent's steps) and
   any times you had to type something to keep it going.
3. Save the final message to `reply.txt` and record it (example: 1300 s,
   4 failed tool calls, 1 manual step):

   Windows:
   ```powershell
   .venv\Scripts\python.exe scripts\add_result.py telco_churn Langflow reply.txt --wall-time 1300 --failed-tool-calls 4 --manual 1
   ```
   Ubuntu:
   ```bash
   .venv/bin/python scripts/add_result.py telco_churn Langflow reply.txt --wall-time 1300 --failed-tool-calls 4 --manual 1
   ```

---

## Part 3 - What to send back

When all datasets are done, send:

- `results/comparison.md`
- the whole `results/` folder (per-run JSON files; the `raw/` folders are
  git-ignored, so zip them if you share by other means)
- for each dataset: the test-row count and first 5 IDs from Step 2
- your OS, and anything unusual (runs you stopped, errors, a different model)

---

## What is held the same

| Setting | Value (from NeuralPilot's `config.py`) |
|---|---|
| Rows | Same loading rules: empty rows dropped, text-number columns converted, missing-target rows dropped |
| Holdout | 20%, `train_test_split(index, test_size=0.2, random_state=42, stratify=y)` (stratify only for classification) |
| Model comparison | 3-fold CV on training rows only |
| Metric | `f1_weighted`, or `f1_macro` + balanced weights if the smallest class is < 10%; regression `r2`; a metric named in the request wins |
| Attempts / time | 5 iterations, 900 s |
| Latency | median of 3 `predict` calls on 1,000 training rows, ms per 1,000 rows |
| Model families | LinearModel, DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting, XGBoost, CatBoost |
| LLM | `NVIDIA_MODEL` from NeuralPilot's `.env` (tested with `openai/gpt-oss-20b`), same NVIDIA endpoint |

The full protocol the agent follows is in
[protocol/agent_instructions.md](protocol/agent_instructions.md).

Latency depends on the machine. Compare latency only between runs made on the
same computer.

## Langflow-specific limits (expected, not bugs)

- **The Python Interpreter is locked down** in Langflow 1.13. Code may not
  contain `import` lines or call `open()`, `getattr()` or `.format()`, and
  modules must be used by full name (`pandas.read_csv`,
  `sklearn.model_selection.train_test_split`). The instructions tell the
  agent this, but it still sometimes writes `import` lines. Those count as
  failed tool calls, which is part of what is being measured.
- Calling `.mean()` or `.std()` on a numpy array, or printing one, crashes in
  the sandbox (`KeyError('__import__')`). `numpy.mean(arr)` and
  `arr.tolist()` work. XGBoost, CatBoost and sklearn CV/grid search all work.
- Each tool call starts empty, so the agent reloads the data in every call.
- The agent may make at most 60 tool calls per run (`AGENT_MAX_ITERATIONS` in
  `scripts/build_flow.py`).
- Langflow leaves no saved model, MLflow record or SHAP output.
- A smoke test (load + split only) took about 9 minutes, mostly LLM time,
  and 5 of its 6 tool calls failed before the agent got it right. It did land
  on exactly the same test rows as NeuralPilot.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ConnectionRefused` on port 7860 | Langflow isn't running or is still starting. Wait for http://127.0.0.1:7860 to open. |
| `403 ... requires a valid API key` | Delete `.langflow/api_key.txt` and re-run `build_flow.py`. |
| Agent says the model is not found / `410 Gone` | Set `NVIDIA_MODEL` in the root `.env` to a model your key can use, then re-run `build_flow.py`. |
| `The flow is set up for 'X'` | Run `build_flow.py <dataset>` for the dataset you are testing. |
| `check_split.py` says `MISMATCH` | Stop. The datasets.json entry (path/target/problem_type) doesn't match what NeuralPilot used. |
| `export_neuralpilot.py` finds no runs | The CSV file name must match `path` in datasets.json, and the runs must have finished. |
| Ubuntu: `$'\r': command not found` | The `.sh` file got Windows line endings. Run `sed -i 's/\r$//' langflow_benchmark/*.sh`. |
| Ubuntu: `uv: command not found` right after installing | Open a new terminal, or run `source $HOME/.local/bin/env`. |
| Port 7860 already in use | Another Langflow is already running. Use it, or stop it with `Ctrl+C` in its terminal first. |
| Start over with Langflow | Stop it and delete the `langflow_benchmark/.langflow` folder. Nothing else is affected. |

## Folder layout

```
langflow_benchmark/
  README.md                  this guide
  setup.ps1 / setup.sh       one-time install into .venv (Windows / Ubuntu)
  start_langflow.ps1 / .sh   starts Langflow on http://127.0.0.1:7860
  requirements.txt           pinned versions
  datasets.json              datasets, targets and the exact request text
  protocol/agent_instructions.md   the rules the Langflow agent follows
  flows/                     exported copy of the last flow built (reference only)
  scripts/
    check_split.py           prints the test rows every system must use
    export_neuralpilot.py    copies NeuralPilot runs into results/
    build_flow.py            creates the flow in Langflow
    run_flow.py              runs the flow N times and saves the results
    render_prompt.py         fills in the protocol (--claude for a Claude chat)
    add_result.py            saves a reply you copied by hand
    compare.py               builds results/comparison.md
    common.py                shared settings and helpers
  results/<dataset>/         one JSON per run per system
```
