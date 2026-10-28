"""Create the "NeuralPilot Benchmark" flow in a running Langflow.

    python scripts/build_flow.py telco_churn

What it builds (Chat Input -> Agent -> Chat Output, with a Python Interpreter
connected to the Agent as its only tool):
  - Agent model: NVIDIA_MODEL from NeuralPilot's .env (the model NeuralPilot
    itself runs with) through NVIDIA's OpenAI-compatible endpoint.
  - Agent instructions: protocol/agent_instructions.md filled in for the dataset.
  - Python Interpreter Global Imports: pandas, numpy, scikit-learn, xgboost,
    catboost, time, json (Langflow does not allow `import` lines in the code).

The NVIDIA key is read from NVIDIA_API_KEY (environment, or NeuralPilot's .env)
and stored as a Langflow credential variable, not in the flow.
The flow is built from the "Simple Agent" starter that ships with this
Langflow version, so its format always matches the installed version.
Re-running replaces the flow of the same name.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import secrets
from pathlib import Path

import requests

from common import BENCH_DIR, PROJECT_ROOT, get_dataset
from render_prompt import render

LANGFLOW_URL = os.environ.get("LANGFLOW_URL", "http://127.0.0.1:7860")
FLOW_NAME = "NeuralPilot Benchmark"


def neuralpilot_env(name: str, default: str | None = None) -> str | None:
    """A setting from the environment, else NeuralPilot's .env (read only)."""
    if os.environ.get(name):
        return os.environ[name]
    env = PROJECT_ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines() if env.exists() else []:
        key, _, value = line.strip().partition("=")
        if key.strip() == name and value.strip():
            return value.strip().strip("'\"")
    return default


# Same model and endpoint NeuralPilot is configured with
MODEL = neuralpilot_env("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
BASE_URL = neuralpilot_env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
GLOBAL_IMPORTS = ",".join([
    "pandas", "numpy", "time", "json", "math",
    "sklearn", "sklearn.model_selection", "sklearn.metrics", "sklearn.pipeline",
    "sklearn.compose", "sklearn.preprocessing", "sklearn.impute",
    "sklearn.linear_model", "sklearn.tree", "sklearn.ensemble",
    "xgboost", "catboost",
])
AGENT_MAX_ITERATIONS = 60  # tool calls per run; 5 ML iterations need many calls
# Found through the installed package, so it works on Windows and Linux venvs
STARTER = (
    Path(importlib.util.find_spec("langflow").origin).parent
    / "initial_setup" / "starter_projects" / "Simple Agent.json"
)
CONFIG_FILE = BENCH_DIR / ".langflow" / "benchmark.json"


def session() -> requests.Session:
    s = requests.Session()
    token = s.get(f"{LANGFLOW_URL}/api/v1/auto_login", timeout=30).json()["access_token"]
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def api_session() -> requests.Session:
    """Session for /api/v1/run, which needs a Langflow API key (since v1.5).
    The key is created once and kept in .langflow/ (git-ignored)."""
    key_file = BENCH_DIR / ".langflow" / "api_key.txt"
    if not key_file.exists():
        r = session().post(f"{LANGFLOW_URL}/api/v1/api_key/", json={"name": "neuralpilot-benchmark"}, timeout=30)
        r.raise_for_status()
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(r.json()["api_key"], encoding="utf-8")
    s = requests.Session()
    s.headers["x-api-key"] = key_file.read_text(encoding="utf-8").strip()
    return s


def nvidia_key() -> str:
    key = neuralpilot_env("NVIDIA_API_KEY")
    if not key:
        raise SystemExit("Set NVIDIA_API_KEY (or put it in NeuralPilot's .env).")
    return key


def set_variable(s: requests.Session, name: str, value: str, kind: str) -> None:
    existing = {v["name"]: v for v in s.get(f"{LANGFLOW_URL}/api/v1/variables/", timeout=30).json()}
    body = {"name": name, "value": value, "type": kind, "default_fields": []}
    if name in existing:
        r = s.patch(f"{LANGFLOW_URL}/api/v1/variables/{existing[name]['id']}", json={**body, "id": existing[name]["id"]}, timeout=30)
    else:
        r = s.post(f"{LANGFLOW_URL}/api/v1/variables/", json=body, timeout=30)
    r.raise_for_status()


def edge(source: dict, output: str, output_types: list[str], target: dict, field: str, input_types: list[str]) -> dict:
    """An edge in the same shape the Langflow editor saves."""
    src = {"dataType": source["data"]["type"], "id": source["id"], "name": output, "output_types": output_types}
    tgt = {"fieldName": field, "id": target["id"], "inputTypes": input_types, "type": "other"}
    enc = lambda d: json.dumps(d).replace('"', "œ")  # noqa: E731 - Langflow's handle encoding
    return {
        "animated": False, "className": "", "selected": False,
        "data": {"sourceHandle": src, "targetHandle": tgt},
        "id": f"reactflow__edge-{source['id']}{enc(src)}-{target['id']}{enc(tgt)}",
        "source": source["id"], "sourceHandle": enc(src),
        "target": target["id"], "targetHandle": enc(tgt),
    }


def build(dataset: str, s: requests.Session) -> dict:
    flow = json.loads(STARTER.read_text(encoding="utf-8"))
    nodes = {n["data"]["type"]: n for n in flow["data"]["nodes"]}
    chat_in, agent, chat_out = nodes["ChatInput"], nodes["Agent"], nodes["ChatOutput"]
    tool_example = nodes["URLComponent"]

    # Python Interpreter node from this server's own component catalogue
    catalogue = s.get(f"{LANGFLOW_URL}/api/v1/all", timeout=120).json()
    repl_template = next(
        comp for group in catalogue.values() for name, comp in group.items() if name == "PythonREPLComponent"
    )
    repl = copy.deepcopy(tool_example)
    repl["id"] = "PythonREPLComponent-" + secrets.token_hex(3)
    repl["data"]["id"] = repl["id"]
    repl["data"]["type"] = "PythonREPLComponent"
    repl["data"]["node"] = copy.deepcopy(repl_template)
    repl["data"]["node"]["tool_mode"] = True
    repl["data"]["node"]["outputs"] = copy.deepcopy(tool_example["data"]["node"]["outputs"])
    repl["data"]["node"]["template"]["global_imports"]["value"] = GLOBAL_IMPORTS
    repl["data"]["node"]["template"].pop("tools_metadata", None)
    repl["position"] = {"x": chat_in["position"]["x"] - 50, "y": chat_in["position"]["y"] + 300}

    providers = s.get(f"{LANGFLOW_URL}/api/v1/models", timeout=60).json()
    listed = next(
        (m for p in providers if p.get("provider") == "OpenAI Compatible"
         for m in p.get("models", []) if m["model_name"] == MODEL),
        None,
    )
    if listed is None:
        print(f"warning: '{MODEL}' is not in the endpoint's model list; the editor may show it as unavailable")
    template = agent["data"]["node"]["template"]
    template["model"]["value"] = [{
        "provider": "OpenAI Compatible", "name": MODEL, "icon": "Plug",
        "metadata": (listed or {}).get("metadata", {}),
    }]
    template["system_prompt"]["value"] = render(dataset, "Langflow")
    template["max_iterations"]["value"] = AGENT_MAX_ITERATIONS

    flow["data"]["nodes"] = [chat_in, agent, chat_out, repl]
    flow["data"]["edges"] = [
        e for e in flow["data"]["edges"] if e["source"] in (chat_in["id"], agent["id"])
    ] + [edge(repl, "component_as_tool", ["Tool"], agent, "tools", ["Tool"])]
    return {
        "name": FLOW_NAME,
        "description": f"AutoML protocol run for the NeuralPilot comparison ({dataset})",
        "data": flow["data"],
        "endpoint_name": "neuralpilot-benchmark",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    get_dataset(args.dataset)

    s = session()
    set_variable(s, "OPENAI_COMPATIBLE_BASE_URL", BASE_URL, "Generic")
    set_variable(s, "OPENAI_COMPATIBLE_API_KEY", nvidia_key(), "Credential")

    body = build(args.dataset, s)
    for old in s.get(f"{LANGFLOW_URL}/api/v1/flows/", params={"get_all": "true"}, timeout=60).json():
        if isinstance(old, dict) and old.get("name") == FLOW_NAME:
            s.delete(f"{LANGFLOW_URL}/api/v1/flows/{old['id']}", timeout=30)
    r = s.post(f"{LANGFLOW_URL}/api/v1/flows/", json=body, timeout=60)
    if not r.ok:
        raise SystemExit(f"Langflow refused the flow: {r.status_code} {r.text[:500]}")
    flow_id = r.json()["id"]

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps({"flow_id": flow_id, "dataset": args.dataset}, indent=2), encoding="utf-8")
    export = BENCH_DIR / "flows" / "neuralpilot_benchmark.json"
    export.parent.mkdir(exist_ok=True)
    exported = {**body, "data": copy.deepcopy(body["data"])}
    export.write_text(json.dumps(exported, indent=2), encoding="utf-8")
    print(f"flow '{FLOW_NAME}' created for {args.dataset}: {LANGFLOW_URL}/flow/{flow_id}")
    print(f"exported copy (no API key inside): {export.relative_to(BENCH_DIR)}")


if __name__ == "__main__":
    main()
