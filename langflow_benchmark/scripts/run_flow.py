"""Run the benchmark flow through Langflow's API and save the result.

    python scripts/run_flow.py telco_churn --runs 3

Each run uses a new chat session (no memory of earlier runs), sends the
dataset's request from datasets.json verbatim, times the whole run, counts the
agent's tool calls and failed tool calls, and saves
results/<dataset>/langflow_runN.json (+ the full reply under raw/).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid

from build_flow import CONFIG_FILE, LANGFLOW_URL, MODEL, api_session
from common import extract_json, get_dataset, missing_fields, save_result

RUN_TIMEOUT_S = 1800  # protocol budget is 900 s; allow LLM latency on top


def _is_failure(output) -> bool:
    """The interpreter reports blocked code as {"error": ...} but a runtime
    exception as a normal result string such as "TypeError('...')"."""
    if isinstance(output, dict):
        if output.get("error"):
            return True
        output = output.get("result", "")
    text = str(output or "").strip()
    return bool(re.match(r"^[A-Za-z_]*(Error|Exception)\(", text)) or "Traceback (most recent call last)" in text


def tool_calls(response: dict) -> tuple[int, int]:
    """(tool calls, failed tool calls) from the agent message's content blocks."""
    calls = failed = 0
    for output in response.get("outputs", []):
        for result in output.get("outputs", []):
            message = (result.get("results") or {}).get("message") or {}
            for block in message.get("content_blocks") or []:
                for item in block.get("contents") or []:
                    if item.get("type") != "tool_use":
                        continue
                    calls += 1
                    out = item.get("output")
                    artifact = out.get("artifact") if isinstance(out, dict) else None
                    if item.get("error") or _is_failure(artifact if isinstance(artifact, dict) else out):
                        failed += 1
    return calls, failed


def reply_text(response: dict) -> str:
    for output in response.get("outputs", []):
        for result in output.get("outputs", []):
            message = (result.get("results") or {}).get("message") or {}
            if message.get("text"):
                return message["text"]
    return ""


def run_once(dataset: dict, flow_id: str) -> tuple[dict, str]:
    s = api_session()
    body = {
        "input_value": dataset["request"],
        "input_type": "chat",
        "output_type": "chat",
        "session_id": f"bench-{uuid.uuid4().hex[:12]}",
    }
    started = time.perf_counter()
    try:
        r = s.post(f"{LANGFLOW_URL}/api/v1/run/{flow_id}", params={"stream": "false"}, json=body, timeout=RUN_TIMEOUT_S)
        response = r.json() if r.ok else {"error": f"HTTP {r.status_code}: {r.text[:2000]}"}
    except Exception as exc:  # timeouts and dropped connections count as a failed run
        response = {"error": f"{type(exc).__name__}: {exc}"}
    wall = round(time.perf_counter() - started, 1)

    text = reply_text(response)
    calls, failed = tool_calls(response)
    try:
        result = extract_json(text)
        missing = missing_fields(result)
        status = "incomplete" if missing else "ok"
    except ValueError:
        result, missing, status = {}, [], "failed"
    result.update({
        "system": "Langflow",
        "status": status,
        "missing_fields": missing,
        "wall_time_s_reported": result.get("wall_time_s"),
        "wall_time_s": wall,  # measured from outside, so it includes LLM time
        "tool_calls": calls,
        "failed_tool_calls": failed,
        "manual_interventions": 0,
        "llm": f"{MODEL} (NVIDIA)",
    })
    if "error" in response:
        result["error"] = response["error"]
    raw = text or json.dumps(response, indent=2)[:200000]
    return result, raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    if dataset["request"].startswith("PASTE HERE"):
        raise SystemExit("Put the exact NeuralPilot request text into datasets.json first.")
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    if config["dataset"] != args.dataset:
        raise SystemExit(f"The flow is set up for '{config['dataset']}'. Run build_flow.py {args.dataset} first.")

    for i in range(args.runs):
        print(f"run {i + 1}/{args.runs} ...", flush=True)
        result, raw = run_once(dataset, config["flow_id"])
        path = save_result(args.dataset, "Langflow", result, raw_text=raw)
        holdout = (result.get("holdout_metrics") or {}).get(result.get("primary_metric"))
        print(
            f"  {result['status']}: {result.get('model_family')} holdout {result.get('primary_metric')}={holdout}"
            f"  wall {result['wall_time_s']} s, tool calls {result['tool_calls']} ({result['failed_tool_calls']} failed)"
            f"  -> {path.name}"
        )
        if result.get("error"):
            print(f"  error: {result['error'][:300]}")


if __name__ == "__main__":
    main()
