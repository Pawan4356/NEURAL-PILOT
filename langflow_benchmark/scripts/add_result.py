"""Save a final reply you copied by hand (Claude chat, or Langflow's Playground).

Paste the agent's whole last message into a text file, then:

    python scripts/add_result.py telco_churn Claude reply.txt --wall-time 410
    python scripts/add_result.py telco_churn Langflow reply.txt --failed-tool-calls 2 --manual 1

The JSON block is pulled out of the reply, checked for the required fields and
saved as results/<dataset>/<system>_runN.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import extract_json, missing_fields, save_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("system", help="Claude or Langflow")
    parser.add_argument("reply_file")
    parser.add_argument("--wall-time", type=float, help="seconds, if the reply's own value is missing or wrong")
    parser.add_argument("--failed-tool-calls", type=int, default=0)
    parser.add_argument("--manual", type=int, default=0, help="times you had to step in by hand")
    args = parser.parse_args()

    text = Path(args.reply_file).read_text(encoding="utf-8")
    try:
        result = extract_json(text)
        status = "ok"
    except ValueError:
        result, status = {}, "failed"
    missing = missing_fields(result) if result else []
    if missing:
        status = "incomplete"

    result["system"] = args.system
    result["status"] = status
    result["missing_fields"] = missing
    result["failed_tool_calls"] = args.failed_tool_calls
    result["manual_interventions"] = args.manual
    if args.wall_time is not None:
        result["wall_time_s"] = args.wall_time

    path = save_result(args.dataset, args.system, result, raw_text=text)
    print(f"saved {path} (status: {status}{', missing ' + ', '.join(missing) if missing else ''})")


if __name__ == "__main__":
    main()
