"""Fill the protocol in for one dataset, from protocol/agent_instructions.md.

    python scripts/render_prompt.py telco_churn            # Langflow agent instructions
    python scripts/render_prompt.py telco_churn --claude   # prompt to paste into a new Claude chat

The output is written to work/<dataset>_<system>_prompt.txt and printed.
"""

from __future__ import annotations

import argparse

from common import INSTRUCTIONS_FILE, WORK_DIR, get_dataset

TOOL_SECTION_START = "HOW THE TOOL BEHAVES"
TOOL_SECTION_END = "FIXED PROTOCOL"


def render(dataset: str, system: str) -> str:
    entry = get_dataset(dataset)
    text = INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    text = text.replace("{DATA_PATH}", entry["abs_path"].replace("\\", "/")).replace("{TARGET}", entry.get("target", ""))
    if system == "Claude":
        # Claude's code tool keeps state and the file is attached, so the
        # Langflow-specific notes are dropped and the request goes inline.
        head, rest = text.split(TOOL_SECTION_START, 1)
        _, tail = rest.split(TOOL_SECTION_END, 1)
        text = head + TOOL_SECTION_END + tail
        text = text.replace("You have one tool: a Python interpreter.\nUse it to", "Use code execution to")
        text = text.replace(entry["abs_path"].replace("\\", "/"), f"{entry['path'].split('/')[-1]} (attached)")
        text = text.replace("The user's message is the USER REQUEST. Treat it as verbatim requirements.",
                            f'USER REQUEST (verbatim): "{entry["request"]}"')
        text = text.replace('"system": "Langflow"', '"system": "Claude"')
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--claude", action="store_true")
    args = parser.parse_args()
    system = "Claude" if args.claude else "Langflow"
    text = render(args.dataset, system)
    WORK_DIR.mkdir(exist_ok=True)
    out = WORK_DIR / f"{args.dataset}_{system.lower()}_prompt.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n--- written to {out}")


if __name__ == "__main__":
    main()
