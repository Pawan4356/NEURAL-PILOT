#!/usr/bin/env bash
# One-time setup (Linux/macOS): creates langflow_benchmark/.venv (Python 3.12)
# with Langflow and the ML libraries. NeuralPilot's own .venv is not touched.
#   bash langflow_benchmark/setup.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv venv --python 3.12 "$here/.venv"
uv pip install --python "$here/.venv/bin/python" -r "$here/requirements.txt"
echo
echo "Done. Next: start Langflow with  bash langflow_benchmark/start_langflow.sh"
