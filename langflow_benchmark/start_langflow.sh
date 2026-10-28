#!/usr/bin/env bash
# Start Langflow for the benchmark (Linux/macOS), isolated from NeuralPilot.
# Its database, logs and settings live in langflow_benchmark/.langflow.
#   bash langflow_benchmark/start_langflow.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LANGFLOW_CONFIG_DIR="$here/.langflow"
export LANGFLOW_AUTO_LOGIN=true
export LANGFLOW_DO_NOT_TRACK=true
export PYTHONIOENCODING=utf-8
# CatBoost writes catboost_info/ into the working directory; keep it in work/
mkdir -p "$here/work"
cd "$here/work"

exec "$here/.venv/bin/langflow" run --host "${LANGFLOW_HOST:-127.0.0.1}" --port 7860 --no-open-browser
