# Start Langflow for the benchmark, isolated from NeuralPilot.
# Its database, logs and settings live in langflow_benchmark\.langflow.
#   powershell -ExecutionPolicy Bypass -File langflow_benchmark\start_langflow.ps1

$here = $PSScriptRoot
$env:LANGFLOW_CONFIG_DIR = Join-Path $here ".langflow"
$env:LANGFLOW_AUTO_LOGIN = "true"
$env:LANGFLOW_DO_NOT_TRACK = "true"
$env:PYTHONIOENCODING = "utf-8"
# CatBoost writes catboost_info\ into the working directory; keep it in work\
New-Item -ItemType Directory -Force (Join-Path $here "work") | Out-Null
Set-Location (Join-Path $here "work")

& (Join-Path $here ".venv\Scripts\langflow.exe") run --host 127.0.0.1 --port 7860 --no-open-browser
