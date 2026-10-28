# One-time setup: creates langflow_benchmark\.venv (Python 3.12) with Langflow
# and the ML libraries. NeuralPilot's own .venv is not touched.
#   powershell -ExecutionPolicy Bypass -File langflow_benchmark\setup.ps1

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    Write-Host "uv is not installed. Install it first:"
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
}

& $uv venv --python 3.12 (Join-Path $here ".venv")
& $uv pip install --python (Join-Path $here ".venv\Scripts\python.exe") -r (Join-Path $here "requirements.txt")
Write-Host ""
Write-Host "Done. Next: start Langflow with  powershell -ExecutionPolicy Bypass -File langflow_benchmark\start_langflow.ps1"
