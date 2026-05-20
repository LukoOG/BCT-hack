# Run from anywhere:  powershell -File C:\Users\User\Desktop\nothing\bct\BCT-hack\run.ps1
# Or double-click after: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$Root = $PSScriptRoot
Set-Location $Root
Write-Host "Project: $Root" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating venv..."
    python -m venv .venv
}

& "$Root\.venv\Scripts\Activate.ps1"

if (-not (Test-Path "data\raw\books_reviews_sample.parquet")) {
    Write-Host "Fetching sample data (first time)..."
    python scripts/setup_demilade.py --size 10000
}

Write-Host "Starting Streamlit..." -ForegroundColor Green
streamlit run app/frontend/streamlit_app.py
