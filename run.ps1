# One command: venv, fetch all categories, build profiles, launch demo
$Root = $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv")) { python -m venv .venv }
& "$Root\.venv\Scripts\Activate.ps1"
pip install -q -r requirements-demilade.txt

if (-not (Test-Path "data\raw\books_reviews_sample.parquet")) {
    Write-Host "Building data pipeline (first run, may take a while)..." -ForegroundColor Yellow
    python scripts/build_all.py --size 50000
} elseif (-not (Test-Path "data\processed\user_profiles.parquet")) {
    python scripts/build_all.py --skip-fetch
}

streamlit run app/frontend/streamlit_app.py
