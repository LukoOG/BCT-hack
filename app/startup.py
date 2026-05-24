#!/usr/bin/env python3
"""
startup.py — Container entrypoint for Railway / cloud deployments.

Checks whether FAISS indexes exist. If not, rebuilds them from
the committed parquet files before starting the main service.

Usage (set as your Railway start command):
    API:       python startup.py api
    Streamlit: python startup.py demo
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Paths ──────────────────────────────────────────────────────────────────────
# Check both data/ (local) and deployment_data/ (committed snapshot)
def _find_data_root():
    for candidate in [ROOT / "data", ROOT / "deployment_data"]:
        if (candidate / "processed" / "reviews.parquet").exists():
            return candidate
    return None

def _faiss_exists(data_root: Path) -> bool:
    embeddings_dir = data_root / "embeddings"
    if not embeddings_dir.exists():
        return False
    faiss_files = list(embeddings_dir.glob("*.faiss"))
    return len(faiss_files) > 0

def _run(cmd: list[str]) -> None:
    print(f"[startup] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print(f"[startup] Command failed with code {result.returncode}", flush=True)
        sys.exit(result.returncode)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("api", "demo"):
        print("Usage: python startup.py api|demo")
        sys.exit(1)

    mode = sys.argv[1]

    print("[startup] Checking data ...", flush=True)
    data_root = _find_data_root()

    if data_root is None:
        print("[startup] ERROR: No processed data found.", flush=True)
        print("[startup]   Expected deployment_data/processed/reviews.parquet", flush=True)
        print("[startup]   Run locally: python scripts/package_deployment_data.py", flush=True)
        sys.exit(1)

    print(f"[startup] Found data at: {data_root}", flush=True)

    # If data/ doesn't exist but deployment_data/ does, symlink so the
    # app code that hardcodes 'data/' still works without modification.
    local_data = ROOT / "data"
    if not local_data.exists() and data_root.name == "deployment_data":
        print(f"[startup] Symlinking deployment_data/ → data/", flush=True)
        local_data.symlink_to(data_root)

    # Rebuild FAISS indexes if missing
    if not _faiss_exists(data_root):
        print("[startup] FAISS indexes not found — rebuilding from parquet ...", flush=True)
        _run([sys.executable, "scripts/build_embeddings.py", "--force"])
        print("[startup] FAISS rebuild complete.", flush=True)
    else:
        print("[startup] FAISS indexes present — skipping rebuild.", flush=True)

    # ── Start the service ──────────────────────────────────────────────────────
    print(f"[startup] Starting service: {mode}", flush=True)

    if mode == "api":
        _run([
            sys.executable, "-m", "uvicorn", "app.api.server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "1",
        ])
    elif mode == "demo":
        _run([
            sys.executable, "-m", "streamlit", "run",
            "app/frontend/streamlit_app.py",
            "--server.port", "8080",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
        ])


if __name__ == "__main__":
    main()
