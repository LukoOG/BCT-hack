"""Load .env from repo root or parent workspace folder."""

from __future__ import annotations

from pathlib import Path


def load_project_env(root: Path | None = None) -> None:
    """Load env vars from BCT-hack/.env, then parent bctt/.env as fallback."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    base = root or Path(__file__).resolve().parents[2]
    load_dotenv(base.parent / ".env")
    load_dotenv(base / ".env", override=True)
