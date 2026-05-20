"""
One-shot setup for Demilade's track: fetch samples, run EDA, smoke-test eval.

    python scripts/setup_demilade.py
    python scripts/setup_demilade.py --size 15000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    py = sys.executable
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fetch = [py, str(ROOT / "scripts" / "fetch_samples.py"), "--size", str(args.size)]
    if args.force:
        fetch.append("--force")
    run(fetch)
    run([py, str(ROOT / "scripts" / "run_eda.py")])
    run([py, str(ROOT / "scripts" / "run_eval_on_sample.py"), "--max-users", "15"])
    print("\nSetup complete. Next:")
    print("  streamlit run app/frontend/streamlit_app.py")


if __name__ == "__main__":
    main()
