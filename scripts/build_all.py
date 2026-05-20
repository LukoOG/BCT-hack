"""
Full Demilade pipeline: fetch -> process -> profiles -> EDA -> eval.

    python scripts/build_all.py
    python scripts/build_all.py --skip-fetch   # use existing parquets
    python scripts/build_all.py --size 50000 --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    py = sys.executable
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=50_000)
    parser.add_argument("--meta-size", type=int, default=25_000)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.skip_fetch:
        fetch = [py, str(ROOT / "scripts/fetch_samples.py"), "--size", str(args.size)]
        meta = [py, str(ROOT / "scripts/fetch_meta_samples.py"), "--size", str(args.meta_size)]
        if args.force:
            fetch.append("--force")
            meta.append("--force")
        run(fetch)
        run(meta)

    from app.data.from_samples import build_processed_from_samples
    from app.profile.user_profiles import build_user_profiles

    reviews, _ = build_processed_from_samples()
    profiles = build_user_profiles(reviews)
    print(f"User profiles: {len(profiles):,}")

    run([py, str(ROOT / "scripts/run_eda.py")])
    run([py, str(ROOT / "scripts/run_eval_on_sample.py"), "--max-users", "40"])
    print("\nBuild complete. Run: streamlit run app/frontend/streamlit_app.py")


if __name__ == "__main__":
    main()
