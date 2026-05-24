"""
scripts/package_deployment_data.py

Packages ONLY the small processed parquet files into deployment_data/.
FAISS indexes are intentionally excluded — they exceed GitHub's 100MB
limit and are rebuilt automatically at container startup via startup.py.

Run once after build_all.py:
    python scripts/package_deployment_data.py

Then commit:
    git add deployment_data/
    git commit -m "add deployment data snapshot"
    git push
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "data"
DEST = ROOT / "deployment_data"

# Only parquet files — FAISS (.faiss, .pkl) are rebuilt at startup
REQUIRED_FILES = [
    "processed/reviews.parquet",
    "processed/items.parquet",
    "processed/user_profiles.parquet",
]

OPTIONAL_FILES = [
    "processed/reviews.duckdb",
]


def main():
    missing = [f for f in REQUIRED_FILES if not (SRC / f).exists()]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  data/{f}")
        print("\nRun: python scripts/build_all.py --skip-fetch")
        sys.exit(1)

    print(f"Packaging parquet files: {SRC} -> {DEST}\n")

    for rel in REQUIRED_FILES:
        src_path  = SRC / rel
        dest_path = DEST / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        size_mb = dest_path.stat().st_size / 1_048_576
        print(f"  OK  {rel:45s} ({size_mb:.2f} MB)")

    for rel in OPTIONAL_FILES:
        src_path = SRC / rel
        if src_path.exists():
            size_mb = src_path.stat().st_size / 1_048_576
            if size_mb < 50:
                dest_path = DEST / rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                print(f"  OK  {rel:45s} ({size_mb:.2f} MB)  [optional]")
            else:
                print(f"  --  {rel:45s} skipped (> 50 MB)")

    # Write a .gitignore inside deployment_data/ that explicitly
    # un-excludes parquet (root .gitignore has *.parquet globally excluded)
    gitignore = DEST / ".gitignore"
    gitignore.write_text(
        "# Allow parquet — small files here (not full corpus)\n"
        "!*.parquet\n"
        "!processed/*.parquet\n"
        "# Never commit FAISS indexes — rebuilt at startup\n"
        "*.faiss\n"
        "*.pkl\n"
    )
    print(f"  OK  deployment_data/.gitignore written")

    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"\nTotal size: {total / 1_048_576:.1f} MB  (well under GitHub 100MB limit)")
    print("\nNext steps:")
    print("  1. git add deployment_data/")
    print("  2. git commit -m 'add deployment data snapshot'")
    print("  3. git push")
    print("\n  FAISS indexes rebuild inside container on first boot (~60s).")
    print("  Subsequent boots reuse cached indexes.")


if __name__ == "__main__":
    main()
