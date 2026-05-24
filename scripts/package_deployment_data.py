"""
scripts/package_deployment_data.py

Copies the minimum processed files needed to run the demo into
deployment_data/ so they can be committed to the repo and served
in cloud deployments where data/ is not volume-mounted.

Run once after build_all.py:
    python scripts/package_deployment_data.py

Then commit deployment_data/ to git:
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


FILES = {
    # Processed parquet — core data
    "processed/reviews.parquet":       True,   # required
    "processed/items.parquet":         True,   # required
    "processed/user_profiles.parquet": True,   # required
    # FAISS indexes — all .faiss and .pkl files
    "embeddings":                      True,   # copy whole folder
}


def main():
    if not (SRC / "processed" / "reviews.parquet").exists():
        print("ERROR: data/processed/reviews.parquet not found.")
        print("Run: python scripts/build_all.py --skip-fetch")
        sys.exit(1)

    print(f"Packaging deployment data from {SRC} → {DEST}")

    # Processed parquet files
    for rel, required in FILES.items():
        src_path  = SRC / rel
        dest_path = DEST / rel

        if src_path.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(src_path, dest_path)
            count = len(list(dest_path.rglob("*")))
            print(f"  Copied dir  {rel}/ ({count} files)")

        elif src_path.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            size_mb = dest_path.stat().st_size / 1_048_576
            print(f"  Copied file {rel} ({size_mb:.2f} MB)")

        elif required:
            print(f"  MISSING (required): {rel}")
            sys.exit(1)

    # Print total size
    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"\nTotal deployment_data size: {total / 1_048_576:.1f} MB")
    print("\nNext steps:")
    print("  1. Add deployment_data/ to git (remove it from .gitignore if present)")
    print("  2. git add deployment_data/ && git commit -m 'add deployment snapshot'")
    print("  3. git push")
    print("  4. Deploy to Railway — it will find the data at deployment_data/")


if __name__ == "__main__":
    main()
