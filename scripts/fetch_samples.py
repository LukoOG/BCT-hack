"""
Stream Amazon Reviews 2023 samples from HuggingFace and cache as Parquet.

Usage (from repo root):
    python scripts/fetch_samples.py
    python scripts/fetch_samples.py --size 50000 --force
    python scripts/fetch_samples.py --categories Books Electronics --size 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import config
from app.core import constants as C


def hf_review_config(category: str) -> str:
    return f"raw_review_{category}"


def hf_meta_config(category: str) -> str:
    return f"raw_meta_{category}"


def _normalize_reviews(df, category: str):
    import pandas as pd

    rename = {
        "text": C.F_REVIEW_TEXT,
        "rating": C.F_RATING,
        "user_id": C.F_USER_ID,
        "parent_asin": C.F_ITEM_ID,
        "timestamp": C.F_TIMESTAMP,
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    if "asin" in df.columns and C.F_ITEM_ID not in df.columns:
        df[C.F_ITEM_ID] = df["asin"]
    # Namespace IDs to match preprocess.py
    df[C.F_USER_ID] = df[C.F_USER_ID].astype(str).map(
        lambda u: u if str(u).startswith("amz_") else f"amz_{u}"
    )
    df[C.F_ITEM_ID] = df[C.F_ITEM_ID].astype(str).map(
        lambda i: i if str(i).startswith("amz_") else f"amz_{i}"
    )
    df[C.F_CATEGORY] = category
    df[C.F_DOMAIN] = C.DOMAIN_AMAZON
    return df


def fetch_reviews(category: str, sample_size: int, force: bool) -> Path:
    import pandas as pd
    from datasets import load_dataset

    out = config.DATA_RAW_DIR / f"{category.lower()}_reviews_sample.parquet"
    if out.exists() and not force:
        mb = out.stat().st_size / 1_048_576
        print(f"[{category}] cache hit ({mb:.1f} MB) -> {out.name}")
        return out

    config_name = hf_review_config(category)
    print(f"[{category}] streaming up to {sample_size:,} reviews ({config_name}) ...")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        config_name,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    rows = []
    for i, row in enumerate(ds):
        if i >= sample_size:
            break
        rows.append(row)
        if (i + 1) % 10_000 == 0:
            print(f"  ... {i + 1:,} rows")

    df = _normalize_reviews(pd.DataFrame(rows), category)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    mb = out.stat().st_size / 1_048_576
    print(f"[{category}] saved {len(df):,} reviews ({mb:.1f} MB)")

    counts = df[C.F_USER_ID].value_counts()
    demo_users = counts[counts >= config.HOLDOUT_LAST_N + 1].head(200).index.tolist()
    users_path = config.DATA_RAW_DIR / f"{category.lower()}_demo_users.json"
    users_path.write_text(json.dumps(demo_users), encoding="utf-8")
    print(f"[{category}] {len(demo_users)} demo users, {df[C.F_ITEM_ID].nunique():,} items")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", type=int, default=config.AMAZON_SAMPLE_SIZE_PER_CATEGORY,
    )
    parser.add_argument(
        "--categories", nargs="+", default=config.AMAZON_CATEGORIES,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    total_mb = 0.0
    paths = []
    for cat in args.categories:
        p = fetch_reviews(cat, args.size, args.force)
        paths.append(p)
        total_mb += p.stat().st_size / 1_048_576

    print(f"\nDone: {len(paths)} categories, ~{total_mb:.1f} MB total on disk")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
