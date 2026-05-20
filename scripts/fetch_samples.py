"""
Stream Amazon Reviews 2023 samples from HuggingFace and cache as Parquet.

Usage (from repo root):
    python scripts/fetch_samples.py
    python scripts/fetch_samples.py --size 50000 --categories Books Electronics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import config
from app.core import constants as C


HF_CONFIG = {
    "Books": "raw_review_Books",
    "Electronics": "raw_review_Electronics",
}


def _normalize(df, category: str):
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
    df[C.F_CATEGORY] = category
    df[C.F_DOMAIN] = C.DOMAIN_AMAZON
    return df


def fetch_category(category: str, sample_size: int, force: bool) -> Path:
    import pandas as pd
    from datasets import load_dataset

    out = config.DATA_RAW_DIR / f"{category.lower()}_reviews_sample.parquet"
    if out.exists() and not force:
        print(f"[{category}] cache hit -> {out}")
        return out

    config_name = HF_CONFIG[category]
    print(f"[{category}] streaming up to {sample_size:,} rows ({config_name}) ...")
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

    df = _normalize(pd.DataFrame(rows), category)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[{category}] saved {len(df):,} rows -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20_000, help="rows per category")
    parser.add_argument(
        "--categories", nargs="+", default=["Books", "Electronics"],
    )
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    paths = []
    for cat in args.categories:
        paths.append(fetch_category(cat, args.size, args.force))
    print("Done:", *[str(p) for p in paths], sep="\n  ")


if __name__ == "__main__":
    main()
