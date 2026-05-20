"""
Stream item metadata samples (titles, descriptions, price) per category.

    python scripts/fetch_meta_samples.py --size 30000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import config
from app.core import constants as C
from scripts.fetch_samples import hf_meta_config


def _normalize_meta(df, category: str):
    item_col = C.F_ITEM_ID
    if "parent_asin" in df.columns:
        df[item_col] = df["parent_asin"]
    elif "asin" in df.columns:
        df[item_col] = df["asin"]
    df[item_col] = df[item_col].astype(str).map(
        lambda i: i if str(i).startswith("amz_") else f"amz_{i}"
    )
    df[C.F_CATEGORY] = category
    df[C.F_DOMAIN] = C.DOMAIN_AMAZON
    if "title" not in df.columns:
        df["title"] = ""
    if "description" not in df.columns:
        df["description"] = ""
    return df


def fetch_meta(category: str, sample_size: int, force: bool) -> Path:
    import pandas as pd
    from datasets import load_dataset

    out = config.DATA_RAW_DIR / f"{category.lower()}_meta_sample.parquet"
    if out.exists() and not force:
        print(f"[{category}] meta cache hit -> {out.name}")
        return out

    print(f"[{category}] streaming up to {sample_size:,} meta rows ...")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        hf_meta_config(category),
        split="full",
        streaming=True,
        trust_remote_code=True,
    )
    rows = []
    for i, row in enumerate(ds):
        if i >= sample_size:
            break
        rows.append(row)

    df = _normalize_meta(pd.DataFrame(rows), category)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    mb = out.stat().st_size / 1_048_576
    print(f"[{category}] meta saved {len(df):,} items ({mb:.1f} MB)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=30_000)
    parser.add_argument("--categories", nargs="+", default=config.AMAZON_CATEGORIES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for cat in args.categories:
        fetch_meta(cat, args.size, args.force)


if __name__ == "__main__":
    main()
