"""
Build offline TF-IDF + FAISS indexes from local parquet (no network).

Usage:
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --categories Books Electronics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import config
from app.pipeline.retriever import ReviewRetriever, load_train_reviews


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local FAISS indexes from cached reviews")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=config.AMAZON_CATEGORIES,
        help="Categories to index (default: all configured)",
    )
    args = parser.parse_args()

    built = 0
    for cat in args.categories:
        train = load_train_reviews(cat)
        if train.empty:
            print(f"[{cat}] skip — no train reviews in local data")
            continue
        retriever = ReviewRetriever(cat)
        retriever.build(train)
        print(f"[{cat}] indexed {len(train):,} train reviews")
        built += 1

    if not built:
        print("Nothing indexed. Ensure data/processed/reviews.parquet or data/raw samples exist.")
        return 1

    print(f"\nDone: {built} categories -> {config.DATA_EMBEDDINGS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
