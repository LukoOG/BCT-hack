"""
main.py — CLI entry point for the data pipeline.

Usage:
    python main.py download                         # download all datasets
    python main.py download --categories Books Electronics
    python main.py process                          # preprocess + store
    python main.py process --skip-download          # if already downloaded
    python main.py stats                            # print DB stats
    python main.py run                              # download + process in one go
"""

import argparse
import sys
from pathlib import Path

from app.core.config import AMAZON_CATEGORIES, HOLDOUT_LAST_N
from app.utils.logger import logger


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────────────────────

def stage_download(categories: list[str]) -> None:
    from app.data.loader import download_amazon, download_goodreads

    logger.info(f"=== STAGE: Download (categories: {categories}) ===")
    download_amazon(categories)
    download_goodreads()
    logger.success("=== Download complete ===")


def stage_process(categories: list[str]) -> None:
    import pandas as pd
    from app.data.loader import find_amazon_files, find_goodreads_files
    from app.data.preprocess import (
        parse_amazon_category,
        parse_goodreads,
        apply_quality_filters,
        add_train_test_split,
    )
    from app.data.dataset import build_dataset

    logger.info("=== STAGE: Process ===")

    all_reviews: list[pd.DataFrame] = []
    all_items:   list[pd.DataFrame] = []

    # ── Amazon ────────────────────────────────────────────────────────────────
    try:
        amazon_paths = find_amazon_files(categories)
        for cat, paths in amazon_paths.items():
            reviews_df, items_df = parse_amazon_category(
                paths["reviews"], paths["meta"], category=cat
            )
            all_reviews.append(reviews_df)
            all_items.append(items_df)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Goodreads ─────────────────────────────────────────────────────────────
    try:
        gr_paths = find_goodreads_files()
        gr_reviews, gr_books = parse_goodreads(gr_paths["reviews"], gr_paths["books"])
        all_reviews.append(gr_reviews)
        all_items.append(gr_books)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Merge ─────────────────────────────────────────────────────────────────
    logger.info("Merging sources ...")
    reviews = pd.concat(all_reviews, ignore_index=True)
    items   = pd.concat(all_items,   ignore_index=True)
    logger.info(f"Combined: {len(reviews):,} reviews | {len(items):,} items")

    # ── Filter + split ────────────────────────────────────────────────────────
    reviews = apply_quality_filters(reviews)
    reviews = add_train_test_split(reviews, holdout_last_n=HOLDOUT_LAST_N)

    # ── Persist to Parquet + DuckDB ───────────────────────────────────────────
    build_dataset(reviews, items)

    logger.success("=== Processing complete ===")


def stage_stats() -> None:
    from app.data.dataset import get_connection, get_dataset_stats

    logger.info("=== STAGE: Stats ===")
    con   = get_connection()
    stats = get_dataset_stats(con)

    print("\n── Dataset summary ──────────────────────────")
    for k, v in stats.items():
        if k == "domains":
            print(f"  domains:")
            for d in v:
                print(f"    {d['domain']}: {d['n']:,} reviews")
        else:
            print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    print("─────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LLM Agent Hackathon — data pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download raw datasets")
    dl.add_argument(
        "--categories", nargs="+", default=AMAZON_CATEGORIES,
        metavar="CAT",
        help=f"Amazon categories to download (default: {AMAZON_CATEGORIES})",
    )

    # process
    pr = sub.add_parser("process", help="Preprocess raw files and load into DuckDB")
    pr.add_argument(
        "--categories", nargs="+", default=AMAZON_CATEGORIES, metavar="CAT",
    )
    pr.add_argument(
        "--skip-download", action="store_true",
        help="Skip download; expect raw files to already exist",
    )

    # stats
    sub.add_parser("stats", help="Print dataset stats from processed Parquet files")

    # run (download + process in one shot)
    run = sub.add_parser("run", help="Download then process (full pipeline)")
    run.add_argument(
        "--categories", nargs="+", default=AMAZON_CATEGORIES, metavar="CAT",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "download":
        stage_download(args.categories)

    elif args.command == "process":
        if not args.skip_download:
            stage_download(args.categories)
        stage_process(args.categories)

    elif args.command == "stats":
        stage_stats()

    elif args.command == "run":
        stage_download(args.categories)
        stage_process(args.categories)


if __name__ == "__main__":
    main()
