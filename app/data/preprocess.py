"""
preprocess.py — Parse raw JSONL files into the unified schema.

Pipeline per source:
  1. Parse raw JSONL records field by field
  2. Clean text (strip HTML, collapse whitespace)
  3. Normalise ratings to float in [1.0, 5.0]
  4. Apply quality filters (min text length, valid rating)
  5. Emit rows conforming to the unified schema

Unified review schema
─────────────────────
  user_id       str       source-namespaced user identifier
  item_id       str       source-namespaced item identifier
  rating        float     1.0–5.0
  review_text   str       cleaned review body
  timestamp     int       unix seconds (best-effort)
  domain        str       "amazon" | "goodreads"
  category      str       e.g. "Books", "Electronics"
  split         str       "train" | "test"  ← assigned in dataset.py

Unified item schema
───────────────────
  item_id       str
  title         str
  description   str
  avg_rating    float
  price_tier    str       "budget" | "mid" | "premium" | "luxury" | "unknown"
  category      str
  domain        str
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional

import pandas as pd

from app.core.config import MIN_REVIEW_CHARS, MIN_USER_REVIEWS, MIN_ITEM_REVIEWS
from app.core.constants import (
    DOMAIN_AMAZON, DOMAIN_GOODREADS,
    F_USER_ID, F_ITEM_ID, F_RATING, F_REVIEW_TEXT,
    F_TIMESTAMP, F_DOMAIN, F_CATEGORY,
)
from app.utils.helpers import (
    clean_text, clamp_rating, bucket_price,
    iter_jsonl_gz, save_parquet, Timer,
)
from app.utils.logger import logger


# ──────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ts_from_amazon(record: dict) -> int:
    """Amazon 2023 provides timestamp as unix milliseconds or seconds."""
    ts = record.get("timestamp")
    if ts is None:
        return 0
    ts = int(ts)
    # Milliseconds if 13 digits
    return ts // 1000 if ts > 1_000_000_000_000 else ts


_GOODREADS_DATE_FMTS = [
    "%a %b %d %H:%M:%S %z %Y",  # "Fri Jan 10 00:44:31 -0800 2014"
    "%Y-%m-%d",
]


def _ts_from_goodreads(record: dict) -> int:
    raw = record.get("date_updated") or record.get("date_added") or ""
    for fmt in _GOODREADS_DATE_FMTS:
        try:
            return int(datetime.strptime(raw.strip(), fmt).timestamp())
        except (ValueError, AttributeError):
            pass
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Amazon parsers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_amazon_review(record: dict, category: str) -> Optional[dict]:
    """
    Parse one Amazon review record into the unified review schema.
    Returns None if the record fails quality checks.
    """
    text = clean_text(record.get("text") or record.get("reviewText") or "")
    if len(text) < MIN_REVIEW_CHARS:
        return None

    rating = clamp_rating(record.get("rating") or record.get("overall"))
    if rating is None:
        return None

    user_id = record.get("user_id") or record.get("reviewerID") or ""
    item_id = record.get("asin") or record.get("parent_asin") or ""
    if not user_id or not item_id:
        return None

    return {
        F_USER_ID:     f"amz_{user_id}",
        F_ITEM_ID:     f"amz_{item_id}",
        F_RATING:      rating,
        F_REVIEW_TEXT: text,
        F_TIMESTAMP:   _ts_from_amazon(record),
        F_DOMAIN:      DOMAIN_AMAZON,
        F_CATEGORY:    category,
    }


def _parse_amazon_meta(record: dict, category: str) -> Optional[dict]:
    """Parse one Amazon metadata record into the unified item schema."""
    item_id = record.get("asin") or record.get("parent_asin") or ""
    if not item_id:
        return None

    # description can be a list of strings (or numpy array from parquet)
    raw_desc = record.get("description")
    if raw_desc is None:
        raw_desc = ""
    elif isinstance(raw_desc, str):
        pass
    elif hasattr(raw_desc, "__iter__"):
        raw_desc = " ".join(str(x) for x in raw_desc)
    else:
        raw_desc = str(raw_desc)

    return {
        F_ITEM_ID:            f"amz_{item_id}",
        "title":              clean_text(record.get("title") or ""),
        "description":        clean_text(raw_desc),
        "avg_rating":         clamp_rating(record.get("average_rating")) or 0.0,
        "price_tier":         bucket_price(record.get("price")),
        F_CATEGORY:           category,
        F_DOMAIN:             DOMAIN_AMAZON,
    }


def parse_amazon_category(
    review_path: Path,
    meta_path: Path,
    category: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a full Amazon category (reviews + metadata).

    Returns:
        (reviews_df, items_df)
    """
    logger.info(f"[Amazon:{category}] Parsing reviews ...")
    reviews: List[dict] = []
    with Timer(f"[Amazon:{category}] review parse"):
        for rec in iter_jsonl_gz(review_path, desc=f"Amazon {category} reviews"):
            parsed = _parse_amazon_review(rec, category)
            if parsed:
                reviews.append(parsed)

    logger.info(f"[Amazon:{category}] Parsed {len(reviews):,} valid reviews")

    logger.info(f"[Amazon:{category}] Parsing metadata ...")
    items: List[dict] = []
    with Timer(f"[Amazon:{category}] meta parse"):
        for rec in iter_jsonl_gz(meta_path, desc=f"Amazon {category} meta"):
            parsed = _parse_amazon_meta(rec, category)
            if parsed:
                items.append(parsed)

    logger.info(f"[Amazon:{category}] Parsed {len(items):,} item records")

    return pd.DataFrame(reviews), pd.DataFrame(items)


# ──────────────────────────────────────────────────────────────────────────────
# Goodreads parsers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_goodreads_review(record: dict) -> Optional[dict]:
    text = clean_text(record.get("review_text") or "")
    if len(text) < MIN_REVIEW_CHARS:
        return None

    rating = clamp_rating(record.get("rating"))
    if rating is None or rating == 0.0:   # Goodreads uses 0 for "no rating"
        return None

    user_id = record.get("user_id") or ""
    item_id = record.get("book_id") or ""
    if not user_id or not item_id:
        return None

    return {
        F_USER_ID:     f"gr_{user_id}",
        F_ITEM_ID:     f"gr_{item_id}",
        F_RATING:      rating,
        F_REVIEW_TEXT: text,
        F_TIMESTAMP:   _ts_from_goodreads(record),
        F_DOMAIN:      DOMAIN_GOODREADS,
        F_CATEGORY:    "Books",         # Goodreads is books-only
    }


def _parse_goodreads_book(record: dict) -> Optional[dict]:
    item_id = record.get("book_id") or ""
    if not item_id:
        return None

    # Derive a price tier from num_pages as a rough proxy (no price in Goodreads)
    pages = record.get("num_pages")
    price_tier = "unknown"
    try:
        p = int(pages)
        price_tier = "budget" if p < 200 else "mid" if p < 400 else "premium"
    except (TypeError, ValueError):
        pass

    # Top shelves = genre signals
    shelves = record.get("popular_shelves") or []
    category = shelves[0]["name"] if shelves else "Books"

    return {
        F_ITEM_ID:     f"gr_{item_id}",
        "title":       clean_text(record.get("title") or ""),
        "description": clean_text(record.get("description") or ""),
        "avg_rating":  clamp_rating(record.get("average_rating")) or 0.0,
        "price_tier":  price_tier,
        F_CATEGORY:    category,
        F_DOMAIN:      DOMAIN_GOODREADS,
    }


def parse_goodreads(
    review_path: Path,
    book_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse Goodreads reviews + books.

    Returns:
        (reviews_df, books_df)
    """
    logger.info("[Goodreads] Parsing reviews ...")
    reviews: List[dict] = []
    with Timer("[Goodreads] review parse"):
        for rec in iter_jsonl_gz(review_path, desc="Goodreads reviews"):
            parsed = _parse_goodreads_review(rec)
            if parsed:
                reviews.append(parsed)

    logger.info(f"[Goodreads] Parsed {len(reviews):,} valid reviews")

    logger.info("[Goodreads] Parsing books ...")
    books: List[dict] = []
    with Timer("[Goodreads] book parse"):
        for rec in iter_jsonl_gz(book_path, desc="Goodreads books"):
            parsed = _parse_goodreads_book(rec)
            if parsed:
                books.append(parsed)

    logger.info(f"[Goodreads] Parsed {len(books):,} book records")

    return pd.DataFrame(reviews), pd.DataFrame(books)


# ──────────────────────────────────────────────────────────────────────────────
# Quality filtering  (applied after merging all sources)
# ──────────────────────────────────────────────────────────────────────────────

def apply_quality_filters(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Remove low-signal users and items.

    Filters applied (thresholds in config.py):
      - MIN_USER_REVIEWS : drop users with too few reviews
      - MIN_ITEM_REVIEWS : drop items with too few reviews
    """
    before = len(reviews)
    logger.info(f"Quality filter: starting with {before:,} reviews")

    # Filter sparse users
    user_counts = reviews[F_USER_ID].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_REVIEWS].index
    reviews = reviews[reviews[F_USER_ID].isin(valid_users)]
    logger.info(
        f"  After min_user_reviews={MIN_USER_REVIEWS}: "
        f"{len(reviews):,} reviews ({len(valid_users):,} users retained)"
    )

    # Filter sparse items
    item_counts = reviews[F_ITEM_ID].value_counts()
    valid_items = item_counts[item_counts >= MIN_ITEM_REVIEWS].index
    reviews = reviews[reviews[F_ITEM_ID].isin(valid_items)]
    logger.info(
        f"  After min_item_reviews={MIN_ITEM_REVIEWS}: "
        f"{len(reviews):,} reviews ({len(valid_items):,} items retained)"
    )

    logger.info(f"Quality filter: removed {before - len(reviews):,} reviews ({(1 - len(reviews)/before)*100:.1f}%)")
    return reviews.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────────────────────────────────────

def add_train_test_split(reviews: pd.DataFrame, holdout_last_n: int = 1) -> pd.DataFrame:
    """
    Assign split column: the last *holdout_last_n* interactions per user → "test",
    everything else → "train".

    Requires a timestamp column; ties broken by original row order.
    """
    logger.info(f"Splitting: holdout_last_n={holdout_last_n}")
    reviews = reviews.sort_values([F_USER_ID, F_TIMESTAMP], ascending=[True, True])
    reviews["split"] = "train"

    # Mark the last N rows per user as test
    test_mask = reviews.groupby(F_USER_ID).cumcount(ascending=False) < holdout_last_n
    reviews.loc[test_mask, "split"] = "test"

    n_test  = (reviews["split"] == "test").sum()
    n_train = (reviews["split"] == "train").sum()
    logger.info(f"  Train: {n_train:,}  |  Test: {n_test:,}")

    return reviews.reset_index(drop=True)