"""
Build unified processed tables from cached HF Parquet samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from app.core import config
from app.core import constants as C
from app.data.preprocess import add_train_test_split, apply_quality_filters
from app.utils.helpers import clean_text, clamp_rating, save_parquet
from app.utils.logger import logger


def list_review_samples() -> List[Path]:
    return sorted(config.DATA_RAW_DIR.glob("*_reviews_sample.parquet"))


def list_meta_samples() -> List[Path]:
    return sorted(config.DATA_RAW_DIR.glob("*_meta_sample.parquet"))


def _category_from_stem(stem: str) -> str:
    key = stem.replace("_reviews_sample", "").replace("_meta_sample", "")
    name_map = {c.lower().replace(" ", "_"): c for c in config.AMAZON_CATEGORIES}
    return name_map.get(key, key)


def load_all_review_samples(categories: Optional[List[str]] = None) -> pd.DataFrame:
    frames = []
    for path in list_review_samples():
        cat = _category_from_stem(path.stem)
        if categories and cat not in categories:
            continue
        df = pd.read_parquet(path)
        if C.F_CATEGORY not in df.columns:
            df[C.F_CATEGORY] = cat
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No review samples. Run: python scripts/fetch_samples.py")
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} reviews from {len(frames)} category files")
    return combined


def load_all_meta_samples() -> pd.DataFrame:
    from app.data.preprocess import _parse_amazon_meta

    items: list[dict] = []
    for path in list_meta_samples():
        cat = _category_from_stem(path.stem)
        df = pd.read_parquet(path)
        for rec in df.to_dict(orient="records"):
            parsed = _parse_amazon_meta(rec, cat)
            if parsed:
                items.append(parsed)
    if not items:
        logger.warning("No meta samples found")
        return pd.DataFrame()
    return pd.DataFrame(items).drop_duplicates(subset=[C.F_ITEM_ID], keep="first")


def prepare_reviews_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean sample reviews to unified schema."""
    df = raw.copy()
    text_col = C.F_REVIEW_TEXT if C.F_REVIEW_TEXT in df.columns else "text"
    df[C.F_REVIEW_TEXT] = df[text_col].fillna("").astype(str).map(clean_text)
    df = df[df[C.F_REVIEW_TEXT].str.len() >= config.MIN_REVIEW_CHARS]
    df[C.F_RATING] = df[C.F_RATING].map(clamp_rating)
    df = df.dropna(subset=[C.F_RATING])
    if C.F_TIMESTAMP not in df.columns:
        df[C.F_TIMESTAMP] = 0
    ts = df[C.F_TIMESTAMP].astype("int64")
    df[C.F_TIMESTAMP] = ts.where(ts < 1_000_000_000_000, ts // 1000)
    return df.reset_index(drop=True)


def build_processed_from_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge samples, filter, split, save to data/processed/."""
    reviews = prepare_reviews_df(load_all_review_samples())
    reviews = apply_quality_filters(reviews)
    reviews = add_train_test_split(reviews, config.HOLDOUT_LAST_N)

    items = load_all_meta_samples()

    rev_path = config.DATA_PROCESSED_DIR / "reviews.parquet"
    save_parquet(reviews, rev_path, desc="Processed reviews")
    if len(items):
        save_parquet(items, config.DATA_PROCESSED_DIR / "items.parquet", desc="Processed items")

    logger.info(
        f"Processed: {len(reviews):,} reviews, {len(items):,} items, "
        f"{reviews[C.F_USER_ID].nunique():,} users"
    )
    return reviews, items
