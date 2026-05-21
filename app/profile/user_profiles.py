"""
User profiles for personalization and recommendation.

Built from all cached category samples — same user_id across categories
is merged into one profile (Amazon user IDs are global).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core import config
from app.core import constants as C
from app.data.from_samples import load_all_review_samples, prepare_reviews_df

PROFILES_PATH = config.DATA_PROCESSED_DIR / "user_profiles.parquet"


def build_user_profiles(reviews: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if reviews is None:
        reviews = prepare_reviews_df(load_all_review_samples())

    reviews = reviews.sort_values([C.F_USER_ID, C.F_TIMESTAMP])

    rows: List[Dict[str, Any]] = []
    for user_id, grp in reviews.groupby(C.F_USER_ID):
        texts = grp[C.F_REVIEW_TEXT].fillna("").astype(str)
        cats = grp[C.F_CATEGORY].value_counts()
        top_cats = cats.head(config.PROFILE_MAX_TOP_CATEGORIES).index.tolist()
        sample = grp.tail(config.PROFILE_MAX_SAMPLE_REVIEWS)

        rows.append({
            C.F_USER_ID: user_id,
            "review_count": int(len(grp)),
            "avg_rating": float(grp[C.F_RATING].mean()),
            "rating_std": float(grp[C.F_RATING].std()) if len(grp) > 1 else 0.0,
            "review_length_avg": float(texts.str.split().str.len().mean()),
            "top_categories": top_cats,
            "category_count": int(grp[C.F_CATEGORY].nunique()),
            "sample_reviews": [
                {
                    "item_id": r[C.F_ITEM_ID],
                    "rating": float(r[C.F_RATING]),
                    "text": str(r[C.F_REVIEW_TEXT])[:400],
                    "category": r[C.F_CATEGORY],
                }
                for _, r in sample.iterrows()
            ],
        })

    profiles = pd.DataFrame(rows)
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(PROFILES_PATH, index=False)
    return profiles


def load_user_profiles() -> pd.DataFrame:
    if not PROFILES_PATH.exists():
        try:
            return build_user_profiles()
        except FileNotFoundError:
            return pd.DataFrame()
    return pd.read_parquet(PROFILES_PATH)


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    df = load_user_profiles()
    if df.empty or C.F_USER_ID not in df.columns:
        return None
    hit = df[df[C.F_USER_ID] == user_id]
    if hit.empty:
        hit = df[df[C.F_USER_ID].astype(str).str.contains(str(user_id), na=False)]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()
