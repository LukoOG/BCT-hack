"""
catalog.py — Drop-in router for judge-facing discovery endpoints.

Add to server.py with:
    from app.api.catalog import router as catalog_router
    app.include_router(catalog_router)

Endpoints
─────────
GET /catalog/users
    Returns a sample of valid user_ids judges can paste into /predict or /recommend.
    Query params:
        category  str   filter to users who have reviews in this category (optional)
        limit     int   max results, default 20, max 100

GET /catalog/items
    Returns a sample of valid item_ids judges can pass as target_item_id in /predict.
    Query params:
        category  str   filter by category (optional)
        limit     int   max results, default 20, max 100

GET /catalog/sample
    Returns one ready-to-use (user_id, item_id, category) triple — paste directly
    into /predict to get a result with zero setup.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/catalog", tags=["Catalog — DB Discovery"])

# ── Lazy-load processed data ───────────────────────────────────────────────────
# Mirrors the same pattern used in the rest of the pipeline so we don't
# introduce a second loading path.

_reviews_df: Optional[pd.DataFrame] = None
_items_df:   Optional[pd.DataFrame] = None


def _get_reviews() -> pd.DataFrame:
    global _reviews_df
    if _reviews_df is None:
        try:
            _reviews_df = pd.read_parquet("data/processed/reviews.parquet")
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Processed data not found. "
                    "Run: python scripts/build_all.py --skip-fetch"
                )
            )
    return _reviews_df


def _get_items() -> pd.DataFrame:
    global _items_df
    if _items_df is None:
        try:
            _items_df = pd.read_parquet("data/processed/items.parquet")
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Item metadata not found. "
                    "Run: python scripts/build_all.py --skip-fetch"
                )
            )
    return _items_df


# ── Response models ────────────────────────────────────────────────────────────

class UserEntry(BaseModel):
    user_id: str
    review_count: int
    avg_rating: float
    top_category: str
    sample_use: dict   # ready-to-paste /predict request body


class ItemEntry(BaseModel):
    item_id: str
    title: str
    category: str
    avg_rating: float
    review_count: int
    sample_use: dict   # ready-to-paste /predict request body with this item


class SampleTriple(BaseModel):
    user_id: str
    item_id: str
    category: str
    predict_request: dict    # paste this directly into POST /predict
    recommend_request: dict  # paste this directly into POST /recommend
    note: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get(
    "/users",
    response_model=list[UserEntry],
    summary="List valid user IDs",
    description=(
        "Returns users that have enough review history to produce a meaningful prediction. "
        "Use any `user_id` from this list in POST /predict or POST /recommend."
    ),
)
def list_users(
    category: Optional[str] = Query(
        None,
        description="Filter to users who have reviews in this category "
                    "(e.g. Books, Electronics, Home_and_Kitchen)"
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of results (max 100)"),
):
    df = _get_reviews()

    if category:
        df = df[df["category"].str.lower() == category.lower()]
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No reviews found for category '{category}'. "
                       f"Available: {_get_reviews()['category'].unique().tolist()}"
            )

    # Aggregate per user
    agg = (
        df.groupby("user_id")
          .agg(
              review_count=("rating", "count"),
              avg_rating=("rating", "mean"),
              top_category=("category", lambda x: x.value_counts().index[0]),
          )
          .reset_index()
          .sort_values("review_count", ascending=False)
          .head(limit)
    )

    return [
        UserEntry(
            user_id=row.user_id,
            review_count=int(row.review_count),
            avg_rating=round(float(row.avg_rating), 2),
            top_category=row.top_category,
            sample_use={
                "endpoint": "POST /predict",
                "body": {
                    "user_id": row.user_id,
                    "category": row.top_category,
                    "target_item_id": None,
                }
            }
        )
        for row in agg.itertuples()
    ]


@router.get(
    "/items",
    response_model=list[ItemEntry],
    summary="List valid item IDs",
    description=(
        "Returns items that appear in the processed dataset. "
        "Use any `item_id` as `target_item_id` in POST /predict."
    ),
)
def list_items(
    category: Optional[str] = Query(
        None,
        description="Filter by category (e.g. Books, Electronics)"
    ),
    limit: int = Query(20, ge=1, le=100, description="Number of results (max 100)"),
):
    reviews_df = _get_reviews()
    items_df   = _get_items()

    # Aggregate review counts per item from reviews
    item_stats = (
        reviews_df.groupby("item_id")
                  .agg(review_count=("rating", "count"), avg_review_rating=("rating", "mean"))
                  .reset_index()
    )

    # Join with item metadata for title and category
    merged = item_stats.merge(
        items_df[["item_id", "title", "category", "avg_rating"]].drop_duplicates("item_id"),
        on="item_id",
        how="left"
    )

    if category:
        merged = merged[merged["category"].str.lower() == category.lower()]
        if merged.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No items found for category '{category}'."
            )

    merged = merged.sort_values("review_count", ascending=False).head(limit)

    return [
        ItemEntry(
            item_id=row.item_id,
            title=str(row.title or "(no title)"),
            category=str(row.category or "unknown"),
            avg_rating=round(float(row.avg_rating or 0), 2),
            review_count=int(row.review_count),
            sample_use={
                "endpoint": "POST /predict",
                "body": {
                    "user_id": "<pick any user_id from GET /catalog/users>",
                    "category": str(row.category or "Books"),
                    "target_item_id": row.item_id,
                }
            }
        )
        for row in merged.itertuples()
    ]


@router.get(
    "/sample",
    response_model=SampleTriple,
    summary="Get one ready-to-use test triple",
    description=(
        "Returns a single (user_id, item_id, category) triple with pre-built request bodies "
        "you can paste directly into POST /predict and POST /recommend. "
        "The user and item are guaranteed to exist in the processed dataset."
    ),
)
def get_sample(
    category: Optional[str] = Query(
        "Books",
        description="Category to sample from (default: Books)"
    ),
):
    df = _get_reviews()
    cat_df = df[df["category"].str.lower() == (category or "books").lower()]

    if cat_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for category '{category}'. "
                   f"Available: {df['category'].unique().tolist()}"
        )

    # Pick the user with the most reviews in this category (most reliable test subject)
    user_counts = cat_df["user_id"].value_counts()
    best_user   = user_counts.index[0]

    # Pick the most-reviewed item in this category (most likely to have rich context)
    item_counts = cat_df["item_id"].value_counts()
    best_item   = item_counts.index[0]

    actual_category = cat_df["category"].iloc[0]

    return SampleTriple(
        user_id=best_user,
        item_id=best_item,
        category=actual_category,
        predict_request={
            "user_id": best_user,
            "category": actual_category,
            "target_item_id": best_item,
        },
        recommend_request={
            "user_id": best_user,
            "category": actual_category,
            "k": 5,
        },
        note=(
            f"This user has {int(user_counts[best_user])} reviews in {actual_category}. "
            f"The item has {int(item_counts[best_item])} reviews. "
            "Paste predict_request into POST /predict, or recommend_request into POST /recommend."
        )
    )
