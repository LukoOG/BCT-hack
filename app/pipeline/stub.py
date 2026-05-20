"""
Pipeline using cached samples. Uses optional Claude when ANTHROPIC_API_KEY is set.
Replace with architect's retrieval + FAISS implementation when ready.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from app.core import config
from app.core import constants as C
from app.data.sample_store import load_sample
from app.profile.user_profiles import get_user_profile


def _load_category(category: str) -> pd.DataFrame:
    return load_sample(category)


def _history_from_rows(rows: pd.DataFrame, text_col: str) -> List[dict]:
    history: List[dict] = []
    for row in rows.itertuples():
        text = str(getattr(row, text_col, "") or getattr(row, "text", "") or "")[:500]
        title = str(getattr(row, "title", "") or "")[:120]
        history.append({
            "item_id": str(getattr(row, C.F_ITEM_ID, "") or getattr(row, "item_id", "")),
            "rating": int(float(getattr(row, C.F_RATING, 0) or 0)),
            "text": text,
            "title": title,
        })
    return history


def _similar_reviews(df: pd.DataFrame, item_id: str, user_id: str, text_col: str, k: int = 3) -> List[dict]:
    others = df[(df[C.F_ITEM_ID] == item_id) & (df[C.F_USER_ID] != user_id)]
    if others.empty:
        others = df[df[C.F_ITEM_ID] != item_id].head(k * 3)
    picked = others.head(k)
    return _history_from_rows(picked, text_col)


def predict_next_review(user_id: str, category: str = "Books") -> dict:
    profile = get_user_profile(str(user_id))

    # Cross-category history for user model (Amazon user_id is global)
    try:
        from app.data.from_samples import load_all_review_samples
        df = load_all_review_samples()
    except FileNotFoundError:
        df = _load_category(category)

    text_col = C.F_REVIEW_TEXT if C.F_REVIEW_TEXT in df.columns else "text"
    uid = str(user_id) if str(user_id).startswith("amz_") else f"amz_{user_id}"

    user_rows = df[df[C.F_USER_ID].astype(str) == uid].sort_values(C.F_TIMESTAMP)
    cat_rows = user_rows[user_rows[C.F_CATEGORY] == category] if C.F_CATEGORY in user_rows.columns else user_rows
    predict_rows = cat_rows if len(cat_rows) else user_rows
    df_cat = df[df[C.F_CATEGORY] == category] if C.F_CATEGORY in df.columns else df

    user_rows = predict_rows
    if user_rows.empty:
        user_rows = df[df[C.F_USER_ID].astype(str).str.contains(str(user_id), na=False)].sort_values(
            C.F_TIMESTAMP
        )

    if user_rows.empty:
        return {
            "user_history": [],
            "prediction": {
                "rating": 4,
                "title": "(user not in sample)",
                "text": f"User {user_id} not found. Run: python scripts/build_all.py",
            },
            "retrieved": [],
            "profile": profile,
        }

    if len(user_rows) >= config.HOLDOUT_LAST_N + 1:
        history_rows = user_rows.iloc[: -config.HOLDOUT_LAST_N]
        target_row = user_rows.iloc[-1]
    else:
        history_rows = user_rows.iloc[:-1]
        target_row = user_rows.iloc[-1]

    # Profile history can include all categories; target review is in selected category
    all_history = df[df[C.F_USER_ID] == uid].sort_values(C.F_TIMESTAMP)
    history = _history_from_rows(
        all_history.iloc[:-1].tail(config.PROFILE_MAX_SAMPLE_REVIEWS)
        if len(all_history) > 1
        else history_rows.tail(config.PROFILE_MAX_SAMPLE_REVIEWS),
        text_col,
    )
    target_item = str(target_row[C.F_ITEM_ID])
    retrieved = _similar_reviews(df_cat, target_item, uid, text_col)

    item_meta = {
        "title": str(target_row.get("title", "") or "Unknown product"),
        "category": category,
        "description": str(target_row.get(text_col, ""))[:300],
    }

    # Optional real LLM
    llm_out = None
    try:
        from app.prompts.generate import generate_review
        llm_out = generate_review(history, item_meta, retrieved, category=category)
    except Exception:
        llm_out = None

    if llm_out and "rating" in llm_out and "text" in llm_out:
        prediction = {
            "rating": int(llm_out["rating"]),
            "title": str(llm_out.get("title", "")),
            "text": str(llm_out["text"]),
        }
        mode = "llm"
    else:
        avg_rating = int(round(history_rows[C.F_RATING].mean())) if len(history_rows) else int(
            target_row[C.F_RATING]
        )
        avg_rating = max(1, min(5, avg_rating))
        snippet = str(target_row.get(text_col, ""))[:200]
        prediction = {
            "rating": avg_rating,
            "title": str(target_row.get("title", "(stub) Next review"))[:80],
            "text": f"(stub/heuristic) Typical rating ~{avg_rating}. Last known: {snippet}...",
        }
        mode = "stub"

    return {
        "user_history": history,
        "prediction": prediction,
        "retrieved": retrieved,
        "profile": profile,
        "meta": {
            "mode": mode,
            "target_item_id": target_item,
            "category_count": (profile or {}).get("category_count"),
            "top_categories": (profile or {}).get("top_categories"),
        },
    }
