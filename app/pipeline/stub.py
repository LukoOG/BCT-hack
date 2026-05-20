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
    df = _load_category(category)
    text_col = C.F_REVIEW_TEXT if C.F_REVIEW_TEXT in df.columns else "text"

    user_rows = df[df[C.F_USER_ID].astype(str) == str(user_id)].sort_values(C.F_TIMESTAMP)
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
                "text": f"User {user_id} not found. Run: python scripts/fetch_samples.py --size 20000",
            },
            "retrieved": [],
        }

    if len(user_rows) >= config.HOLDOUT_LAST_N + 1:
        history_rows = user_rows.iloc[: -config.HOLDOUT_LAST_N]
        target_row = user_rows.iloc[-1]
    else:
        history_rows = user_rows.iloc[:-1]
        target_row = user_rows.iloc[-1]

    history = _history_from_rows(history_rows.tail(config.PROFILE_MAX_SAMPLE_REVIEWS), text_col)
    target_item = str(target_row[C.F_ITEM_ID])
    retrieved = _similar_reviews(df, target_item, str(user_id), text_col)

    item_meta = {
        "title": str(target_row.get("title", "") or "Unknown product"),
        "category": category,
        "description": str(target_row.get(text_col, ""))[:300],
    }

    # Optional real LLM
    llm_out = None
    try:
        from app.prompts.generate import generate_review
        llm_out = generate_review(history, item_meta, retrieved)
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
        "meta": {"mode": mode, "target_item_id": target_item},
    }
