"""
Stub pipeline — uses cached Parquet samples until retrieval + LLM are wired.

Replace this module's implementation when the architect connects embeddings/FAISS/Claude.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from app.core import config
from app.core import constants as C


def _load_category(category: str) -> pd.DataFrame:
    path = config.DATA_RAW_DIR / f"{category.lower()}_reviews_sample.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No sample for {category}. Run: python scripts/fetch_samples.py"
        )
    return pd.read_parquet(path)


def predict_next_review(user_id: str, category: str = "Books") -> dict:
    """
    Sample-data stub: returns the user's last reviews from the cached sample and a
    placeholder prediction. Signature matches TEAM_CONTRACT.md / streamlit_app.py.
    """
    df = _load_category(category)
    user_col = C.F_USER_ID
    text_col = C.F_REVIEW_TEXT

    user_rows = df[df[user_col] == user_id].sort_values(C.F_TIMESTAMP)
    if user_rows.empty:
        # Prefix match for HF ids without amz_ prefix in samples
        user_rows = df[df[user_col].astype(str).str.contains(user_id, na=False)]

    history: List[dict] = []
    for row in user_rows.tail(config.PROFILE_MAX_SAMPLE_REVIEWS).itertuples():
        history.append({
            "item_id": str(getattr(row, C.F_ITEM_ID, "") or getattr(row, "item_id", "")),
            "rating":  int(getattr(row, C.F_RATING, 0) or 0),
            "text":    str(getattr(row, text_col, "") or "")[:500],
        })

    if not history:
        return {
            "user_history": [],
            "prediction": {
                "rating": 4,
                "title": "(no user in sample)",
                "text": f"User {user_id} not found in {category} sample. "
                        "Run fetch_samples with more rows or use a real pipeline.",
            },
            "retrieved": [],
        }

    last = history[-1]
    return {
        "user_history": history[:-1] or history,
        "prediction": {
            "rating": last["rating"],
            "title":  "(stub) Next review",
            "text":   f"(stub) Based on {len(history)} past reviews in sample.",
        },
        "retrieved": history[:2],
    }
