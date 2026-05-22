"""
Recommendation agent wrapper built on the same data/model signals as Task A.

The goal is a practical hackathon baseline: use a user's review history as the
query, retrieve similar reviews, aggregate them by item, and return ranked items.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import pandas as pd

from app.core import constants as C
from app.data.sample_store import has_sample, load_sample
from app.pipeline.stub import _history_from_rows, _load_reviews, _normalize_uid, _text_col
from app.pipeline.retriever import get_retriever
from app.profile.user_profiles import get_user_profile
from app.utils.logger import logger


def _query_from_rows(rows: pd.DataFrame, text_col: str) -> str:
    if rows.empty:
        return ""
    recent = rows.tail(5)
    return " ".join(recent[text_col].fillna("").astype(str).tolist())


def _popular_items(corpus: pd.DataFrame, exclude_items: set[str], k: int) -> list[dict]:
    grouped = (
        corpus[~corpus[C.F_ITEM_ID].astype(str).isin(exclude_items)]
        .groupby(C.F_ITEM_ID)
        .agg(
            mean_rating=(C.F_RATING, "mean"),
            review_count=(C.F_RATING, "size"),
            title=("title", "first") if "title" in corpus.columns else (C.F_RATING, "size"),
        )
        .reset_index()
    )
    if grouped.empty:
        return []
    grouped["score"] = grouped["mean_rating"] * 0.7 + grouped["review_count"].clip(upper=50) / 50 * 0.3
    grouped = grouped.sort_values(["score", "review_count", "mean_rating"], ascending=False).head(k)
    recs: list[dict] = []
    for row in grouped.itertuples():
        recs.append({
            "item_id": str(getattr(row, C.F_ITEM_ID)),
            "score": float(row.score),
            "title": str(getattr(row, "title", "") or ""),
            "reason": "Popular with strong average rating in this category.",
        })
    return recs


def _candidate_rows(corpus: pd.DataFrame, candidate_item_ids: Iterable[str]) -> pd.DataFrame:
    candidates = {str(i) if str(i).startswith("amz_") else f"amz_{i}" for i in candidate_item_ids}
    return corpus[corpus[C.F_ITEM_ID].astype(str).isin(candidates)]


def recommend_items(
    user_id: str,
    category: str = "Books",
    *,
    k: int = 10,
    candidate_item_ids: Optional[list[str]] = None,
) -> dict:
    """
    Rank recommended items for a user.

    If candidate_item_ids is supplied, ranking is restricted to those items. This
    is useful for offline Task B evaluation with sampled negatives.
    """
    uid = _normalize_uid(user_id)
    profile = get_user_profile(str(user_id))

    try:
        corpus = load_sample(category) if has_sample(category) else _load_reviews(category)
    except FileNotFoundError:
        return {
            "recommendations": [],
            "user_history": [],
            "profile": profile,
            "meta": {"mode": "no-data", "category": category},
        }

    text_col = _text_col(corpus)
    user_rows = corpus[corpus[C.F_USER_ID].astype(str) == uid].sort_values(C.F_TIMESTAMP)
    history_rows = user_rows.iloc[:-1] if len(user_rows) > 1 else user_rows.iloc[:0]
    seen_items = set(user_rows[C.F_ITEM_ID].astype(str).tolist())
    query = _query_from_rows(history_rows, text_col)
    history = _history_from_rows(history_rows.tail(5), text_col)

    if candidate_item_ids:
        candidates = _candidate_rows(corpus, candidate_item_ids)
        if candidates.empty:
            return {
                "recommendations": [],
                "user_history": history,
                "profile": profile,
                "meta": {"mode": "candidate-rank", "category": category, "candidate_count": 0},
            }
        user_avg = float(history_rows[C.F_RATING].mean()) if len(history_rows) else 4.0
        if pd.isna(user_avg):
            user_avg = 4.0
        grouped = candidates.groupby(C.F_ITEM_ID)
        recs: list[dict] = []
        for item_id, grp in grouped:
            title = str(grp["title"].dropna().astype(str).iloc[0]) if "title" in grp.columns and grp["title"].notna().any() else ""
            item_avg = float(grp[C.F_RATING].mean())
            count_bonus = min(len(grp), 25) / 25
            rating_fit = 1.0 - min(abs(user_avg - item_avg), 4.0) / 4.0
            score = rating_fit * 0.75 + count_bonus * 0.25
            recs.append({
                "item_id": str(item_id),
                "score": float(score),
                "title": title,
                "reason": f"Matches the user's {user_avg:.1f}-star rating tendency.",
            })
        recs.sort(key=lambda r: r["score"], reverse=True)
        return {
            "recommendations": recs[:k],
            "user_history": history,
            "profile": profile,
            "meta": {"mode": "candidate-rank", "category": category, "candidate_count": len(recs)},
        }

    scores: dict[str, dict] = defaultdict(lambda: {"score": 0.0, "count": 0, "title": ""})
    retrieval_mode = "faiss"
    try:
        hits = get_retriever(category, auto_build=True).search(query, k=max(k * 8, 20), exclude_user=uid)
        for rank, hit in enumerate(hits):
            item_id = str(hit["item_id"])
            if item_id in seen_items:
                continue
            scores[item_id]["score"] += float(hit.get("score", 0.0)) + 1.0 / (rank + 1)
            scores[item_id]["count"] += 1
            scores[item_id]["title"] = str(hit.get("title", "") or scores[item_id]["title"])
    except Exception as exc:
        logger.warning(f"Recommendation retrieval failed for {category}: {exc}")
        retrieval_mode = "popular"

    recs = [
        {
            "item_id": item_id,
            "score": float(data["score"]),
            "title": str(data["title"] or ""),
            "reason": "Similar to reviews in the user's recent history.",
        }
        for item_id, data in scores.items()
    ]
    recs.sort(key=lambda r: r["score"], reverse=True)
    if len(recs) < k:
        recs.extend(_popular_items(corpus, seen_items | {r["item_id"] for r in recs}, k - len(recs)))

    return {
        "recommendations": recs[:k],
        "user_history": history,
        "profile": profile,
        "meta": {"mode": "retrieval+profile", "retrieval": retrieval_mode, "category": category},
    }
