"""
Task B — personalized item recommendation.

Reuses user profiles + FAISS retrieval over review embeddings, aggregated to item level.
No Wi-Fi required — works on cached samples only.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from app.core import config
from app.core import constants as C
from app.data.sample_store import has_sample, load_sample
from app.pipeline.retriever import get_retriever
from app.profile.user_profiles import get_user_profile
from app.utils.logger import logger


def _normalize_uid(user_id: str) -> str:
    uid = str(user_id)
    return uid if uid.startswith("amz_") else f"amz_{uid}"


def _profile_query(profile: dict) -> str:
    raw = profile.get("sample_reviews")
    if raw is None:
        samples = []
    elif hasattr(raw, "tolist"):
        samples = raw.tolist() if not isinstance(raw, list) else raw
    else:
        samples = list(raw) if raw else []
    parts = [str(r.get("text", "")) for r in samples[:3] if isinstance(r, dict) and r.get("text")]
    if parts:
        return " ".join(parts)
    top = profile.get("top_categories") or []
    if hasattr(top, "tolist"):
        top = top.tolist()
    return f"interested in {', '.join(str(x) for x in top)}" if top else "product recommendations"


def _user_seen_items(user_id: str, category: str) -> set[str]:
    if not has_sample(category):
        return set()
    df = load_sample(category)
    uid = _normalize_uid(user_id)
    return set(df[df[C.F_USER_ID].astype(str) == uid][C.F_ITEM_ID].astype(str).tolist())


def _item_stats(category: str) -> pd.DataFrame:
    """Per-item avg rating and review count from raw sample."""
    if not has_sample(category):
        return pd.DataFrame()
    df = load_sample(category)
    g = df.groupby(C.F_ITEM_ID).agg(
        avg_rating=(C.F_RATING, "mean"),
        review_count=(C.F_RATING, "count"),
    )
    return g.reset_index()


def recommend_items(
    user_id: str,
    category: str = "Books",
    *,
    k: int = 10,
    exclude_seen: bool = True,
    seen_item_ids: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Rank items for a user using profile + semantic retrieval.

    Scoring (hybrid, hackathon MVP):
      0.6 * retrieval score (FAISS)
      0.3 * category preference match
      0.1 * item avg rating (normalized)
    """
    uid = _normalize_uid(user_id)
    profile = get_user_profile(uid)
    if profile is None:
        return {
            "recommendations": [],
            "profile": None,
            "meta": {"mode": "task_b", "error": "user_not_found"},
        }

    if category not in config.AMAZON_CATEGORIES:
        category = "Books"

    query = _profile_query(profile)
    if seen_item_ids is not None:
        seen = {str(i) for i in seen_item_ids}
    elif exclude_seen:
        seen = _user_seen_items(uid, category)
    else:
        seen = set()
    raw_top = profile.get("top_categories")
    if raw_top is None:
        top_cats: set[str] = set()
    elif hasattr(raw_top, "tolist"):
        top_cats = set(raw_top.tolist())
    else:
        top_cats = set(raw_top)

    try:
        retriever = get_retriever(category, auto_build=True)
        hits = retriever.search(query, k=max(k * 10, 30), exclude_user=uid)
    except Exception as exc:
        logger.warning(f"Task B retrieval failed for {category}: {exc}")
        hits = []

    stats = _item_stats(category)
    stats_map = {
        str(r[C.F_ITEM_ID]): (float(r["avg_rating"]), int(r["review_count"]))
        for _, r in stats.iterrows()
    } if len(stats) else {}

    # Aggregate best hit per item
    best_by_item: dict[str, dict] = {}
    for hit in hits:
        iid = str(hit["item_id"])
        if exclude_seen and iid in seen:
            continue
        prev = best_by_item.get(iid)
        if prev is None or hit.get("score", 0) > prev.get("score", 0):
            best_by_item[iid] = hit

    candidates = []
    for iid, hit in best_by_item.items():
        retr = float(hit.get("score", 0))
        cat_bonus = 0.15 if category in top_cats else 0.0
        avg_r, _ = stats_map.get(iid, (3.5, 0))
        quality = (avg_r - 1.0) / 4.0  # map 1-5 to 0-1
        score = 0.6 * retr + 0.3 * cat_bonus + 0.1 * quality
        candidates.append({
            "item_id": iid,
            "title": hit.get("title") or "Unknown item",
            "score": round(score, 4),
            "avg_rating": round(avg_r, 2),
            "snippet": str(hit.get("text", ""))[:200],
            "reason": f"Similar to your review style; avg rating {avg_r:.1f}",
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    recs = candidates[:k]

    return {
        "recommendations": recs,
        "profile": {
            "user_id": uid,
            "avg_rating": profile.get("avg_rating"),
            "top_categories": profile.get("top_categories"),
            "review_count": profile.get("review_count"),
        },
        "meta": {
            "mode": "task_b",
            "category": category,
            "retrieval": "faiss",
            "query_chars": len(query),
            "candidates": len(candidates),
        },
    }
