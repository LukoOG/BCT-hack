"""
FAISS-backed review predictor using local TF-IDF embeddings.

No HuggingFace model downloads — indexes are built from cached parquet only.
Optional Claude generation when ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from app.core import config
from app.core import constants as C
from app.data.sample_store import load_sample
from app.pipeline.retriever import get_retriever
from app.profile.user_profiles import get_user_profile
from app.utils.logger import logger


def _load_reviews(category: str) -> pd.DataFrame:
    from app.data.sample_store import has_sample, load_sample
    if has_sample(category):
        return load_sample(category)

    proc = config.DATA_PROCESSED_DIR / "reviews.parquet"
    if proc.exists():
        df = pd.read_parquet(proc)
        if C.F_CATEGORY in df.columns:
            cat = df[df[C.F_CATEGORY] == category]
            if len(cat):
                return cat.reset_index(drop=True)
        return df
    return load_sample(category)


def _text_col(df: pd.DataFrame) -> str:
    return C.F_REVIEW_TEXT if C.F_REVIEW_TEXT in df.columns else "text"


def _normalize_uid(user_id: str) -> str:
    uid = str(user_id)
    return uid if uid.startswith("amz_") else f"amz_{uid}"


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


def _query_from_history(history: List[dict]) -> str:
    if not history:
        return ""
    recent = history[-3:]
    return " ".join(h["text"] for h in recent if h.get("text"))


def _heuristic_prediction(history: List[dict], retrieved: List[dict], fallback_rating: float) -> dict:
    ratings = [h["rating"] for h in history if h.get("rating")]
    if retrieved:
        ratings.extend(r["rating"] for r in retrieved if r.get("rating"))
    avg = int(round(sum(ratings) / len(ratings))) if ratings else int(round(fallback_rating))
    avg = max(1, min(5, avg))

    snippet = ""
    if retrieved and retrieved[0].get("text"):
        snippet = retrieved[0]["text"][:220]
    elif history and history[-1].get("text"):
        snippet = history[-1]["text"][:220]

    title = retrieved[0].get("title", "") if retrieved else (history[-1].get("title", "") if history else "")
    text = (
        f"Based on your review style and similar shoppers, I'd expect around {avg} stars. "
        f"{snippet}"
    ).strip()

    return {"rating": avg, "title": title or "Predicted review", "text": text}


def _item_meta_from_corpus(
    corpus: pd.DataFrame,
    target_item: str,
    category: str,
    text_col: str,
) -> dict:
    """Build item context from other users' reviews — never from the holdout row."""
    peers = corpus[corpus[C.F_ITEM_ID].astype(str) == target_item]
    title = ""
    if len(peers) and "title" in peers.columns:
        titles = peers["title"].dropna().astype(str)
        title = titles.iloc[0] if len(titles) else ""
    desc = ""
    if len(peers):
        desc = " ".join(peers[text_col].fillna("").astype(str).head(3).tolist())[:300]
    return {
        "title": title or "Unknown product",
        "category": category,
        "description": desc,
    }


def predict_next_review(
    user_id: str,
    category: str = "Books",
    *,
    target_item_id: str | None = None,
) -> dict:
    profile = get_user_profile(str(user_id))
    uid = _normalize_uid(user_id)

    try:
        df = _load_reviews(category)
    except FileNotFoundError:
        return {
            "user_history": [],
            "prediction": {
                "rating": 4,
                "title": "(no local data)",
                "text": "Run: python scripts/build_all.py --skip-fetch",
            },
            "retrieved": [],
            "profile": profile,
            "meta": {"mode": "stub", "retrieval": "none"},
        }

    text_col = _text_col(df)
    user_rows = df[df[C.F_USER_ID].astype(str) == uid].sort_values(C.F_TIMESTAMP)

    if user_rows.empty:
        return {
            "user_history": [],
            "prediction": {
                "rating": 4,
                "title": "(user not in sample)",
                "text": f"User {user_id} not found in cached {category} data.",
            },
            "retrieved": [],
            "profile": profile,
            "meta": {"mode": "stub", "retrieval": "none"},
        }

    if len(user_rows) >= config.HOLDOUT_LAST_N + 1:
        history_rows = user_rows.iloc[: -config.HOLDOUT_LAST_N]
        target_row = user_rows.iloc[-1]
    else:
        history_rows = user_rows.iloc[:-1] if len(user_rows) > 1 else user_rows.iloc[:0]
        target_row = user_rows.iloc[-1]

    if target_item_id:
        target_item = str(target_item_id)
        if not target_item.startswith("amz_"):
            target_item = f"amz_{target_item}"
    else:
        target_item = str(target_row[C.F_ITEM_ID])

    # Cross-category history for personalization
    try:
        from app.data.from_samples import load_all_review_samples
        all_df = load_all_review_samples()
        all_user = all_df[all_df[C.F_USER_ID].astype(str) == uid].sort_values(C.F_TIMESTAMP)
        hist_source = (
            all_user.iloc[: -config.HOLDOUT_LAST_N].tail(config.PROFILE_MAX_SAMPLE_REVIEWS)
            if len(all_user) > config.HOLDOUT_LAST_N
            else history_rows.tail(config.PROFILE_MAX_SAMPLE_REVIEWS)
        )
    except FileNotFoundError:
        hist_source = history_rows.tail(config.PROFILE_MAX_SAMPLE_REVIEWS)

    history = _history_from_rows(hist_source, text_col)
    query = _query_from_history(history)
    if not query and target_row is not None:
        query = str(target_row.get(text_col, ""))

    retrieved: List[dict] = []
    retrieval_mode = "faiss"
    try:
        from app.data.sample_store import has_sample, load_sample
        corpus = load_sample(category) if has_sample(category) else df
    except FileNotFoundError:
        corpus = df

    try:
        retriever = get_retriever(category, auto_build=True)
        same_item_rows = corpus[
            (corpus[C.F_ITEM_ID].astype(str) == target_item)
            & (corpus[C.F_USER_ID].astype(str) != uid)
        ]
        retrieved = _history_from_rows(same_item_rows.head(5), text_col)
        if len(retrieved) < 5:
            faiss_hits = retriever.search(
                query,
                k=5 - len(retrieved),
                exclude_user=uid,
                prefer_item_id=target_item,
            )
            seen = {r["item_id"] + r["text"][:40] for r in retrieved}
            for hit in faiss_hits:
                key = hit["item_id"] + hit["text"][:40]
                if key not in seen:
                    retrieved.append({k: v for k, v in hit.items() if k != "score"})
                    seen.add(key)
                if len(retrieved) >= 5:
                    break
    except Exception as exc:
        logger.warning(f"FAISS retrieval failed for {category}: {exc}")
        retrieval_mode = "fallback"
        others = corpus[
            (corpus[C.F_ITEM_ID].astype(str) == target_item)
            & (corpus[C.F_USER_ID].astype(str) != uid)
        ]
        if others.empty:
            others = corpus[corpus[C.F_USER_ID].astype(str) != uid].head(5)
        retrieved = _history_from_rows(others.head(5), text_col)

    if target_item_id:
        item_meta = _item_meta_from_corpus(corpus, target_item, category, text_col)
    else:
        item_meta = {
            "title": str(target_row.get("title", "") or "Unknown product"),
            "category": category,
            "description": str(target_row.get(text_col, ""))[:300],
        }

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
        fallback = float(history_rows[C.F_RATING].mean()) if len(history_rows) else 4.0
        if pd.isna(fallback):
            fallback = 4.0
        prediction = _heuristic_prediction(history, retrieved, fallback)
        mode = "faiss+heuristic"

    return {
        "user_history": history,
        "prediction": prediction,
        "retrieved": retrieved,
        "profile": profile,
        "meta": {
            "mode": mode,
            "retrieval": retrieval_mode,
            "target_item_id": target_item,
            "category_count": (profile or {}).get("category_count"),
            "top_categories": (profile or {}).get("top_categories"),
        },
    }
