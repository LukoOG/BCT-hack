"""
runner.py — Orchestrates evaluation over a held-out set.

Expected input shapes
─────────────────────
holdout_df : pd.DataFrame with at minimum:
    user_id, item_id, true_rating, true_text
    (rename from the unified-schema columns at the call site)

predictions : list[dict] each shaped like:
    {
      "user_id": str,
      "item_id": str,
      "pred_rating": float,
      "pred_text":   str,
      "retrieved_item_ids": list[str],   # optional, enables retrieval metrics
    }

The runner is intentionally agnostic to *how* predictions were produced.
It only consumes them and produces a metrics dict.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from app.utils.logger import logger
from .metrics import (
    mean_text_metrics,
    ndcg_at_k,
    rating_metrics,
    recall_at_k,
)


def evaluate_predictions(
    holdout_df: pd.DataFrame,
    predictions: List[dict],
    k_values: Optional[List[int]] = None,
) -> dict:
    """
    Compute rating + text + (optionally) retrieval metrics over a held-out set.

    Returns a nested dict suitable for logging or JSON dump:
        {
          "n": int,
          "rating": {"mae": ..., "rmse": ...},
          "text":   {"rouge1_f": ..., "rougeL_f": ...},
          "retrieval": {"recall@5": ..., "ndcg@5": ..., ...}  # if available
        }
    """
    if not predictions:
        logger.warning("evaluate_predictions called with empty predictions")
        return {"n": 0}

    k_values = k_values or [5, 10]
    pred_df = pd.DataFrame(predictions)

    merged = holdout_df.merge(pred_df, on=["user_id", "item_id"], how="inner")
    if merged.empty:
        logger.warning("No overlap between holdout and predictions on (user_id, item_id)")
        return {"n": 0}

    results: dict = {"n": int(len(merged))}

    # Rating
    results["rating"] = rating_metrics(merged["true_rating"], merged["pred_rating"])

    # Text
    pairs = list(zip(merged["true_text"].fillna(""), merged["pred_text"].fillna("")))
    results["text"] = mean_text_metrics(pairs)

    # Retrieval (only if predictions carried retrieved_item_ids)
    if "retrieved_item_ids" in merged.columns:
        retrieval_scores: dict = {}
        for k in k_values:
            recalls, ndcgs = [], []
            for row in merged.itertuples():
                retrieved = list(getattr(row, "retrieved_item_ids") or [])
                # "Relevant" = the true item the user actually reviewed next.
                relevant = [row.item_id]
                recalls.append(recall_at_k(retrieved, relevant, k))
                ndcgs.append(ndcg_at_k(retrieved, relevant, k))
            retrieval_scores[f"recall@{k}"] = float(sum(recalls) / len(recalls))
            retrieval_scores[f"ndcg@{k}"]   = float(sum(ndcgs) / len(ndcgs))
        results["retrieval"] = retrieval_scores

    logger.info(f"Evaluation complete on {results['n']} examples: {results}")
    return results


def format_results_markdown(results: dict) -> str:
    """Pretty-print results dict as a Markdown table — handy for notebooks."""
    if not results or results.get("n", 0) == 0:
        return "_No results._"

    lines = [f"**N = {results['n']}**", "", "| Metric | Value |", "|---|---|"]
    for group, scores in results.items():
        if group == "n" or not isinstance(scores, dict):
            continue
        for name, value in scores.items():
            lines.append(f"| {group}.{name} | {value:.4f} |")
    return "\n".join(lines)
