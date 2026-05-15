"""
metrics.py — Source-agnostic evaluation metrics.

Three families:
  - rating_metrics   : MAE / RMSE for rating prediction
  - recall_at_k /
    ndcg_at_k        : Retrieval quality
  - text_metrics     : ROUGE-1 / ROUGE-L for generated review text

All functions are pure: they take primitives or lists and return dicts/floats.
The runner in runner.py is responsible for wiring them to DataFrames.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Rating ────────────────────────────────────────────────────────────────────

def rating_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    """Compute MAE and RMSE between true and predicted ratings."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ── Retrieval ─────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k retrieved list."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant) / len(relevant)


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at k.
    Binary relevance: an item is either relevant (gain=1) or not (gain=0).
    """
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item_id in enumerate(retrieved_ids[:k])
        if item_id in relevant
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


# ── Text ──────────────────────────────────────────────────────────────────────

# Single shared scorer — initialising one is non-trivial.
_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def text_metrics(reference: str, prediction: str) -> dict:
    """ROUGE-1 and ROUGE-L F1 between reference and prediction."""
    reference = reference or ""
    prediction = prediction or ""
    scored = _ROUGE.score(reference, prediction)
    return {
        "rouge1_f": float(scored["rouge1"].fmeasure),
        "rougeL_f": float(scored["rougeL"].fmeasure),
    }


def mean_text_metrics(pairs: List[tuple[str, str]]) -> dict:
    """Average text metrics over many (reference, prediction) pairs."""
    if not pairs:
        return {"rouge1_f": 0.0, "rougeL_f": 0.0}
    scored = [text_metrics(ref, pred) for ref, pred in pairs]
    return {
        "rouge1_f": float(np.mean([s["rouge1_f"] for s in scored])),
        "rougeL_f": float(np.mean([s["rougeL_f"] for s in scored])),
    }
