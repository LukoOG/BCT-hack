"""
Demo: run eval metrics on synthetic predictions (no LLM required).

Usage:
    python scripts/run_eval_demo.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eval.runner import evaluate_predictions, format_results_markdown


def main():
    holdout = pd.DataFrame([
        {"user_id": "u1", "item_id": "i1", "true_rating": 5.0, "true_text": "Great book, loved every page."},
        {"user_id": "u2", "item_id": "i2", "true_rating": 3.0, "true_text": "Okay but overpriced."},
    ])
    preds = [
        {"user_id": "u1", "item_id": "i1", "pred_rating": 4.0,
         "pred_text": "Really enjoyed this book.", "retrieved_item_ids": ["i1", "i9"]},
        {"user_id": "u2", "item_id": "i2", "pred_rating": 3.0,
         "pred_text": "Average product for the price.", "retrieved_item_ids": ["i2"]},
    ]
    results = evaluate_predictions(holdout, preds)
    print(format_results_markdown(results))


if __name__ == "__main__":
    main()
