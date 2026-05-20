"""
Build a tiny holdout set from cached samples and score stub/heuristic predictions.

Usage:
    python scripts/run_eval_on_sample.py
    python scripts/run_eval_on_sample.py --category Books --max-users 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import config
from app.core import constants as C
from app.data.sample_store import load_sample
from app.eval.runner import evaluate_predictions, format_results_markdown
from app.pipeline.stub import predict_next_review


def build_holdout(category: str, max_users: int) -> tuple[pd.DataFrame, list[dict]]:
    df = load_sample(category).sort_values([C.F_USER_ID, C.F_TIMESTAMP])
    holdout_rows = []
    predictions = []

    for user_id, grp in df.groupby(C.F_USER_ID):
        if len(grp) < config.HOLDOUT_LAST_N + 1:
            continue
        if len(holdout_rows) >= max_users:
            break

        test = grp.iloc[-config.HOLDOUT_LAST_N :]

        for _, row in test.iterrows():
            holdout_rows.append({
                "user_id": user_id,
                "item_id": row[C.F_ITEM_ID],
                "true_rating": float(row[C.F_RATING]),
                "true_text": str(row.get(C.F_REVIEW_TEXT) or row.get("text") or ""),
            })

        # Stub uses full sample; for eval we still call it (simulates pipeline)
        pred = predict_next_review(str(user_id), category)
        predictions.append({
            "user_id": user_id,
            "item_id": holdout_rows[-1]["item_id"],
            "pred_rating": float(pred["prediction"]["rating"]),
            "pred_text": str(pred["prediction"]["text"]),
            "retrieved_item_ids": [r["item_id"] for r in pred.get("retrieved", [])],
        })

    return pd.DataFrame(holdout_rows), predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Books")
    parser.add_argument("--max-users", type=int, default=25)
    args = parser.parse_args()

    holdout, preds = build_holdout(args.category, args.max_users)
    if holdout.empty:
        print("No users with enough reviews. Run fetch_samples with larger --size.")
        return 1

    results = evaluate_predictions(holdout, preds)
    print(format_results_markdown(results))

    out = ROOT / "notebooks" / "outputs" / f"eval_{args.category.lower()}.json"
    import json
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
