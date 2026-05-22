"""
Task B holdout eval — last reviewed item should appear in top-k recommendations.

Usage:
    python scripts/run_eval_task_b.py --category Books --max-users 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import config
from app.core import constants as C
from app.data.sample_store import load_sample
from app.eval.metrics import ndcg_at_k, recall_at_k
from app.tasks.task_b import recommend_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Books")
    parser.add_argument("--max-users", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    df = load_sample(args.category).sort_values([C.F_USER_ID, C.F_TIMESTAMP])
    recalls, ndcgs = [], []
    n = 0

    for user_id, grp in df.groupby(C.F_USER_ID):
        if len(grp) < config.HOLDOUT_LAST_N + 1:
            continue
        if n >= args.max_users:
            break
        true_item = str(grp.iloc[-1][C.F_ITEM_ID])
        train_items = set(grp.iloc[:-config.HOLDOUT_LAST_N][C.F_ITEM_ID].astype(str))
        result = recommend_items(
            str(user_id), args.category, k=args.k, seen_item_ids=train_items,
        )
        rec_ids = [r["item_id"] for r in result.get("recommendations", [])]
        recalls.append(recall_at_k(rec_ids, [true_item], args.k))
        ndcgs.append(ndcg_at_k(rec_ids, [true_item], args.k))
        n += 1

    if n == 0:
        print("No eligible users.")
        return 1

    results = {
        "task": "B",
        "category": args.category,
        "n": n,
        "k": args.k,
        "recall@k": float(sum(recalls) / n),
        "ndcg@k": float(sum(ndcgs) / n),
    }
    print(json.dumps(results, indent=2))

    out = ROOT / "notebooks" / "outputs" / f"eval_task_b_{args.category.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
