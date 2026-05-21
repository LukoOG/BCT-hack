"""
Evaluate Task A (review simulation) and Task B (recommendation ranking).

Examples:
    python scripts/evaluate_tasks.py --task all --category Books --max-users 40
    python scripts/evaluate_tasks.py --task B --category Electronics --k 10
    python scripts/evaluate_tasks.py --source data/processed/reviews.duckdb --task A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import config
from app.core import constants as C
from app.eval.metrics import ndcg_at_k, recall_at_k
from app.eval.runner import evaluate_predictions, format_results_markdown
from app.pipeline import predict_next_review, recommend_items


def _read_duckdb(path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = con.execute("SHOW TABLES").df()["name"].astype(str).tolist()
        if C.TABLE_REVIEWS in tables:
            return con.execute(f"SELECT * FROM {C.TABLE_REVIEWS}").df()
        if "reviews" in tables:
            return con.execute("SELECT * FROM reviews").df()
        raise FileNotFoundError(f"No reviews table found in {path}. Tables: {tables}")
    finally:
        con.close()


def _load_reviews(source: str | None) -> pd.DataFrame:
    if source:
        path = Path(source)
        if not path.is_absolute():
            path = ROOT / path
        if path.suffix == ".duckdb" or path.suffix == ".db":
            return _read_duckdb(path)
        return pd.read_parquet(path)

    parquet = config.DATA_PROCESSED_DIR / "reviews.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if config.DB_PATH.exists():
        return _read_duckdb(config.DB_PATH)

    from app.data.from_samples import load_all_review_samples, prepare_reviews_df

    return prepare_reviews_df(load_all_review_samples())


def _normalise_reviews(df: pd.DataFrame, category: str) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    if "asin" in df.columns and C.F_ITEM_ID not in df.columns:
        rename["asin"] = C.F_ITEM_ID
    if "parent_asin" in df.columns and C.F_ITEM_ID not in df.columns:
        rename["parent_asin"] = C.F_ITEM_ID
    if "overall" in df.columns and C.F_RATING not in df.columns:
        rename["overall"] = C.F_RATING
    if "text" in df.columns and C.F_REVIEW_TEXT not in df.columns:
        rename["text"] = C.F_REVIEW_TEXT
    df = df.rename(columns=rename)

    required = [C.F_USER_ID, C.F_ITEM_ID, C.F_RATING, C.F_REVIEW_TEXT]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Review data is missing required columns: {missing}")

    if C.F_CATEGORY not in df.columns:
        df[C.F_CATEGORY] = category
    if C.F_TIMESTAMP not in df.columns:
        df[C.F_TIMESTAMP] = 0

    df = df[df[C.F_CATEGORY].astype(str) == category].copy()
    for col in (C.F_USER_ID, C.F_ITEM_ID):
        df[col] = df[col].astype(str)
    df[C.F_RATING] = df[C.F_RATING].astype(float)
    df[C.F_REVIEW_TEXT] = df[C.F_REVIEW_TEXT].fillna("").astype(str)
    return df.sort_values([C.F_USER_ID, C.F_TIMESTAMP]).reset_index(drop=True)


def build_holdout(df: pd.DataFrame, max_users: int) -> pd.DataFrame:
    rows: list[dict] = []
    for user_id, grp in df.groupby(C.F_USER_ID, sort=False):
        if len(grp) < config.HOLDOUT_LAST_N + 1:
            continue
        test = grp[grp.get(C.F_SPLIT, "") == "test"] if C.F_SPLIT in grp.columns else grp.tail(1)
        if test.empty:
            test = grp.tail(1)
        row = test.iloc[-1]
        rows.append({
            "user_id": str(user_id),
            "item_id": str(row[C.F_ITEM_ID]),
            "true_rating": float(row[C.F_RATING]),
            "true_text": str(row[C.F_REVIEW_TEXT]),
        })
        if len(rows) >= max_users:
            break
    return pd.DataFrame(rows)


def evaluate_task_a(holdout: pd.DataFrame, category: str) -> dict:
    predictions: list[dict] = []
    for row in holdout.itertuples():
        pred = predict_next_review(str(row.user_id), category, target_item_id=str(row.item_id))
        predictions.append({
            "user_id": str(row.user_id),
            "item_id": str(row.item_id),
            "pred_rating": float(pred["prediction"]["rating"]),
            "pred_text": str(pred["prediction"]["text"]),
            "retrieved_item_ids": [r["item_id"] for r in pred.get("retrieved", [])],
        })
    return evaluate_predictions(holdout, predictions)


def _negative_candidates(df: pd.DataFrame, true_item: str, user_id: str, n: int) -> list[str]:
    pool = df[
        (df[C.F_ITEM_ID].astype(str) != true_item)
        & (df[C.F_USER_ID].astype(str) != user_id)
    ]
    popular = (
        pool.groupby(C.F_ITEM_ID)
        .size()
        .sort_values(ascending=False)
        .head(n)
        .index.astype(str)
        .tolist()
    )
    return popular


def evaluate_task_b(holdout: pd.DataFrame, df: pd.DataFrame, category: str, k: int, negatives: int) -> dict:
    recalls, ndcgs = [], []
    examples = []
    for row in holdout.itertuples():
        candidates = [str(row.item_id)] + _negative_candidates(df, str(row.item_id), str(row.user_id), negatives)
        rec = recommend_items(str(row.user_id), category, k=k, candidate_item_ids=candidates)
        item_ids = [r["item_id"] for r in rec.get("recommendations", [])]
        recalls.append(recall_at_k(item_ids, [str(row.item_id)], k))
        ndcgs.append(ndcg_at_k(item_ids, [str(row.item_id)], k))
        examples.append({
            "user_id": str(row.user_id),
            "true_item_id": str(row.item_id),
            "recommended_item_ids": item_ids,
        })
    return {
        "n": int(len(holdout)),
        "recommendation": {
            f"recall@{k}": float(sum(recalls) / len(recalls)) if recalls else 0.0,
            f"ndcg@{k}": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        },
        "examples": examples[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["A", "B", "all"], default="all")
    parser.add_argument("--category", default="Books")
    parser.add_argument("--source", default=None, help="Optional reviews parquet/duckdb path")
    parser.add_argument("--max-users", type=int, default=25)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        df = _normalise_reviews(_load_reviews(args.source), args.category)
    except FileNotFoundError as exc:
        print(str(exc))
        print(
            "Expected one of: data/processed/reviews.parquet, "
            "data/processed/reviews.duckdb, or data/raw/*_reviews_sample.parquet."
        )
        return 1
    if df.empty:
        print(f"No rows found for category {args.category!r}.")
        return 1

    holdout = build_holdout(df, args.max_users)
    if holdout.empty:
        print("No users with enough reviews for holdout evaluation.")
        return 1

    results: dict = {
        "category": args.category,
        "holdout_users": int(len(holdout)),
        "source_rows": int(len(df)),
    }
    if args.task in ("A", "all"):
        results["task_a"] = evaluate_task_a(holdout, args.category)
    if args.task in ("B", "all"):
        results["task_b"] = evaluate_task_b(holdout, df, args.category, args.k, args.negatives)

    print(json.dumps(results, indent=2))
    if "task_a" in results:
        print("\nTask A")
        print(format_results_markdown(results["task_a"]))
    if "task_b" in results:
        rec = results["task_b"]["recommendation"]
        print("\nTask B")
        for name, value in rec.items():
            print(f"{name}: {value:.4f}")

    out = Path(args.output) if args.output else ROOT / "notebooks" / "outputs" / f"eval_{args.category.lower()}_{args.task.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
