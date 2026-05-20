"""
Run EDA on cached Parquet samples and write plots + summary JSON.

Usage (from repo root, after fetch_samples.py):
    python scripts/run_eda.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import config
from app.core import constants as C

OUT_DIR = ROOT / "notebooks" / "outputs"
SUMMARY_PATH = OUT_DIR / "eda_summary.json"


def _load_samples() -> dict[str, pd.DataFrame]:
    frames = {}
    for cat in config.AMAZON_CATEGORIES:
        path = config.DATA_RAW_DIR / f"{cat.lower()}_reviews_sample.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python scripts/fetch_samples.py"
            )
        frames[cat] = pd.read_parquet(path)
        print(f"Loaded {cat}: {len(frames[cat]):,} rows")
    return frames


def _ts_to_datetime(series: pd.Series) -> pd.Series:
    ts = series.astype("Int64")
    unit = "ms" if ts.dropna().max() > 1_000_000_000_000 else "s"
    return pd.to_datetime(ts, unit=unit, errors="coerce")


def _section_stats(df: pd.DataFrame, category: str) -> dict:
    text_col = C.F_REVIEW_TEXT if C.F_REVIEW_TEXT in df.columns else "text"
    rating_col = C.F_RATING if C.F_RATING in df.columns else "rating"
    user_col = C.F_USER_ID if C.F_USER_ID in df.columns else "user_id"
    item_col = C.F_ITEM_ID if C.F_ITEM_ID in df.columns else "parent_asin"

    df = df.copy()
    df["word_count"] = df[text_col].fillna("").astype(str).str.split().str.len()
    df["date"] = _ts_to_datetime(df[C.F_TIMESTAMP])

    per_user = df.groupby(user_col).size()
    per_item = df.groupby(item_col).size()
    needed = config.HOLDOUT_LAST_N + 1
    viable_users = int((per_user >= needed).sum())
    total_users = int(len(per_user))

    rating_counts = df[rating_col].value_counts().sort_index()
    rating_share = (rating_counts / rating_counts.sum() * 100).round(1).to_dict()

    wc = df["word_count"]
    return {
        "category": category,
        "n_reviews": int(len(df)),
        "n_users": total_users,
        "n_items": int(df[item_col].nunique()),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "word_count": {
            "p50": float(wc.quantile(0.5)),
            "p90": float(wc.quantile(0.9)),
            "p95": float(wc.quantile(0.95)),
            "p99": float(wc.quantile(0.99)),
            "max": float(wc.max()),
        },
        "rating_share_pct": {str(k): v for k, v in rating_share.items()},
        "users_with_min_reviews_for_task": {
            "needed": needed,
            "count": viable_users,
            "pct": round(viable_users / total_users * 100, 2) if total_users else 0,
        },
        "users_meeting_MIN_USER_REVIEWS": int(
            (per_user >= config.MIN_USER_REVIEWS).sum()
        ),
        "items_meeting_MIN_ITEM_REVIEWS": int(
            (per_item >= config.MIN_ITEM_REVIEWS).sum()
        ),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
    }


def _plot_all(frames: dict[str, pd.DataFrame], summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    combined = pd.concat(
        [
            f.assign(_category=cat)
            for cat, f in frames.items()
        ],
        ignore_index=True,
    )
    text_col = C.F_REVIEW_TEXT
    combined["word_count"] = (
        combined[text_col].fillna("").astype(str).str.split().str.len()
    )
    combined["date"] = _ts_to_datetime(combined[C.F_TIMESTAMP])

    # B — review length
    fig, ax = plt.subplots(figsize=(9, 4))
    for cat, df in frames.items():
        wc = df[text_col].fillna("").astype(str).str.split().str.len().clip(0, 1000)
        ax.hist(wc, bins=50, alpha=0.5, label=cat, density=True)
    ax.set_title("Review length (words, clipped 1000)")
    ax.legend()
    fig.savefig(OUT_DIR / "B_review_length.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # C — rating distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (cat, df) in zip(axes, frames.items()):
        rc = df[C.F_RATING].value_counts().sort_index()
        rc.plot.bar(ax=ax, color="#27ae60")
        ax.set_title(f"Ratings — {cat}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "C_rating_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # D — reviews per user
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (cat, df) in zip(axes, frames.items()):
        pu = df.groupby(C.F_USER_ID).size().clip(upper=50)
        pu.hist(bins=40, ax=ax, color="#8e44ad")
        ax.set_yscale("log")
        ax.set_title(f"Reviews per user — {cat}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "D_reviews_per_user.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # E — reviews per item
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (cat, df) in zip(axes, frames.items()):
        pi = df.groupby(C.F_ITEM_ID).size().clip(upper=100)
        pi.hist(bins=40, ax=ax, color="#e67e22")
        ax.set_yscale("log")
        ax.set_title(f"Reviews per item — {cat}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "E_reviews_per_item.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # F — by year
    fig, ax = plt.subplots(figsize=(10, 4))
    for cat in frames:
        sub = combined[combined["_category"] == cat]
        by_year = sub["date"].dt.year.value_counts().sort_index()
        by_year.plot(ax=ax, label=cat, alpha=0.8)
    ax.set_title("Reviews by year")
    ax.legend()
    fig.savefig(OUT_DIR / "F_reviews_by_year.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # H — side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(
        data=combined, x="word_count", hue="_category", bins=50,
        ax=axes[0], element="step", stat="density",
    )
    axes[0].set_title("Word count by category")
    sns.countplot(data=combined, x=C.F_RATING, hue="_category", ax=axes[1])
    axes[1].set_title("Rating by category")
    for cat in frames:
        sub = combined[combined["_category"] == cat]
        sub["date"].dt.year.value_counts().sort_index().plot(
            ax=axes[2], label=cat, alpha=0.8,
        )
    axes[2].set_title("Reviews by year")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "H_books_vs_electronics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved under {OUT_DIR}/")


def _build_implications(summary: dict) -> str:
    books = summary["categories"]["Books"]
    elec = summary["categories"].get("Electronics", {})
    b = books["word_count"]
    u = books["users_with_min_reviews_for_task"]

    lines = [
        "## Implications for the pipeline (auto-generated from sample EDA)",
        "",
        f"- **Embedding chunk size (§B):** 95th percentile ≈ {b['p95']:.0f} words "
        f"(99th ≈ {b['p99']:.0f}). "
        + (
            "Most reviews fit one BGE chunk; no splitting needed for MVP."
            if b["p95"] < 400
            else "Consider chunking or truncating at ~400 words for embeddings."
        ),
        "",
        f"- **Trainable users (§D, Books sample):** {u['count']:,} / {books['n_users']:,} users "
        f"({u['pct']}%) have ≥ {u['needed']} reviews for predict-last-review. "
        + (
            "**OK for the task** on this sample."
            if u["pct"] > 30
            else "**RISK:** low multi-review users — confirm on full data or relax filters."
        ),
        "",
        f"- **Rating skew (§C, Books):** "
        + ", ".join(
            f"{k}★={v}%"
            for k, v in sorted(books["rating_share_pct"].items(), key=lambda x: -x[1])[:3]
        )
        + ". Report MAE **and** per-rating accuracy / macro-F1 in eval.",
        "",
        f"- **Temporal scope (§F):** {books['date_min'][:10]} → {books['date_max'][:10]}. "
        "Consider filtering to 2015+ for demo relevance if older years dominate.",
        "",
        f"- **Books vs Electronics (§H):** Books n={books['n_reviews']:,}, "
        f"Electronics n={elec.get('n_reviews', 0):,} in sample. "
        "Compare plots in `notebooks/outputs/H_books_vs_electronics.png`.",
        "",
        "**Contract for architect:** embed per-review text; aggregate at retrieval time. "
        "`predict_next_review(user_id, category)` — see `TEAM_CONTRACT.md`.",
    ]
    return "\n".join(lines)


def main():
    frames = _load_samples()
    summary = {
        "sample_note": "Stats from streamed HF samples; not full corpus.",
        "categories": {},
    }
    for cat, df in frames.items():
        summary["categories"][cat] = _section_stats(df, cat)

    _plot_all(frames, summary)
    implications = _build_implications(summary)
    summary["implications_markdown"] = implications

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "IMPLICATIONS.md").write_text(implications, encoding="utf-8")
    print(implications)
    print(f"\nWrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
