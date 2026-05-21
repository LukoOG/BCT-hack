"""
dataset.py — Persist processed data and query it via DuckDB.

Responsibilities:
  - Write processed DataFrames to Parquet (source of truth on disk)
  - Load them into DuckDB for fast analytical queries
  - Expose clean query helpers used by profile_builder and retriever

DuckDB is used as a query engine over Parquet files — not as a persistent
database server.  This keeps the pipeline stateless and reproducible:
delete the Parquet files, re-run the pipeline, get the same result.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd

from app.core.config import (
    DATA_PROCESSED_DIR,
    DB_PATH,
    HOLDOUT_LAST_N,
)
from app.core.constants import (
    TABLE_REVIEWS, TABLE_ITEMS,
    TABLE_TRAIN, TABLE_TEST,
    F_USER_ID, F_ITEM_ID, F_DOMAIN, F_CATEGORY,
)
from app.utils.helpers import save_parquet, load_parquet, Timer
from app.utils.logger import logger


# ── File layout ────────────────────────────────────────────────────────────────
REVIEWS_PARQUET = DATA_PROCESSED_DIR / "reviews.parquet"
ITEMS_PARQUET   = DATA_PROCESSED_DIR / "items.parquet"


# ──────────────────────────────────────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────────────────────────────────────

def save_reviews(df: pd.DataFrame) -> None:
    save_parquet(df, REVIEWS_PARQUET, desc="Saving reviews")


def save_items(df: pd.DataFrame) -> None:
    # Deduplicate items across sources — keep first occurrence
    before = len(df)
    df = df.drop_duplicates(subset=[F_ITEM_ID], keep="first")
    logger.info(f"Item dedup: {before:,} → {len(df):,} unique items")
    save_parquet(df, ITEMS_PARQUET, desc="Saving items")


def save_duckdb(reviews_df: pd.DataFrame, items_df: pd.DataFrame) -> None:
    """Materialise processed tables into the configured DuckDB file."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(f"DROP TABLE IF EXISTS {TABLE_REVIEWS}")
        con.execute(f"DROP TABLE IF EXISTS {TABLE_ITEMS}")
        con.register("_reviews_df", reviews_df)
        con.register("_items_df", items_df.drop_duplicates(subset=[F_ITEM_ID], keep="first"))
        con.execute(f"CREATE TABLE {TABLE_REVIEWS} AS SELECT * FROM _reviews_df")
        con.execute(f"CREATE TABLE {TABLE_ITEMS} AS SELECT * FROM _items_df")
        logger.info(f"Saved DuckDB database: {DB_PATH}")
    finally:
        con.close()


# ──────────────────────────────────────────────────────────────────────────────
# DuckDB connection (in-memory, backed by Parquet)
# ──────────────────────────────────────────────────────────────────────────────

def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Open an in-memory DuckDB connection with Parquet views registered.

    Views available after calling this:
        reviews  — all processed reviews (train + test)
        items    — deduplicated item metadata
        train    — reviews WHERE split = 'train'
        test     — reviews WHERE split = 'test'
    """
    con = duckdb.connect(database=":memory:")

    for name, path in [
        (TABLE_REVIEWS, REVIEWS_PARQUET),
        (TABLE_ITEMS,   ITEMS_PARQUET),
    ]:
        if not Path(path).exists():
            logger.warning(f"Parquet file not found: {path}. View '{name}' will be unavailable.")
            continue
        con.execute(f"""
            CREATE OR REPLACE VIEW {name}
            AS SELECT * FROM read_parquet('{path}')
        """)

    # Convenience split views
    con.execute(f"""
        CREATE OR REPLACE VIEW {TABLE_TRAIN}
        AS SELECT * FROM {TABLE_REVIEWS} WHERE split = 'train'
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW {TABLE_TEST}
        AS SELECT * FROM {TABLE_REVIEWS} WHERE split = 'test'
    """)

    logger.debug("DuckDB connection ready (in-memory + Parquet views)")
    return con


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers  (used by profile_builder, retriever, evaluator)
# ──────────────────────────────────────────────────────────────────────────────

def get_user_reviews(
    con: duckdb.DuckDBPyConnection,
    user_id: str,
    split: str = "train",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch all reviews for a user from the specified split,
    ordered chronologically (oldest → newest).
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    return con.execute(f"""
        SELECT *
        FROM {TABLE_REVIEWS}
        WHERE user_id = ? AND split = ?
        ORDER BY timestamp ASC
        {limit_clause}
    """, [user_id, split]).df()


def get_item_metadata(
    con: duckdb.DuckDBPyConnection,
    item_id: str,
) -> Optional[dict]:
    """Return metadata dict for a single item, or None if not found."""
    rows = con.execute(f"""
        SELECT * FROM {TABLE_ITEMS} WHERE item_id = ? LIMIT 1
    """, [item_id]).df()
    return rows.iloc[0].to_dict() if len(rows) else None


def get_all_user_ids(
    con: duckdb.DuckDBPyConnection,
    split: str = "train",
    domain: Optional[str] = None,
) -> List[str]:
    """Return list of all unique user_ids in the given split (and domain)."""
    domain_filter = f"AND domain = '{domain}'" if domain else ""
    rows = con.execute(f"""
        SELECT DISTINCT user_id
        FROM {TABLE_REVIEWS}
        WHERE split = ? {domain_filter}
        ORDER BY user_id
    """, [split]).df()
    return rows["user_id"].tolist()


def get_dataset_stats(con: duckdb.DuckDBPyConnection) -> dict:
    """Return a summary dict useful for sanity checks and EDA handoff."""
    stats = {}
    for view in (TABLE_REVIEWS, TABLE_TRAIN, TABLE_TEST):
        row = con.execute(f"SELECT COUNT(*) as n FROM {view}").fetchone()
        stats[f"{view}_count"] = row[0]

    stats["unique_users"] = con.execute(
        f"SELECT COUNT(DISTINCT user_id) FROM {TABLE_REVIEWS}"
    ).fetchone()[0]

    stats["unique_items"] = con.execute(
        f"SELECT COUNT(DISTINCT item_id) FROM {TABLE_REVIEWS}"
    ).fetchone()[0]

    stats["domains"] = con.execute(
        f"SELECT domain, COUNT(*) as n FROM {TABLE_REVIEWS} GROUP BY domain"
    ).df().to_dict(orient="records")

    stats["avg_rating"] = round(
        con.execute(f"SELECT AVG(rating) FROM {TABLE_REVIEWS}").fetchone()[0], 3
    )

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline entry point (called from main.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    reviews_df: pd.DataFrame,
    items_df: pd.DataFrame,
) -> duckdb.DuckDBPyConnection:
    """
    Persist processed DataFrames and return a live DuckDB connection.

    Call this after preprocess.py has produced clean, filtered DataFrames.
    """
    with Timer("Dataset build"):
        save_reviews(reviews_df)
        save_items(items_df)
        save_duckdb(reviews_df, items_df)
        con = get_connection()

    stats = get_dataset_stats(con)
    logger.info("Dataset stats:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    return con
