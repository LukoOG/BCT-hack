"""
Central configuration for the pipeline.
All tuneable constants live here — import this everywhere.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_EMBEDDINGS_DIR= ROOT_DIR / "data" / "embeddings"
DB_PATH            = DATA_PROCESSED_DIR / "reviews.duckdb"

# Ensure directories exist at import time
for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_EMBEDDINGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ── Dataset sources ────────────────────────────────────────────────────────────
# Amazon Reviews 2023  (UCSD McAuley Lab)
AMAZON_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023"

# Categories to download — start small, scale later
AMAZON_CATEGORIES: List[str] = [
    "Books",
    "Electronics",
]

# Goodreads (UCSD Mengting Wan)
GOODREADS_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads"
GOODREADS_FILES = {
    "reviews": "goodreads_reviews_dedup.json.gz",
    "books":   "goodreads_books.json.gz",
}


# ── Filtering thresholds ───────────────────────────────────────────────────────
MIN_USER_REVIEWS  = 5    # drop users with fewer reviews than this
MIN_ITEM_REVIEWS  = 10   # drop items with fewer reviews than this
MIN_REVIEW_CHARS  = 20   # drop reviews shorter than this (noise)
MAX_REVIEW_CHARS  = 4000 # truncate very long reviews


# ── Splits ─────────────────────────────────────────────────────────────────────
# For each user, the last N interactions are held out for evaluation
HOLDOUT_LAST_N = 1   # Task A: predict the last review written


# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"   # primary
EMBEDDING_FALLBACK= "all-MiniLM-L6-v2"         # if memory constrained
EMBEDDING_BATCH_SIZE = 256
FAISS_INDEX_TYPE  = "Flat"   # "Flat" (exact) | "IVFFlat" (approx, faster at scale)


# ── User profile ───────────────────────────────────────────────────────────────
PROFILE_MAX_SAMPLE_REVIEWS = 5   # how many past reviews to store as few-shot examples
PROFILE_MAX_TOP_CATEGORIES = 3


# ── LLM ────────────────────────────────────────────────────────────────────────
LLM_MODEL       = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS  = 1024
LLM_TEMPERATURE = 0.7


# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = ROOT_DIR / "pipeline.log"