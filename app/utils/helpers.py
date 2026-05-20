"""
Shared utilities used across the pipeline.
"""

import gzip
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Generator, Iterable

import pandas as pd
from tqdm import tqdm

from app.core.config import MAX_REVIEW_CHARS
from app.core.constants import PRICE_TIER_BINS, PRICE_TIER_LABELS
from app.utils.logger import logger


# ── Text cleaning ──────────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE   = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace, and truncate to MAX_REVIEW_CHARS."""
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:MAX_REVIEW_CHARS]


# ── Rating helpers ─────────────────────────────────────────────────────────────

def clamp_rating(rating: Any) -> float:
    """Coerce a value to a float in [1.0, 5.0]."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return None
    return max(1.0, min(5.0, r))


def bucket_price(price: Any) -> str:
    """Map a raw price value to a human-readable tier label."""
    try:
        p = float(str(price).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return "unknown"
    for i, (lo, hi) in enumerate(zip(PRICE_TIER_BINS, PRICE_TIER_BINS[1:])):
        if lo <= p < hi:
            return PRICE_TIER_LABELS[i]
    return "unknown"


# ── ID normalisation ───────────────────────────────────────────────────────────

def stable_id(*parts: str) -> str:
    """
    Create a stable, short ID by hashing multiple string parts together.
    Used when source IDs might collide across domains.
    """
    combined = "|".join(str(p) for p in parts)
    return hashlib.md5(combined.encode()).hexdigest()[:16]


# ── File I/O ──────────────────────────────────────────────────────────────────

def iter_jsonl_gz(path: Path, desc: str = "") -> Generator[dict, None, None]:
    """
    Lazily iterate over a gzipped JSON-lines file.
    Each line is a JSON object (dict). Yields one dict at a time —
    never loads the whole file into memory.
    """
    path = Path(path)
    total = path.stat().st_size
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc or path.name) as bar:
            for line in fh:
                bar.update(len(line.encode("utf-8")))
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line in {path.name}")


def save_parquet(df: pd.DataFrame, path: Path, desc: str = "") -> None:
    """Save a DataFrame to Parquet with logging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    size_mb = path.stat().st_size / 1_048_576
    logger.info(f"{'Saved' if not desc else desc} -> {path.name} ({len(df):,} rows, {size_mb:.1f} MB)")


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame with logging."""
    path = Path(path)
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded {path.name} -> {len(df):,} rows, {df.shape[1]} cols")
    return df


# ── Timing ─────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        logger.info(f"{self.label} completed in {elapsed:.2f}s")


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunked(iterable: Iterable, size: int) -> Generator[list, None, None]:
    """Yield successive fixed-size chunks from an iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk