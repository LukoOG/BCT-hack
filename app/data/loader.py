"""
loader.py — Download and locate raw dataset files.

Responsibilities:
  - Download Amazon Reviews 2023 (reviews + metadata) by category
  - Download Goodreads reviews + books files
  - Verify file integrity (size check)
  - Never re-download files that already exist

This module does NOT parse or transform data — that's preprocess.py.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx
from tqdm import tqdm

from app.core.config import (
    AMAZON_BASE_URL,
    AMAZON_CATEGORIES,
    DATA_RAW_DIR,
    GOODREADS_BASE_URL,
    GOODREADS_FILES,
)
from app.utils.logger import logger


# ── Download helpers ───────────────────────────────────────────────────────────

# Chunk size for streaming writes — 8 MB balances memory use vs syscall overhead
_CHUNK_SIZE = 8 * 1024 * 1024


def _download(url: str, dest: Path, timeout: int = 300) -> Path:
    """
    Stream *url* to *dest* using httpx.

    - Skips if the file already exists and is non-empty.
    - Streams in 8 MB chunks — never loads the full file into memory.
    - Writes to a `.part` temp file and renames on success, so a killed
      download never leaves a corrupt file behind.
    - Follows redirects automatically.
    - Returns the local path.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Already exists, skipping download: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    logger.info(f"Downloading {url}")
    logger.info(f"  → {dest}")

    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30, read=timeout, write=timeout, pool=30),
    ) as response:
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0)) or None

        with part.open("wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=dest.name,
        ) as bar:
            for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                fh.write(chunk)
                bar.update(len(chunk))

    part.rename(dest)   # atomic rename — only happens on clean completion

    size_mb = dest.stat().st_size / 1_048_576
    logger.success(f"Download complete: {dest.name} ({size_mb:.1f} MB)")
    return dest


# ── Amazon Reviews 2023 ────────────────────────────────────────────────────────
#
# File layout on the UCSD server (2023 release, per HF dataset card):
#   Reviews : {BASE}/raw/review_categories/{Category}.jsonl.gz
#   Metadata: {BASE}/raw/meta_categories/meta_{Category}.jsonl.gz
#
# Each review line schema (relevant fields):
#   user_id, asin (item_id), rating, text, timestamp, verified_purchase
#
# Each metadata line schema (relevant fields):
#   asin, title, description, price, average_rating, categories

def amazon_review_url(category: str) -> str:
    return f"{AMAZON_BASE_URL}/{category}.jsonl.gz"


def amazon_meta_url(category: str) -> str:
    return f"{AMAZON_BASE_URL}/{category}.jsonl.gz"


def download_amazon(
    categories: Optional[list] = None,
    dest_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Download Amazon review + metadata files for each category.

    Returns a mapping:
        { category: { "reviews": Path, "meta": Path } }
    """
    categories = categories or AMAZON_CATEGORIES
    dest_dir   = Path(dest_dir or DATA_RAW_DIR) / "amazon"

    paths: Dict[str, Dict[str, Path]] = {}
    for cat in categories:
        logger.info(f"[Amazon] Fetching category: {cat}")
        paths[cat] = {
            "reviews": _download(amazon_review_url(cat), dest_dir / f"review_{cat}.jsonl.gz"),
            "meta":    _download(amazon_meta_url(cat),   dest_dir / f"meta_{cat}.jsonl.gz"),
        }

    return paths


# ── Goodreads ──────────────────────────────────────────────────────────────────
#
# File layout:
#   Reviews: {BASE}/goodreads_reviews_dedup.json.gz
#   Books  : {BASE}/goodreads_books.json.gz
#
# Each review line schema (relevant fields):
#   user_id, book_id, rating, review_text, date_added
#
# Each book line schema (relevant fields):
#   book_id, title, description, average_rating, popular_shelves, isbn

def download_goodreads(dest_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Download Goodreads reviews and books files.

    Returns:
        { "reviews": Path, "books": Path }
    """
    dest_dir = Path(dest_dir or DATA_RAW_DIR) / "goodreads"
    paths: Dict[str, Path] = {}

    for key, filename in GOODREADS_FILES.items():
        url = f"{GOODREADS_BASE_URL}/{filename}"
        paths[key] = _download(url, dest_dir / filename)

    return paths


# ── Convenience: download everything ──────────────────────────────────────────

def download_all(
    amazon_categories: Optional[list] = None,
) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, Path]]:
    """
    Download all configured datasets.

    Returns:
        (amazon_paths, goodreads_paths)
    """
    logger.info("=== Starting dataset downloads ===")
    amazon    = download_amazon(amazon_categories)
    goodreads = download_goodreads()
    logger.success("=== All downloads complete ===")
    return amazon, goodreads


# ── Path discovery (for already-downloaded files) ─────────────────────────────

def find_amazon_files(
    categories: Optional[list] = None,
    raw_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Return paths for already-downloaded Amazon files without downloading.
    Raises FileNotFoundError if any file is missing.
    """
    categories = categories or AMAZON_CATEGORIES
    raw_dir    = Path(raw_dir or DATA_RAW_DIR) / "amazon"
    paths: Dict[str, Dict[str, Path]] = {}

    for cat in categories:
        review_path = raw_dir / f"review_{cat}.jsonl.gz"
        meta_path   = raw_dir / f"meta_{cat}.jsonl.gz"
        for p in (review_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing raw file: {p}\n"
                    f"Run download_amazon(['{cat}']) first."
                )
        paths[cat] = {"reviews": review_path, "meta": meta_path}

    return paths


def find_goodreads_files(raw_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Return paths for already-downloaded Goodreads files."""
    raw_dir = Path(raw_dir or DATA_RAW_DIR) / "goodreads"
    paths: Dict[str, Path] = {}

    for key, filename in GOODREADS_FILES.items():
        p = raw_dir / filename
        if not p.exists():
            raise FileNotFoundError(
                f"Missing raw file: {p}\n"
                "Run download_goodreads() first."
            )
        paths[key] = p

    return paths