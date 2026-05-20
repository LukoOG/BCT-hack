"""
TF-IDF or dense embeddings + FAISS index for review retrieval.

Indexes are built from local parquet only — no streaming downloads at index time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from app.core import config
from app.core import constants as C
from app.pipeline.embeddings import EmbeddingBackend, active_backend_name, create_backend, load_backend
from app.utils.helpers import clean_text
from app.utils.logger import logger


def _index_paths(category: str) -> tuple[Path, Path, Path]:
    key = category.lower()
    base = config.DATA_EMBEDDINGS_DIR
    encoder = base / f"{key}_encoder.joblib"
    legacy = base / f"{key}_vectorizer.joblib"
    return (
        base / f"{key}.faiss",
        encoder if encoder.exists() or not legacy.exists() else legacy,
        base / f"{key}_meta.parquet",
    )


class ReviewRetriever:
    """Per-category embedding + FAISS retriever backed by local files."""

    def __init__(self, category: str, backend: EmbeddingBackend | None = None):
        self.category = category
        self.backend = backend or create_backend()
        self.index: faiss.Index | None = None
        self.meta: pd.DataFrame = pd.DataFrame()

    @property
    def is_ready(self) -> bool:
        faiss_path, enc_path, meta_path = _index_paths(self.category)
        return faiss_path.exists() and enc_path.exists() and meta_path.exists()

    def build(self, reviews: pd.DataFrame) -> None:
        text_col = C.F_REVIEW_TEXT if C.F_REVIEW_TEXT in reviews.columns else "text"
        texts = reviews[text_col].fillna("").astype(str).map(clean_text).tolist()
        if not texts:
            raise ValueError(f"No review text to index for {self.category}")

        matrix = self.backend.fit_transform(texts)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

        title_col = "title" if "title" in reviews.columns else None
        self.meta = pd.DataFrame({
            C.F_USER_ID: reviews[C.F_USER_ID].astype(str).values,
            C.F_ITEM_ID: reviews[C.F_ITEM_ID].astype(str).values,
            C.F_RATING: reviews[C.F_RATING].values,
            C.F_REVIEW_TEXT: texts,
            "title": reviews[title_col].fillna("").astype(str).values if title_col else "",
        })

        self.save()
        logger.info(
            f"Built FAISS index for {self.category}: {len(self.meta):,} vectors "
            f"({self.backend.name})"
        )

    def save(self) -> None:
        if self.index is None:
            return
        faiss_path, enc_path, meta_path = _index_paths(self.category)
        config.DATA_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(faiss_path))
        enc_out = config.DATA_EMBEDDINGS_DIR / f"{self.category.lower()}_encoder.joblib"
        self.backend.save(enc_out)
        self.meta.to_parquet(meta_path, index=False)
        manifest = config.DATA_EMBEDDINGS_DIR / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        data[self.category] = {
            "vectors": len(self.meta),
            "faiss": faiss_path.name,
            "backend": self.backend.name,
        }
        manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        faiss_path, enc_path, meta_path = _index_paths(self.category)
        self.index = faiss.read_index(str(faiss_path))
        self.backend = load_backend(enc_path)
        self.meta = pd.read_parquet(meta_path)

    def _ensure_loaded(self) -> None:
        if self.index is None:
            if not self.is_ready:
                raise FileNotFoundError(
                    f"No index for {self.category}. Run: python scripts/build_embeddings.py"
                )
            self.load()

    def embed_query(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        return self.backend.transform([text])

    def search(
        self,
        query_text: str,
        k: int = 5,
        exclude_user: Optional[str] = None,
        prefer_item_id: Optional[str] = None,
    ) -> list[dict]:
        self._ensure_loaded()
        if not query_text.strip():
            return []
        n = int(self.index.ntotal)
        if n == 0:
            return []

        query = self.embed_query(query_text)
        fetch = min(max(k * 8, k), n)
        scores, indices = self.index.search(query, fetch)

        results: list[dict] = []
        same_item: list[dict] = []

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            row = self.meta.iloc[int(idx)]
            uid = str(row[C.F_USER_ID])
            if exclude_user and uid == exclude_user:
                continue

            item = {
                "item_id": str(row[C.F_ITEM_ID]),
                "rating": int(float(row[C.F_RATING])),
                "text": str(row[C.F_REVIEW_TEXT])[:500],
                "title": str(row.get("title", "") or "")[:120],
                "score": float(score),
            }
            if prefer_item_id and item["item_id"] == prefer_item_id:
                same_item.append(item)
            else:
                results.append(item)

        merged = same_item + results
        return merged[:k]


def load_train_reviews(category: Optional[str] = None) -> pd.DataFrame:
    """Load review corpus for indexing — prefer cached raw samples (no download)."""
    if category:
        from app.data.sample_store import has_sample, load_sample
        if has_sample(category):
            return load_sample(category)

    proc = config.DATA_PROCESSED_DIR / "reviews.parquet"
    if proc.exists():
        df = pd.read_parquet(proc)
        if C.F_SPLIT in df.columns:
            df = df[df[C.F_SPLIT] == "train"]
        if category and C.F_CATEGORY in df.columns:
            df = df[df[C.F_CATEGORY] == category]
        if len(df):
            return df.reset_index(drop=True)

    from app.data.from_samples import load_all_review_samples
    return load_all_review_samples()


def get_retriever(category: str, auto_build: bool = True) -> ReviewRetriever:
    retriever = ReviewRetriever(category)
    if retriever.is_ready:
        retriever.load()
        return retriever
    if not auto_build:
        return retriever

    train = load_train_reviews(category)
    if train.empty:
        raise FileNotFoundError(f"No local train reviews for {category}")
    retriever.build(train)
    return retriever
