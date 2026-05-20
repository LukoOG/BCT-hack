"""
Pluggable embedding backends for FAISS retrieval.

Default: TF-IDF (offline, no model download).
Architect upgrade: set EMBEDDING_BACKEND=sentence-transformers when torch/ST are available.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core import config
from app.utils.helpers import clean_text
from app.utils.logger import logger

BackendName = Literal["tfidf", "sentence-transformers"]


def active_backend_name() -> BackendName:
    name = os.getenv("EMBEDDING_BACKEND", "tfidf").strip().lower()
    if name in ("sentence-transformers", "st", "bge"):
        return "sentence-transformers"
    return "tfidf"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class EmbeddingBackend(ABC):
    name: BackendName

    @abstractmethod
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> EmbeddingBackend:
        ...


class TfidfBackend(EmbeddingBackend):
    name: BackendName = "tfidf"

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
        )

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        matrix = self._vectorizer.fit_transform(cleaned).astype(np.float32).toarray()
        return _normalize(matrix)

    def transform(self, texts: list[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        matrix = self._vectorizer.transform(cleaned).astype(np.float32).toarray()
        return _normalize(matrix)

    def save(self, path: Path) -> None:
        joblib.dump({"backend": self.name, "vectorizer": self._vectorizer}, path)

    @classmethod
    def load(cls, path: Path) -> TfidfBackend:
        payload = joblib.load(path)
        obj = cls()
        if isinstance(payload, dict):
            obj._vectorizer = payload.get("vectorizer", payload)
        else:
            obj._vectorizer = payload
        return obj


class SentenceTransformerBackend(EmbeddingBackend):
    """Dense embeddings — requires sentence-transformers + torch (Wi-Fi setup)."""

    name: BackendName = "sentence-transformers"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    def _model_handle(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        model = self._model_handle()
        vecs = model.encode(
            cleaned,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.fit_transform(texts)

    def save(self, path: Path) -> None:
        joblib.dump({"backend": self.name, "model_name": self.model_name}, path)

    @classmethod
    def load(cls, path: Path) -> SentenceTransformerBackend:
        payload = joblib.load(path)
        model_name = payload.get("model_name", config.EMBEDDING_MODEL) if isinstance(payload, dict) else config.EMBEDDING_MODEL
        return cls(model_name=model_name)


def create_backend(name: BackendName | None = None) -> EmbeddingBackend:
    chosen = name or active_backend_name()
    if chosen == "sentence-transformers":
        try:
            import sentence_transformers  # noqa: F401
            import torch  # noqa: F401
            return SentenceTransformerBackend()
        except Exception as exc:
            logger.warning(
                f"sentence-transformers unavailable ({exc}); falling back to TF-IDF"
            )
    return TfidfBackend()


def load_backend(path: Path) -> EmbeddingBackend:
    """Load saved encoder; supports legacy bare TfidfVectorizer pickles."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and payload.get("backend") == "sentence-transformers":
        return SentenceTransformerBackend.load(path)
    return TfidfBackend.load(path)
