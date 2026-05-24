#!/usr/bin/env python3
"""
startup.py — Container entrypoint for Railway / cloud deployments.

Builds FAISS indexes directly from processed parquet files if they
don't exist, then starts the requested service.

Usage (Railway start command):
    API:       python startup.py api
    Streamlit: python startup.py demo
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))   # ensure app.* imports work


# ── Data discovery ─────────────────────────────────────────────────────────────

def _find_data_root():
    for candidate in [ROOT / "data", ROOT / "deployment_data"]:
        if (candidate / "processed" / "reviews.parquet").exists():
            return candidate
    return None


def _faiss_exists(data_root: Path) -> bool:
    embeddings_dir = data_root / "embeddings"
    if not embeddings_dir.exists():
        return False
    return len(list(embeddings_dir.glob("*.faiss"))) > 0


# ── FAISS rebuild directly from processed parquet ─────────────────────────────
# Bypasses build_embeddings.py which requires raw HuggingFace sample files.
# Reads reviews.parquet directly — exactly what we have in deployment_data/.

def _build_faiss(data_root: Path) -> None:
    import pickle
    import numpy as np

    print("[startup] Loading reviews from parquet ...", flush=True)
    import pandas as pd
    reviews = pd.read_parquet(data_root / "processed" / "reviews.parquet")
    print(f"[startup] Loaded {len(reviews):,} reviews", flush=True)

    embeddings_dir = data_root / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Build one FAISS index per category (matches what retriever.py expects)
    try:
        import faiss
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        print(f"[startup] ERROR: Missing dependency: {e}", flush=True)
        print("[startup] Ensure faiss-cpu and scikit-learn are in requirements.", flush=True)
        sys.exit(1)

    categories = reviews["category"].unique()
    print(f"[startup] Building FAISS indexes for {len(categories)} categories ...", flush=True)

    for cat in categories:
        cat_reviews = reviews[reviews["category"] == cat].copy()
        if len(cat_reviews) < 2:
            print(f"[startup]   Skipping {cat} (< 2 reviews)", flush=True)
            continue

        texts = cat_reviews["review_text"].fillna("").tolist()

        # Fit TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        vectors = vectorizer.fit_transform(texts).toarray().astype("float32")

        # Normalise for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        # Build flat FAISS index
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        # Save index and vectorizer using category slug as filename
        slug = cat.lower().replace(" ", "_").replace("&", "and")
        faiss.write_index(index, str(embeddings_dir / f"{slug}.faiss"))
        with open(embeddings_dir / f"{slug}.pkl", "wb") as f:
            pickle.dump(
                {"vectorizer": vectorizer, "review_ids": cat_reviews["user_id"].tolist()},
                f
            )

        print(f"[startup]   {cat}: {len(texts)} reviews, dim={dim}", flush=True)

    print(f"[startup] FAISS build complete → {embeddings_dir}", flush=True)


# ── Service launchers ──────────────────────────────────────────────────────────

def _start_api():
    import subprocess
    subprocess.run([
        sys.executable, "-m", "uvicorn", "app.api.server:app",
        "--host", "0.0.0.0", "--port", "8000", "--workers", "1",
    ], check=True)


def _start_demo():
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/frontend/streamlit_app.py",
        "--server.port", "8080",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ], check=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("api", "demo"):
        print("Usage: python startup.py api|demo")
        sys.exit(1)

    mode = sys.argv[1]

    # ── Find data ──────────────────────────────────────────────────────────────
    print("[startup] Checking data ...", flush=True)
    data_root = _find_data_root()

    if data_root is None:
        print("[startup] ERROR: No processed data found.", flush=True)
        print("[startup]   Expected: deployment_data/processed/reviews.parquet", flush=True)
        sys.exit(1)

    print(f"[startup] Found data at: {data_root}", flush=True)

    # Symlink deployment_data/ → data/ so all existing app code paths work
    # unchanged (they all reference "data/...")
    local_data = ROOT / "data"
    if not local_data.exists():
        print(f"[startup] Symlinking {data_root.name}/ -> data/", flush=True)
        local_data.symlink_to(data_root)

    # ── Build FAISS if missing ─────────────────────────────────────────────────
    if not _faiss_exists(data_root):
        print("[startup] FAISS indexes not found — building from processed parquet ...", flush=True)
        _build_faiss(data_root)
    else:
        print("[startup] FAISS indexes present — skipping build.", flush=True)

    # ── Start service ──────────────────────────────────────────────────────────
    print(f"[startup] Starting: {mode}", flush=True)
    if mode == "api":
        _start_api()
    else:
        _start_demo()


if __name__ == "__main__":
    main()
