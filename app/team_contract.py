"""
Integration contract (Demilade <-> architect). Python module instead of markdown docs.
"""

PREDICT_NEXT_REVIEW_DOC = """
def predict_next_review(
    user_id: str,
    category: str = "Books",
    *,
    target_item_id: str | None = None,
) -> dict:
    Returns:
        user_history: list[{item_id, rating, text}]
        prediction: {rating, title, text}
        retrieved: list[{item_id, rating, text}]
    Note: pass target_item_id during eval to avoid holdout text leakage.
"""

EVAL_HOLDOUT_COLUMNS = ("user_id", "item_id", "true_rating", "true_text")
EVAL_PREDICTION_KEYS = ("user_id", "item_id", "pred_rating", "pred_text", "retrieved_item_ids")

MVP_SCOPE = ("Amazon Books", "predict last review", "MAE + ROUGE + Streamlit demo")

# ── Architect extension points (on main as of FAISS MVP) ─────────────────────

ARCHITECT_HOOKS = {
    "predict_entry": "app.pipeline.predict_next_review",
    "retriever": "app.pipeline.retriever.ReviewRetriever",
    "embeddings": "app.pipeline.embeddings.create_backend",
    "api": "app.api.server:app",
    "build_indexes": "scripts/build_embeddings.py",
}

ARCHITECT_UPGRADE_STEPS = """
1. Install torch + sentence-transformers (Wi-Fi): pip install torch sentence-transformers
2. Set EMBEDDING_BACKEND=sentence-transformers
3. Rebuild indexes locally: python scripts/build_embeddings.py --force
4. Optional: swap heuristic in app/pipeline/stub.py for your own ranker/LLM chain
5. Extend FastAPI in app/api/server.py (batch predict, admin rebuild, etc.)
"""
