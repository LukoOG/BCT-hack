"""
Integration contract (Demilade <-> architect). Python module instead of markdown docs.
"""

PREDICT_NEXT_REVIEW_DOC = """
def predict_next_review(user_id: str, category: str) -> dict:
    Returns:
        user_history: list[{item_id, rating, text}]
        prediction: {rating, title, text}
        retrieved: list[{item_id, rating, text}]
"""

EVAL_HOLDOUT_COLUMNS = ("user_id", "item_id", "true_rating", "true_text")
EVAL_PREDICTION_KEYS = ("user_id", "item_id", "pred_rating", "pred_text", "retrieved_item_ids")

MVP_SCOPE = ("Amazon Books", "predict last review", "MAE + ROUGE + Streamlit demo")
