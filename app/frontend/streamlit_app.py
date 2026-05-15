"""
streamlit_app.py — Demo UI for the next-review predictor.

Run with:
    streamlit run app/frontend/streamlit_app.py

This is a SCAFFOLD. The `predict_next_review` function below is the agreed
contract with the rest of the pipeline. Until embeddings + retrieval + LLM
are wired up, it returns stub data so the UI is fully clickable.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# CONTRACT — agreed signature for the prediction pipeline.
# Replace the stub body with a call into app.pipeline (or wherever your
# teammate exposes the end-to-end predict function).
# ──────────────────────────────────────────────────────────────────────────────

def predict_next_review(user_id: str, category: str) -> dict:
    """
    Returns:
        {
          "user_history": [{"item_id": str, "rating": int, "text": str}, ...],
          "prediction":   {"rating": int, "title": str, "text": str},
          "retrieved":    [{"item_id": str, "rating": int, "text": str}, ...],
        }
    """
    return {
        "user_history": [
            {"item_id": "amz_B001STUB1", "rating": 5, "text": "Loved it. Couldn't put it down."},
            {"item_id": "amz_B001STUB2", "rating": 4, "text": "Solid read, dragged in the middle."},
            {"item_id": "amz_B001STUB3", "rating": 3, "text": "Decent but forgettable."},
        ],
        "prediction": {
            "rating": 4,
            "title":  "(stub) A worthwhile read",
            "text":   "(stub) Predicted review text will appear here once the pipeline is wired up.",
        },
        "retrieved": [
            {"item_id": "amz_B001SIM1", "rating": 5, "text": "Best book I've read this year."},
            {"item_id": "amz_B001SIM2", "rating": 4, "text": "Engaging but uneven pacing."},
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Next-Review Predictor", layout="wide")
st.title("Next-Review Predictor")
st.caption("Hackathon demo — Amazon Reviews 2023")

col_input, col_output = st.columns([1, 2])

with col_input:
    st.header("Input")
    user_id = st.text_input("User ID", value="amz_A1B2C3")
    category = st.selectbox("Category", ["Books", "Electronics"])
    run = st.button("Predict next review", type="primary", use_container_width=True)

    st.divider()
    st.caption("Stub mode — predictions are placeholders until pipeline is connected.")

with col_output:
    if run or st.session_state.get("last_result"):
        if run:
            st.session_state["last_result"] = predict_next_review(user_id, category)
        result = st.session_state["last_result"]

        st.subheader("User's past reviews")
        st.dataframe(
            pd.DataFrame(result["user_history"]),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Predicted next review")
        pred = result["prediction"]
        m1, m2 = st.columns([1, 4])
        with m1:
            st.metric("Predicted rating", f"{pred['rating']} / 5")
        with m2:
            st.markdown(f"**{pred['title']}**")
            st.write(pred["text"])

        with st.expander("Retrieved similar reviews (debug)"):
            st.dataframe(
                pd.DataFrame(result["retrieved"]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Enter a user id and click **Predict next review**.")
