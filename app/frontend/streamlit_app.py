"""
BCT Hack demo — Demilade track: Predict | EDA | Eval | Prompts

Run from repo root:
    streamlit run app/frontend/streamlit_app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core import config
from app.core import constants as C
from app.data.sample_store import (
    default_demo_user,
    has_sample,
    load_demo_users,
    load_sample,
)
from app.pipeline.stub import predict_next_review
from app.prompts.templates import render_user_prompt, system_prompt_for_category

OUTPUTS = ROOT / "notebooks" / "outputs"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _sidebar():
    with st.sidebar:
        st.header("Setup")
        st.code(str(ROOT.name), language=None)
        data_ok = has_sample("Books")
        st.metric("Books sample", "OK" if data_ok else "Missing")
        st.metric("Electronics", "OK" if has_sample("Electronics") else "Missing")
        import os
        st.metric("LLM", "ON" if os.environ.get("ANTHROPIC_API_KEY") else "stub only")
        eval_path = OUTPUTS / "eval_books.json"
        if eval_path.exists():
            ev = json.loads(eval_path.read_text(encoding="utf-8"))
            st.caption("Last eval (Books stub)")
            st.write(f"MAE {ev['rating']['mae']:.2f} | ROUGE {ev['text']['rougeL_f']:.2f}")
        st.markdown("[Demo script](DEMO_SCRIPT.md)")
        st.markdown("[Tasks](DEMILADE_TASKS.md)")


def _run_setup_hint():
    st.warning("No sample data found.")
    st.code("python scripts/setup_demilade.py", language="bash")
    if st.button("Run fetch + EDA (may take a few minutes)"):
        with st.spinner("Downloading sample & running EDA..."):
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "setup_demilade.py")],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
        st.text(r.stdout or "")
        if r.returncode != 0:
            st.error(r.stderr or "Setup failed")
        else:
            st.success("Done. Refresh the page.")
            st.rerun()


def tab_predict():
    st.subheader("Predict next review")
    if not has_sample("Books"):
        _run_setup_hint()
        return

    category = st.selectbox("Category", config.AMAZON_CATEGORIES, key="pred_cat")
    if not has_sample(category):
        st.info(f"No sample for {category}. Run fetch_samples with --categories {category}")
        return

    users = load_demo_users(category, limit=40)
    default_u = default_demo_user(category) or (users[0] if users else "")
    user_id = st.selectbox("User (2+ reviews in sample)", users, index=0 if users else None) if users else st.text_input(
        "User ID", value=default_u or ""
    )

    if st.button("Predict", type="primary"):
        result = predict_next_review(str(user_id), category)
        st.session_state["pred_result"] = result

    if st.session_state.get("pred_result"):
        result = st.session_state["pred_result"]
        mode = result.get("meta", {}).get("mode", "stub")
        st.caption(f"Mode: **{mode}** (set ANTHROPIC_API_KEY for LLM)")

        st.markdown("#### Past reviews")
        st.dataframe(pd.DataFrame(result["user_history"]), use_container_width=True, hide_index=True)

        pred = result["prediction"]
        c1, c2 = st.columns([1, 4])
        with c1:
            st.metric("Rating", f"{pred['rating']} / 5")
        with c2:
            st.markdown(f"**{pred.get('title', '')}**")
            st.write(pred["text"])

        with st.expander("Retrieved similar reviews"):
            st.dataframe(pd.DataFrame(result["retrieved"]), use_container_width=True, hide_index=True)


def tab_eda():
    st.subheader("EDA outputs")
    summary = OUTPUTS / "eda_summary.json"
    implications = OUTPUTS / "IMPLICATIONS.md"

    if implications.exists():
        st.markdown(implications.read_text(encoding="utf-8"))
    else:
        st.info("Run: `python scripts/run_eda.py`")

    if summary.exists():
        with st.expander("Raw stats (JSON)"):
            st.json(json.loads(summary.read_text(encoding="utf-8")))

    plots = sorted(OUTPUTS.glob("*.png")) if OUTPUTS.exists() else []
    if plots:
        cols = st.columns(2)
        for i, p in enumerate(plots):
            cols[i % 2].image(str(p), caption=p.name, use_container_width=True)
    else:
        st.warning("No plots in notebooks/outputs/")


def tab_eval():
    st.subheader("Evaluation on sample holdout")
    st.caption("Scores stub/heuristic predictions on users with 2+ reviews in the cached sample.")

    category = st.selectbox("Category", config.AMAZON_CATEGORIES, key="eval_cat")
    max_users = st.slider("Max users", 5, 50, 20)

    if st.button("Run eval"):
        with st.spinner("Running..."):
            r = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_eval_on_sample.py"),
                    "--category", category,
                    "--max-users", str(max_users),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
        st.code(r.stdout or "(no output)")
        if r.returncode != 0:
            st.error(r.stderr)
        else:
            eval_path = OUTPUTS / f"eval_{category.lower()}.json"
            if eval_path.exists():
                st.json(json.loads(eval_path.read_text(encoding="utf-8")))


def tab_prompts():
    st.subheader("Prompt preview")
    if not has_sample("Books"):
        _run_setup_hint()
        return

    category = st.selectbox("Category", config.AMAZON_CATEGORIES, key="prompt_cat")
    users = load_demo_users(category, limit=20)
    user_id = st.selectbox("User", users, key="prompt_user") if users else st.text_input("User ID")

    if st.button("Build prompt"):
        result = predict_next_review(str(user_id), category)
        item_meta = {
            "title": result["prediction"].get("title", "Product"),
            "category": category,
            "description": result["prediction"].get("text", "")[:300],
        }
        user_prompt = render_user_prompt(
            result["user_history"], item_meta, result["retrieved"]
        )
        st.markdown("#### System")
        st.text(system_prompt_for_category(category))
        st.markdown("#### User")
        st.text(user_prompt)
        st.caption("Full generation: set ANTHROPIC_API_KEY and use app.prompts.generate")


def tab_data():
    st.subheader("Sample data status")
    for cat in config.AMAZON_CATEGORIES:
        ok = has_sample(cat)
        st.write(f"**{cat}:**", "ready" if ok else "missing")
        if ok:
            df = load_sample(cat)
            st.write(f"  {len(df):,} reviews, {df[C.F_USER_ID].nunique():,} users")


st.set_page_config(page_title="BCT Review Predictor", layout="wide", page_icon="📝")
_sidebar()
st.title("BCT Hack — Review Predictor")
st.caption("Demilade: EDA · Eval · Prompts · Demo")

tabs = st.tabs(["Predict", "EDA", "Eval", "Prompts", "Data"])
with tabs[0]:
    tab_predict()
with tabs[1]:
    tab_eda()
with tabs[2]:
    tab_eval()
with tabs[3]:
    tab_prompts()
with tabs[4]:
    tab_data()
