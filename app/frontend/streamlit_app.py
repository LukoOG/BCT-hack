"""
BCT Hack demo — Review Predictor UI.

Run from repo root:
    streamlit run app/frontend/streamlit_app.py
"""

from __future__ import annotations

import html
import json
import os
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
from app.frontend.styles import (
    category_status_row,
    header_block,
    history_list,
    inject_theme,
    metric_grid,
    panel,
    review_card,
    stars,
)
from app.pipeline.stub import predict_next_review
from app.prompts.templates import render_user_prompt, system_prompt_for_category

OUTPUTS = ROOT / "notebooks" / "outputs"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _format_category(name: str) -> str:
    return name.replace("_", " ")


def _load_eval(category: str) -> dict | None:
    path = OUTPUTS / f"eval_{category.lower()}.json"
    if not path.exists():
        path = OUTPUTS / "eval_books.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _dataset_stats() -> dict:
    rev_path = config.DATA_PROCESSED_DIR / "reviews.parquet"
    if not rev_path.exists():
        return {}
    df = pd.read_parquet(rev_path)
    return {
        "reviews": len(df),
        "users": df[C.F_USER_ID].nunique(),
        "items": df[C.F_ITEM_ID].nunique(),
        "train": int((df["split"] == "train").sum()) if "split" in df.columns else 0,
        "test": int((df["split"] == "test").sum()) if "split" in df.columns else 0,
    }


@st.cache_data(show_spinner=False)
def _profiles_df() -> pd.DataFrame:
    path = config.DATA_PROCESSED_DIR / "user_profiles.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _sidebar():
    n_cat = sum(1 for c in config.AMAZON_CATEGORIES if has_sample(c))
    prof_ok = (config.DATA_PROCESSED_DIR / "user_profiles.parquet").exists()
    llm_on = bool(os.environ.get("ANTHROPIC_API_KEY"))
    stats = _dataset_stats()

    with st.sidebar:
        st.markdown("### Control panel")
        st.caption("Pipeline status and quick context for the live demo.")

        metric_grid([
            ("Categories", f"{n_cat}/{len(config.AMAZON_CATEGORIES)}"),
            ("Profiles", "ready" if prof_ok else "missing"),
            ("Engine", "Claude" if llm_on else "stub"),
        ])

        if stats:
            st.markdown("---")
            st.markdown("**Processed dataset**")
            metric_grid([
                ("Reviews", f"{stats['reviews']:,}"),
                ("Users", f"{stats['users']:,}"),
                ("Holdout", f"{stats['test']:,}"),
            ])

        ev = _load_eval("Books")
        if ev:
            st.markdown("---")
            st.markdown("**Latest eval (Books)**")
            metric_grid([
                ("MAE", f"{ev['rating']['mae']:.2f}"),
                ("ROUGE-L", f"{ev['text']['rougeL_f']:.2f}"),
            ])

        st.markdown("---")
        with st.expander("Demo walkthrough"):
            st.markdown(
                "1. **Predict** — pick a user with 2+ reviews\n"
                "2. **Insights** — EDA plots and takeaways\n"
                "3. **Evaluate** — holdout metrics on sample users\n"
                "4. **Prompts** — inspect LLM context\n"
                "5. **Data** — verify cached samples"
            )


def _run_setup_hint():
    st.warning("Sample data is not loaded yet.")
    st.code("python scripts/build_all.py --size 10000", language="bash")
    if st.button("Build sample pipeline", type="primary"):
        with st.spinner("Fetching and processing samples..."):
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts/build_all.py"), "--size", "10000"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
        if r.returncode != 0:
            st.error(r.stderr or "Setup failed")
        else:
            st.success("Build complete. Refreshing...")
            st.cache_data.clear()
            st.rerun()


def tab_predict():
    if not has_sample("Books"):
        _run_setup_hint()
        return

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown("#### Configure prediction")
        category = st.selectbox(
            "Product category",
            config.AMAZON_CATEGORIES,
            format_func=_format_category,
            key="pred_cat",
        )

        if not has_sample(category):
            st.info(f"No sample for {_format_category(category)} yet.")
            return

        users = load_demo_users(category, limit=40)
        default_u = default_demo_user(category) or (users[0] if users else "")
        user_id = (
            st.selectbox("Demo user", users, index=0, format_func=lambda u: u.replace("amz_", ""))
            if users
            else st.text_input("User ID", value=default_u or "")
        )

        st.caption("Users with at least two reviews in the cached sample.")
        run = st.button("Generate prediction", type="primary", use_container_width=True)

    with right:
        st.markdown("#### Model output")

        if run:
            with st.spinner("Running predictor..."):
                st.session_state["pred_result"] = predict_next_review(str(user_id), category)

        result = st.session_state.get("pred_result")
        if not result:
            panel(
                "Ready when you are",
                "<p style='color:var(--muted);margin:0;line-height:1.6;'>"
                "Select a category and user, then generate a next-review prediction. "
                "The stub uses historical ratings and similar reviews; set "
                "<code>ANTHROPIC_API_KEY</code> for Claude generation."
                "</p>",
            )
            return

        pred = result["prediction"]
        mode = result.get("meta", {}).get("mode", "stub")
        review_card(
            pred.get("title", "Predicted review"),
            pred.get("text", ""),
            pred.get("rating", 0),
            mode,
        )

        meta = result.get("meta", {})
        if meta.get("top_categories"):
            st.caption(
                f"Cross-category profile: {meta.get('category_count', 0)} categories · "
                f"top: {', '.join(meta.get('top_categories', []))}"
            )

    st.markdown("---")
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        history_list(result.get("user_history", []))

    with col_b:
        retrieved = result.get("retrieved", [])
        if retrieved:
            rows = []
            for i, item in enumerate(retrieved, 1):
                rows.append(
                    f"<div class='bct-history-item'>"
                    f"<div class='bct-history-meta'><span>Similar #{i}</span>{stars(item.get('rating', 0))}</div>"
                    f"<p class='bct-history-text'>{html.escape(str(item.get('text', '')))}</p></div>"
                )
            panel("Retrieved context", "".join(rows))
        else:
            panel("Retrieved context", "<p style='color:var(--muted);margin:0;'>No similar reviews found.</p>")

        profile = result.get("profile")
        if profile:
            with st.expander("User profile JSON"):
                st.json({k: v for k, v in profile.items() if k != "sample_reviews"})


def tab_insights():
    summary_path = OUTPUTS / "eda_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        impl = data.get("implications_markdown")
        if impl:
            st.markdown("#### Key takeaways")
            st.markdown(impl)

        cats = data.get("categories", {})
        if cats:
            rows = []
            for cat, info in cats.items():
                task = info.get("users_with_min_reviews_for_task", {})
                pct = task.get("pct", 0)
                rows.append(( _format_category(cat), f"{info.get('n_reviews', 0):,} reviews", f"{pct:.0f}% multi-review users"))
            st.markdown("#### Category coverage")
            html_rows = "".join(
                f"<div class='bct-category-row'><span>{c}</span>"
                f"<span style='color:var(--muted);'>{r} · {u}</span></div>"
                for c, r, u in rows
            )
            panel("Streamed samples", html_rows)
    else:
        st.info("Run `python scripts/run_eda.py` to generate insights.")

    plots = sorted(OUTPUTS.glob("*.png")) if OUTPUTS.exists() else []
    if plots:
        st.markdown("#### Visualizations")
        cols = st.columns(2)
        for i, p in enumerate(plots):
            with cols[i % 2]:
                st.image(str(p), caption=p.stem.replace("_", " "), use_container_width=True)
    elif summary_path.exists():
        st.caption("No plot files found in notebooks/outputs/.")


def tab_eval():
    st.markdown("#### Holdout evaluation")
    st.caption("Scores stub or LLM predictions on the last held-out review per user.")

    category = st.selectbox(
        "Category",
        config.AMAZON_CATEGORIES,
        format_func=_format_category,
        key="eval_cat",
    )
    max_users = st.slider("Users to evaluate", 5, 50, 20)

    existing = _load_eval(category)
    if existing:
        st.markdown("##### Current scores")
        metric_grid([
            ("Samples", str(existing.get("n", "—"))),
            ("MAE", f"{existing['rating']['mae']:.3f}"),
            ("RMSE", f"{existing['rating']['rmse']:.3f}"),
            ("ROUGE-1", f"{existing['text']['rouge1_f']:.3f}"),
            ("ROUGE-L", f"{existing['text']['rougeL_f']:.3f}"),
            ("Recall@5", f"{existing['retrieval']['recall@5']:.3f}"),
        ])

    if st.button("Re-run evaluation", type="primary"):
        with st.spinner("Evaluating on holdout users..."):
            r = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_eval_on_sample.py"),
                    "--category", category,
                    "--max-users", str(max_users),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
        if r.returncode != 0:
            st.error(r.stderr or "Evaluation failed")
        else:
            st.success("Evaluation complete.")
            if r.stdout:
                st.code(r.stdout.strip())
            st.cache_data.clear()
            st.rerun()


def tab_prompts():
    if not has_sample("Books"):
        _run_setup_hint()
        return

    col1, col2 = st.columns([1, 1.2], gap="large")
    with col1:
        category = st.selectbox(
            "Category",
            config.AMAZON_CATEGORIES,
            format_func=_format_category,
            key="prompt_cat",
        )
        users = load_demo_users(category, limit=20)
        user_id = st.selectbox("User", users, key="prompt_user") if users else st.text_input("User ID")
        build = st.button("Assemble prompt", type="primary", use_container_width=True)

    with col2:
        if not build:
            panel(
                "Prompt inspector",
                "<p style='color:var(--muted);margin:0;'>Build a prompt to see the exact system "
                "and user messages sent to Claude, including retrieved similar reviews.</p>",
            )
            return

        result = predict_next_review(str(user_id), category)
        item_meta = {
            "title": result["prediction"].get("title", "Product"),
            "category": category,
            "description": result["prediction"].get("text", "")[:300],
        }
        user_prompt = render_user_prompt(result["user_history"], item_meta, result["retrieved"])
        st.markdown("##### System message")
        st.code(system_prompt_for_category(category), language="text")
        st.markdown("##### User message")
        st.code(user_prompt, language="text")


def tab_data():
    st.markdown("#### Cached sample inventory")

    rows_html = []
    total_mb = 0.0
    loaded = 0
    for cat in config.AMAZON_CATEGORIES:
        p = config.DATA_RAW_DIR / f"{cat.lower()}_reviews_sample.parquet"
        if p.exists():
            loaded += 1
            total_mb += p.stat().st_size / 1_048_576
            df = load_sample(cat)
            rows_html.append(category_status_row(_format_category(cat), len(df), df[C.F_USER_ID].nunique()))
        else:
            rows_html.append(category_status_row(_format_category(cat), None, None))

    panel("Amazon categories", "".join(rows_html))

    c1, c2, c3 = st.columns(3)
    c1.metric("Categories loaded", f"{loaded}/{len(config.AMAZON_CATEGORIES)}")
    c2.metric("Raw sample size", f"{total_mb:.1f} MB")
    prof = _profiles_df()
    c3.metric("User profiles", f"{len(prof):,}" if len(prof) else "—")


def tab_profiles():
    df = _profiles_df()
    if df.empty:
        st.info("Run `python scripts/build_all.py` to build cross-category user profiles.")
        return

    st.markdown("#### Cross-category user profiles")
    st.caption("Aggregated behavior across all cached Amazon categories.")

    query = st.text_input("Filter by user ID", placeholder="e.g. amz_A1...")
    view = df.copy()
    if query.strip():
        view = view[view[C.F_USER_ID].astype(str).str.contains(query.strip(), case=False, na=False)]

    display_cols = [c for c in view.columns if c != "sample_reviews"]
    st.dataframe(
        view[display_cols].head(200),
        use_container_width=True,
        hide_index=True,
        height=420,
    )


def main():
    st.set_page_config(
        page_title="Review Predictor · BCT Hack",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    n_cat = sum(1 for c in config.AMAZON_CATEGORIES if has_sample(c))
    llm_on = bool(os.environ.get("ANTHROPIC_API_KEY"))
    stats = _dataset_stats()

    header_block(
        "Review Predictor",
        "Predict a shopper's next product review from their history, similar item reviews, and cross-category profile.",
        [
            (f"{n_cat} categories", "ok" if n_cat == len(config.AMAZON_CATEGORIES) else "warn"),
            ("LLM connected" if llm_on else "Stub mode", "live" if llm_on else "warn"),
            (f"{stats.get('users', 0):,} users" if stats else "No processed data", "ok" if stats else "warn"),
        ],
    )

    _sidebar()

    tabs = st.tabs(["Predict", "Insights", "Evaluate", "Profiles", "Prompts", "Data"])
    with tabs[0]:
        tab_predict()
    with tabs[1]:
        tab_insights()
    with tabs[2]:
        tab_eval()
    with tabs[3]:
        tab_profiles()
    with tabs[4]:
        tab_prompts()
    with tabs[5]:
        tab_data()


if __name__ == "__main__":
    main()
