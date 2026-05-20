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
import urllib.error
import urllib.request
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
from app.frontend.content import (
    ARCHITECT_NOTE,
    COMPETITION_BLURB,
    DATASET_BLURB,
    METRICS_BLURB,
    PIPELINE_STEPS,
    PROBLEM_STATEMENT,
    PROJECT_KICKER,
    PROJECT_TAGLINE,
    PROJECT_TITLE,
    SCORING_BLURB,
    TASK_BLURB,
    TEAM_BLURB,
    TIMELINE_BLURB,
    live_books_callout,
)
from app.frontend.styles import (
    callout,
    category_status_row,
    header_block,
    history_list,
    inject_theme,
    metric_grid,
    panel,
    pipeline_steps,
    review_card,
    stars,
)
from app.pipeline.embeddings import active_backend_name
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


def _api_base() -> str:
    return os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _api_request(method: str, path: str, body: dict | None = None, timeout: float = 30) -> dict:
    url = f"{_api_base()}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _faiss_ready() -> bool:
    manifest = config.DATA_EMBEDDINGS_DIR / "manifest.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return bool(data)
    except json.JSONDecodeError:
        return False


def _engine_label() -> tuple[str, str]:
    """Return (display label, chip tone)."""
    llm_on = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if llm_on:
        return "Claude + FAISS", "live"
    if _faiss_ready():
        backend = active_backend_name()
        return f"FAISS ({backend})", "ok"
    return "Heuristic only", "warn"


def _sidebar():
    n_cat = sum(1 for c in config.AMAZON_CATEGORIES if has_sample(c))
    prof_ok = (config.DATA_PROCESSED_DIR / "user_profiles.parquet").exists()
    llm_on = bool(os.environ.get("ANTHROPIC_API_KEY"))
    engine_label, engine_tone = _engine_label()
    stats = _dataset_stats()

    with st.sidebar:
        st.markdown("### Control panel")
        st.caption("Live status for the BCT Hack demo — Task A: predict the user's last review.")

        metric_grid([
            ("Categories", f"{n_cat}/{len(config.AMAZON_CATEGORIES)}"),
            ("Profiles", "ready" if prof_ok else "missing"),
            ("Engine", engine_label),
        ])

        if stats:
            st.markdown("---")
            st.markdown("**Processed dataset**")
            st.caption("After quality filters and train/test split.")
            metric_grid([
                ("Reviews", f"{stats['reviews']:,}"),
                ("Users", f"{stats['users']:,}"),
                ("Holdout", f"{stats['test']:,}"),
            ])

        ev = _load_eval("Books")
        if ev:
            st.markdown("---")
            st.markdown("**Latest eval (Books)**")
            st.caption("Honest holdout — last review hidden from the model.")
            metric_grid([
                ("MAE", f"{ev['rating']['mae']:.2f}"),
                ("ROUGE-L", f"{ev['text']['rougeL_f']:.2f}"),
                ("Recall@5", f"{ev['retrieval']['recall@5']:.2f}"),
            ])

        st.markdown("---")
        with st.expander("What this demo does"):
            st.markdown(
                "**Task A** — given a shopper's review history and what others wrote about "
                "the same product, predict the star rating, title, and body of their "
                "next review.\n\n"
                "Pipeline: user profile → FAISS retrieval → Claude or heuristic generation."
            )
        with st.expander("Demo walkthrough"):
            st.markdown(
                "1. **About** — problem, dataset, pipeline\n"
                "2. **Predict** — run a live next-review prediction\n"
                "3. **API** — test the FastAPI `/predict` endpoint\n"
                "4. **Insights** — EDA from streamed samples\n"
                "5. **Evaluate** — holdout metrics (MAE, ROUGE, Recall@k)\n"
                "6. **Profiles** — cross-category user behavior\n"
                "7. **Prompts** — inspect LLM context assembly\n"
                "8. **Data** — cached Amazon category samples"
            )
        if not llm_on:
            st.caption("Tip: set `ANTHROPIC_API_KEY` in `.env` for Claude generation.")


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


def tab_about():
    st.markdown("#### Competition context")
    st.markdown(COMPETITION_BLURB)

    st.markdown("#### Task A — User Modeling")
    st.markdown(PROBLEM_STATEMENT)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("##### Dataset")
        st.markdown(DATASET_BLURB)
    with col_r:
        st.markdown("##### Our approach")
        st.markdown(TASK_BLURB)

    callout(live_books_callout())

    st.markdown("#### Official scoring rubric")
    st.markdown(SCORING_BLURB)

    st.markdown("#### Key dates")
    st.markdown(TIMELINE_BLURB)

    st.markdown("#### Pipeline architecture")
    st.caption(
        "Each prediction assembles context before generation — retrieval is live via "
        "FAISS indexes built from cached samples."
    )
    pipeline_steps(PIPELINE_STEPS)

    st.markdown("#### How we score predictions")
    st.markdown(METRICS_BLURB)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("##### Team split")
        st.markdown(TEAM_BLURB)
    with col_b:
        st.markdown("##### Architect upgrade path")
        st.markdown(ARCHITECT_NOTE)


def tab_predict():
    if not has_sample("Books"):
        _run_setup_hint()
        return

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown("#### Configure prediction")
        st.caption(
            "We simulate **Task A**: the model sees this user's earlier reviews plus "
            "similar reviews of the target item, then predicts what they would write next."
        )
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

        st.caption("Only users with 2+ reviews in the cached sample — required for holdout evaluation.")
        run = st.button("Generate prediction", type="primary", use_container_width=True)

    with right:
        st.markdown("#### Model output")
        llm_on = bool(os.environ.get("ANTHROPIC_API_KEY"))

        if run:
            with st.spinner("Retrieving similar reviews and generating prediction..."):
                st.session_state["pred_result"] = predict_next_review(str(user_id), category)

        result = st.session_state.get("pred_result")
        if not result:
            engine_hint = (
                "Claude will write the review when `ANTHROPIC_API_KEY` is set."
                if llm_on
                else "Without an API key, a FAISS-informed heuristic fills in rating and text."
            )
            panel(
                "Ready when you are",
                f"<p style='color:var(--muted);margin:0;line-height:1.6;'>"
                f"Select a category and user, then generate a next-review prediction. "
                f"Retrieval uses FAISS over cached reviews ({active_backend_name()} embeddings). "
                f"{engine_hint}</p>",
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
        retrieval = meta.get("retrieval", "none")
        if retrieval == "faiss":
            st.caption(f"Retrieved {len(result.get('retrieved', []))} similar reviews via FAISS.")
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
            panel(
                "Retrieved context",
                "<p style='color:var(--muted);margin:0;'>No similar reviews found for this item. "
                "Run <code>python scripts/build_embeddings.py</code> if indexes are missing.</p>",
            )

        profile = result.get("profile")
        if profile:
            with st.expander("User profile JSON"):
                st.json({k: v for k, v in profile.items() if k != "sample_reviews"})


def tab_insights():
    st.markdown("#### Exploratory analysis")
    st.caption(
        "Stats from **streamed HuggingFace samples** (10k rows per category by default) — "
        "not the full 571M-review corpus. Use these numbers to judge whether Task A is viable per category."
    )
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
    st.caption(
        "For each user with 2+ reviews, we hide their **last** review and score the prediction "
        "against the ground truth. Retrieval metrics check whether the true item appears in the top-k neighbors."
    )
    st.markdown(METRICS_BLURB)

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
                "<p style='color:var(--muted);margin:0;'>Shows the exact system and user messages "
                "sent to Claude — including retrieved similar reviews, user history, and item metadata. "
                "Useful for debugging tone, length, and context window usage.</p>",
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
    st.caption(
        "Amazon Reviews 2023 samples streamed via HuggingFace — stored locally as Parquet so "
        "the demo works offline after the initial build."
    )

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

    if _faiss_ready():
        manifest = json.loads((config.DATA_EMBEDDINGS_DIR / "manifest.json").read_text(encoding="utf-8"))
        st.markdown("##### FAISS indexes")
        idx_rows = []
        for cat, info in manifest.items():
            idx_rows.append(
                f"<div class='bct-category-row'><span>{_format_category(cat)}</span>"
                f"<span style='color:var(--muted);'>{info.get('vectors', 0):,} vectors</span></div>"
            )
        panel("Embedding indexes", "".join(idx_rows))
        st.caption(f"Backend: {active_backend_name()}. Rebuild with `python scripts/build_embeddings.py --force`.")


def tab_api():
    base = _api_base()
    st.markdown("#### FastAPI deliverable")
    st.caption(
        "The hack brief requires a containerized web app or API. "
        "`POST /predict` accepts user persona + category and returns predicted rating, title, and review text."
    )

    col_h, col_c = st.columns([1, 1.2], gap="large")
    with col_h:
        st.markdown("##### Server status")
        try:
            health = _api_request("GET", "/health")
            metric_grid([
                ("Status", health.get("status", "?")),
                ("Embeddings", health.get("embedding_backend", "?")),
                ("LLM", health.get("llm", "?")),
            ])
            st.success(f"API live at `{base}`")
        except Exception as exc:
            st.warning(f"API not reachable at `{base}`")
            st.caption(str(exc))
            st.code("python scripts/run_api.py", language="bash")
            st.markdown("Or run both services: `docker compose up --build`")

        st.markdown("##### Example request")
        st.code(
            f'curl -X POST {base}/predict \\\n'
            '  -H "Content-Type: application/json" \\\n'
            '  -d \'{"user_id": "amz_...", "category": "Books"}\'',
            language="bash",
        )
        st.markdown(f"OpenAPI docs: [{base}/docs]({base}/docs)")

    with col_c:
        st.markdown("##### Try the API")
        if not has_sample("Books"):
            _run_setup_hint()
            return

        category = st.selectbox(
            "Category",
            config.AMAZON_CATEGORIES,
            format_func=_format_category,
            key="api_cat",
        )
        users = load_demo_users(category, limit=20)
        user_id = st.selectbox("User", users, key="api_user") if users else st.text_input("User ID", key="api_uid")
        holdout = st.checkbox("Eval mode (pass target_item_id)", value=False)
        target_item = ""
        if holdout:
            target_item = st.text_input("Target item ID", placeholder="amz_B001...")
        call = st.button("Call POST /predict", type="primary", use_container_width=True)

        if call:
            try:
                body: dict = {"user_id": str(user_id), "category": category}
                if holdout and target_item.strip():
                    body["target_item_id"] = target_item.strip()
                with st.spinner("Calling API..."):
                    result = _api_request("POST", "/predict", body, timeout=120)
                pred = result.get("prediction", {})
                review_card(
                    pred.get("title", "Predicted review"),
                    pred.get("text", ""),
                    pred.get("rating", 0),
                    result.get("meta", {}).get("mode", "stub"),
                )
                with st.expander("Full JSON response"):
                    st.json(result)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode()
                st.error(f"HTTP {exc.code}: {detail}")
            except Exception as exc:
                st.error(str(exc))


def tab_profiles():
    df = _profiles_df()
    if df.empty:
        st.info("Run `python scripts/build_all.py` to build cross-category user profiles.")
        return

    st.markdown("#### Cross-category user profiles")
    st.caption(
        "Each profile aggregates a shopper's behavior across all eight cached categories — "
        "average rating, typical review length, and top categories — to personalize predictions."
    )

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
    engine_label, engine_tone = _engine_label()

    header_block(
        PROJECT_TITLE,
        PROJECT_TAGLINE,
        [
            (f"{n_cat} categories", "ok" if n_cat == len(config.AMAZON_CATEGORIES) else "warn"),
            (engine_label, engine_tone),
            (f"{stats.get('users', 0):,} users" if stats else "No processed data", "ok" if stats else "warn"),
        ],
        kicker=PROJECT_KICKER,
    )

    _sidebar()

    tabs = st.tabs(["About", "Predict", "API", "Insights", "Evaluate", "Profiles", "Prompts", "Data"])
    with tabs[0]:
        tab_about()
    with tabs[1]:
        tab_predict()
    with tabs[2]:
        tab_api()
    with tabs[3]:
        tab_insights()
    with tabs[4]:
        tab_eval()
    with tabs[5]:
        tab_profiles()
    with tabs[6]:
        tab_prompts()
    with tabs[7]:
        tab_data()


if __name__ == "__main__":
    main()
