"""
UI copy for the BCT Hack demo — problem statement, pipeline, and live stats hooks.

Grounded in the hackathon brief (Task A: predict last review), Amazon Reviews 2023
(McAuley Lab / UCSD), and our streamed-sample implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EDA_PATH = ROOT / "notebooks" / "outputs" / "eda_summary.json"


def load_eda() -> dict[str, Any]:
    if EDA_PATH.exists():
        return json.loads(EDA_PATH.read_text(encoding="utf-8"))
    return {}


def books_task_stats() -> dict[str, Any]:
    """Key EDA numbers for the predict-last-review task (Books sample)."""
    books = load_eda().get("categories", {}).get("Books", {})
    task = books.get("users_with_min_reviews_for_task", {})
    wc = books.get("word_count", {})
    ratings = books.get("rating_share_pct", {})
    return {
        "n_reviews": books.get("n_reviews", 0),
        "n_users": books.get("n_users", 0),
        "multi_review_pct": task.get("pct", 0),
        "multi_review_count": task.get("count", 0),
        "median_words": wc.get("p50", 0),
        "p95_words": wc.get("p95", 0),
        "five_star_pct": ratings.get("5.0", 0),
    }


PROJECT_KICKER = "DSN x BCT · LLM Agent Challenge · Data & AI Summit Hackathon 3.0"

PROJECT_TITLE = "Next-Review Predictor"

PROJECT_TAGLINE = (
    "Task A — User Modeling: build an agent that understands users deeply enough to simulate "
    "their reviews — capturing tone, rating behaviour, and contextual nuance."
)

COMPETITION_BLURB = """
**DSN x BCT LLM Agent Challenge** (May 1 – Jun 10). Design agents that understand how people
behave, what they want, and what they will choose next. Online review platforms are among the
richest sources of human behaviour data — yet most AI systems still treat users as static
profiles rather than dynamic, context-sensitive agents.

Top performers are invited to a talent conversation with Bluechip Tech's data and AI practice leads.
"""

PROBLEM_STATEMENT = """
**Task A — User Modeling.** Build an agent that simulates star ratings and written reviews for
unseen items by leveraging **user history**, **item metadata**, and **contextual signals**.
Judges evaluate **review quality**, **rating accuracy**, and **behavioural fidelity**.

This demo implements that contract: user persona + product context in, predicted rating, title,
and review text out — via Streamlit UI and a FastAPI `/predict` endpoint.
"""

DATASET_BLURB = """
**Approved datasets** (per hack brief): Yelp, Amazon Reviews, and Goodreads — large-scale platforms
with rich behavioural and textual signals.

**Our choice — [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)**
(McAuley Lab, UCSD): **571M+ reviews** across 33 categories. We **stream samples** (default 10k
rows per category) into local Parquet so the demo runs on a laptop without multi-terabyte downloads.
Goodreads preprocess hooks exist for a future extension.
"""

TASK_BLURB = """
**How we frame Task A.** For each user with at least two reviews, we hold out their **most recent**
interaction (`HOLDOUT_LAST_N = 1`) and ask the agent to predict it from earlier history plus
retrieved similar reviews of the same item — simulating an unseen-item review before the user writes it.

**Required deliverable:** a containerized web app or API that takes user persona and product details
as input and generates reviews and ratings as output.
"""

SCORING_BLURB = """
**Official Task A scoring** (from hack brief):

| Criterion | What we measure |
|-----------|-----------------|
| Review text quality | ROUGE / BERTScore |
| Rating accuracy | RMSE (we also report MAE) |
| Behavioural fidelity | Human evaluation |
| Solution paper | Approach, ablations, architecture (4–8 pages) |
| Code reproducibility | Clean repo, README, runnable pipeline |

Judges note: *"A model score reflects what your machine did. A solution paper reveals what you
understood. Both matter — but in a talent-identification context, the paper is what we read first."*
"""

PIPELINE_STEPS = [
    ("1. User history", "Past reviews across categories build a cross-category profile (avg rating, length, top categories)."),
    ("2. Target item", "The product the user is about to review — identified by item ID in eval, last interaction in demo mode."),
    ("3. Retrieval", "FAISS over cached reviews: same-item reviews from other users first, then semantic neighbors (TF-IDF today; BGE upgrade path for architect)."),
    ("4. Generation", "Claude when `ANTHROPIC_API_KEY` is set; otherwise a FAISS-informed heuristic baseline."),
    ("5. Evaluation", "MAE / RMSE on stars, ROUGE on text, Recall@k / NDCG@k on retrieved item IDs."),
]

METRICS_BLURB = """
Our eval harness tracks the brief's core signals: **RMSE / MAE** on star ratings and **ROUGE-1 / ROUGE-L**
on review text. We also report **Recall@5 / NDCG@5** on retrieval — useful for debugging context
assembly before generation. Amazon ratings skew heavily toward 5 stars, so always read text metrics
alongside rating error.
"""

TIMELINE_BLURB = """
| Milestone | Date |
|-----------|------|
| Hackathon launch | 4 May 2026 |
| **Submission deadline** (model + paper + repo) | **24 May 2026** |
| Judging panel review | 25–29 May 2026 |
| Top 4 teams notified | 29 May – 1 Jun 2026 |
| Presentation deck due | 1–8 Jun 2026 |
| Winner announcement (Data & AI Summit) | **10 Jun 2026** |
"""

TEAM_BLURB = """
| Track | Focus |
|-------|--------|
| **Architect** | Dense embeddings (BGE), FAISS tuning, FastAPI (`/predict`), replace heuristic with full agent chain |
| **Demilade (this demo)** | EDA, prompts, eval harness, Streamlit, user profiles, sample pipeline |
"""

ARCHITECT_NOTE = """
**For your teammate:** swap `EMBEDDING_BACKEND=sentence-transformers`, rebuild indexes with
`python scripts/build_embeddings.py --force`, extend `app/api/server.py`. Contract lives in
`app/team_contract.py` — return shape must stay stable for eval and this UI.
"""


def live_books_callout() -> str:
    s = books_task_stats()
    if not s.get("n_reviews"):
        return "Run `python scripts/run_eda.py` to populate live Books statistics."
    return (
        f"On our **Books** sample ({s['n_reviews']:,} reviews, {s['n_users']:,} users): "
        f"**{s['multi_review_pct']:.0f}%** of users have 2+ reviews ({s['multi_review_count']:,} users) "
        f"→ viable for Task A. Median review length **{s['median_words']:.0f} words** "
        f"(p95 ≈ {s['p95_words']:.0f}). **{s['five_star_pct']:.0f}%** of ratings are 5-star."
    )
