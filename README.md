# BCT Hack — Next-Review Predictor

Predict a user's next Amazon product review from their past reviews and similar reviews of the target item. Built on [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). We stream **8 categories** into local Parquet (not the full 571M-review corpus).

## Team split

| Role | Owner | Responsibility |
|------|--------|----------------|
| Architect | Teammate | Embeddings, FAISS retrieval, APIs, full pipeline |
| EDA / eval / demo | Demilade | Data exploration, prompts, metrics, Streamlit UI |

## What is done so far

### Shared foundation (architect — `main` baseline)

- `app/core/config.py` — paths, categories (Books, Electronics), filters, LLM/embeddings settings
- `app/core/constants.py` — unified schema field names
- `app/data/loader.py` — UCSD download helpers (URLs updated to `review_categories/` layout)
- `app/data/preprocess.py` — raw JSONL → unified review/item schema, train/test split
- `app/data/dataset.py` — Parquet + DuckDB query layer
- `app/utils/` — logging, text cleaning, parquet I/O

### Demilade track (`demilade-eda-eval` branch)

**Data (no full dataset download)**

- HuggingFace **streaming** via `scripts/fetch_samples.py` (`trust_remote_code`, `datasets<3`)
- Cached samples: `data/raw/books_reviews_sample.parquet`, `electronics_reviews_sample.parquet` (gitignored)
- Demo user lists: `data/raw/*_demo_users.json` (users with 2+ reviews)

**EDA**

- `notebooks/eda.ipynb` — sections A–H (schema, length, ratings, users/items, time, vocab, category compare)
- `scripts/run_eda.py` — generates plots under `notebooks/outputs/` and `eda_summary.json`
- Key finding (8k Books sample): **~56%** of users have ≥2 reviews → “predict last review” task is viable

**Evaluation**

- `app/eval/metrics.py` — MAE, RMSE, ROUGE-1/L, Recall@k, NDCG@k
- `app/eval/runner.py` — `evaluate_predictions(holdout_df, predictions)`
- `app/eval/judge.py` — LLM-as-judge prompt (optional qualitative scoring)
- `scripts/run_eval_on_sample.py` — holdout last review per user on sample data
- Sample stub results (15 users): **MAE 0.73**, **ROUGE-L 0.69** — `notebooks/outputs/eval_books.json`

**Prompts**

- `app/prompts/templates.py` — system/user templates; category hints (Books vs Electronics)
- `app/prompts/generate.py` — Claude generation when `ANTHROPIC_API_KEY` is set

**Pipeline + demo**

- `app/pipeline/stub.py` — `predict_next_review(user_id, category)` using sample data (+ optional LLM)
- `app/frontend/streamlit_app.py` — tabs: **Predict**, **EDA**, **Eval**, **Prompts**, **Data**
- `app/team_contract.py` — integration signature for architect (replaces markdown docs)
- `run.ps1` — Windows launcher (venv, sample fetch, Streamlit)

**Dependencies**

- `requirements.txt` — full project
- `requirements-demilade.txt` — lighter set (no torch/faiss) for EDA/eval/UI only

### Not done yet (expected)

- Full category `.jsonl.gz` download (multi-GB; avoid on slow wifi)
- Real retrieval (FAISS) and architect pipeline replacing `app/pipeline/stub.py`
- Goodreads integration (preprocess exists; not in MVP)
- PR merge of `demilade-eda-eval` → `main` (open on GitHub when ready)

## Lead plan (current priority)

1. **Download max samples** — `python scripts/build_all.py --size 50000` (8 categories ≈ 400k reviews, ~200–800 MB on disk depending on text length)
2. **User profiles** — `data/processed/user_profiles.parquet` (cross-category, same Amazon `user_id`)
3. **Teammate** — FAISS retrieval + replace `app/pipeline/stub.py`
4. **Demo** — Streamlit with Predict + Profiles tabs

## Quick start

```powershell
cd C:\Users\User\Desktop\nothing\bct\BCT-hack
powershell -File run.ps1
```

Or full pipeline manually:

```powershell
pip install -r requirements-demilade.txt
python scripts/build_all.py --size 50000
streamlit run app/frontend/streamlit_app.py
```

Optional LLM: copy `.env.example` → `.env` and set `ANTHROPIC_API_KEY`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/fetch_samples.py` | Stream HF samples → `data/raw/*.parquet` |
| `scripts/run_eda.py` | Plots + `notebooks/outputs/eda_summary.json` |
| `scripts/run_eval_on_sample.py` | Holdout eval on sample users |
| `scripts/setup_demilade.py` | fetch + EDA + eval in one go |

## Integration contract

Architect implements `predict_next_review(user_id, category) -> dict` as documented in `app/team_contract.py`. Streamlit and eval call this entry point.

**Return shape:**

```python
{
  "user_history": [{"item_id", "rating", "text"}, ...],
  "prediction": {"rating", "title", "text"},
  "retrieved": [{"item_id", "rating", "text"}, ...],
}
```

Eval holdout columns: `true_rating`, `true_text` on holdout; predictions use `pred_rating`, `pred_text`, optional `retrieved_item_ids`.

## Repository layout

```
BCT-hack/
├── app/
│   ├── core/           # config, constants
│   ├── data/           # loader, preprocess, dataset, sample_store
│   ├── eval/           # metrics, runner, judge
│   ├── prompts/        # templates, generate
│   ├── pipeline/       # stub predict_next_review
│   ├── frontend/       # Streamlit app
│   └── team_contract.py
├── notebooks/
│   ├── eda.ipynb
│   ├── evaluation.ipynb
│   └── outputs/        # PNG plots, eda_summary.json, eval_*.json
├── scripts/
├── run.ps1
├── requirements.txt
└── requirements-demilade.txt
```

## Branches

- `main` — architect baseline
- `demilade-eda-eval` — EDA, eval, prompts, Streamlit demo (active development)

## Dataset note

We use **streamed samples** (default 10k–20k rows per category), not the full 571M-review corpus. To refresh:

```powershell
python scripts/fetch_samples.py --size 50000 --force
python scripts/run_eda.py
```

Full per-category files: [Amazon Reviews 2023 dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) or UCSD `review_categories/{Category}.jsonl.gz`.
