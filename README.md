# BCT Hack — Next-Review Predictor

Predict a user's next product review from their history (Amazon Reviews 2023 + optional Goodreads).

## Your role split

| Person | Owns |
|--------|------|
| Architect | embeddings, retrieval, APIs, pipeline |
| Demilade | **EDA**, prompts, evaluation, demo UI |

**Demilade checklist:** [DEMILADE_TASKS.md](DEMILADE_TASKS.md)

## Quick start (EDA / eval side)

```powershell
cd BCT-hack
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 1) Stream small samples (no full dataset download)
python scripts/fetch_samples.py

# 2) Run EDA → plots + summary
python scripts/run_eda.py

# 3) Explore in Jupyter
jupyter notebook notebooks/eda.ipynb
```

Outputs: `notebooks/outputs/` (plots + `eda_summary.json` + `IMPLICATIONS.md`).

## Demo UI

**Easiest (Windows):**

```powershell
cd C:\Users\User\Desktop\nothing\bct\BCT-hack
powershell -File run.ps1
```

Or manually:

```powershell
streamlit run app/frontend/streamlit_app.py
```

**Presentation:** see [DEMO_SCRIPT.md](DEMO_SCRIPT.md)

## Eval results (sample stub, 15 users)

| Metric | Books |
|--------|-------|
| MAE | 0.73 |
| ROUGE-L F1 | 0.69 |
| Recall@5 | 0.13 (stub retrieval) |

Full JSON: `notebooks/outputs/eval_books.json`

## Team contract

See **[TEAM_CONTRACT.md](TEAM_CONTRACT.md)** for `predict_next_review()` signature and eval column names.

## Branches

- `main` — stable baseline from architect
- `demilade-eda-eval` — EDA, eval, prompts, frontend scaffold
