# BCT Hack — Next-Review Predictor (Task A)

**DSN x BCT LLM Agent Challenge** — predict a user's next Amazon product review from their history, retrieved similar reviews, and cross-category profile.

Built on [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). We stream **8 categories** into local Parquet (not the full 571M-review corpus).

## What's implemented

| Layer | Status |
|-------|--------|
| Sample pipeline (HF streaming → Parquet) | Done |
| User profiles (cross-category) | Done |
| FAISS retrieval (TF-IDF, offline) | Done |
| `predict_next_review()` + holdout eval | Done |
| Streamlit demo (About, Predict, API, Eval, …) | Done |
| FastAPI `/predict` + `/health` | Done |
| Docker Compose (API + Streamlit) | Done |
| Groq LLM generation (optional) | Needs `GROQ_API_KEY` |
| Dense embeddings (BGE) | Architect upgrade path |

## Quick start (local)

```powershell
cd BCT-hack
pip install -r requirements-demilade.txt

# First run — streams samples (~10k/category default in fetch script; build_all defaults 50k)
python scripts/build_all.py --size 10000

# Demo UI
python -m streamlit run app/frontend/streamlit_app.py

# API (separate terminal)
python scripts/run_api.py
# → http://127.0.0.1:8000/docs
```

Use existing cached data without re-downloading:

```powershell
python scripts/build_all.py --skip-fetch
```

Evaluate both hackathon tasks from existing processed data:

```powershell
# Uses data/processed/reviews.parquet first, then data/processed/reviews.duckdb
python scripts/evaluate_tasks.py --task all --category Books --max-users 40
python scripts/evaluate_tasks.py --task all --category Electronics --max-users 40

# Or point directly at a teammate's artifact
python scripts/evaluate_tasks.py --source data/processed/reviews.duckdb --task all --category Books
```

Task A reports rating MAE/RMSE plus ROUGE-style text overlap. Task B reports recommendation Recall@k and NDCG@k over sampled candidates.

Optional LLM: copy `.env.example` → `.env` and set `GROQ_API_KEY` (Groq is the default provider).

## Docker (hackathon deliverable)

Build data on the host first, then mount into containers:

```powershell
python scripts/build_all.py --skip-fetch
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit | http://localhost:8501 |
| FastAPI | http://localhost:8000/docs |

## Team split

| Role | Focus |
|------|--------|
| **Architect** | BGE embeddings, FAISS tuning, extend FastAPI, replace heuristic with full agent chain |
| **Demilade** | EDA, eval, prompts, Streamlit, profiles, sample pipeline, Docker |

Integration contract: `app/team_contract.py`

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_all.py` | Fetch → process → profiles → EDA → FAISS indexes → eval |
| `scripts/build_embeddings.py` | Rebuild FAISS indexes only |
| `scripts/run_eval_on_sample.py` | Holdout eval (`--category Books --max-users 15`) |
| `scripts/run_api.py` | FastAPI on port 8000 |
| `scripts/run_eda.py` | Plots + `eda_summary.json` |

**Categories:** Books, Electronics, Home_and_Kitchen, Sports_and_Outdoors, Video_Games, Pet_Supplies, All_Beauty, Office_Products.

## API contract

```python
POST /predict
{"user_id": "amz_...", "category": "Books", "target_item_id": null}

# Returns:
{
  "user_history": [{"item_id", "rating", "text", "title"}, ...],
  "prediction": {"rating", "title", "text"},
  "retrieved": [{"item_id", "rating", "text", "title"}, ...],
  "profile": {...},
  "meta": {"mode", "retrieval", ...}
}
```

Pass `target_item_id` during eval to avoid holdout leakage.

## Architect upgrade

```powershell
pip install torch sentence-transformers
set EMBEDDING_BACKEND=sentence-transformers
python scripts/build_embeddings.py --force
```

See `ARCHITECT_UPGRADE_STEPS` in `app/team_contract.py`.

## Next on the agenda

1. **Scale samples to 50k** — `python scripts/build_all.py --size 50000 --force` (needs Wi-Fi)
2. **Set `GROQ_API_KEY`** — unlock Groq generation for better ROUGE scores
3. **Solution paper** — 4–8 pages (primary judge signal per brief)
4. **Architect** — dense embeddings + full agent chain
5. **Submission** — container link + repo + paper before **24 May 2026**

## Repository layout

```
BCT-hack/
├── app/
│   ├── api/            # FastAPI server
│   ├── pipeline/       # predict_next_review, FAISS retriever, embeddings
│   ├── frontend/       # Streamlit + content copy
│   ├── eval/           # MAE, ROUGE, Recall@k
│   └── team_contract.py
├── data/               # raw samples, processed, embeddings (gitignored)
├── scripts/
├── Dockerfile
└── docker-compose.yml
```
