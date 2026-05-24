# BCT Hack — Next-Review Predictor (Task A) + Recommendation Engine (Task B)

**DSN × BCT LLM Agent Challenge**

An intelligent review prediction and recommendation system built on the **Amazon Reviews 2023** dataset.

The system learns from a user's historical reviews, retrieves similar behavioral patterns, builds a cross-category profile, and predicts what the user is likely to review next (**Task A**). It also supports product recommendation through an API-based recommendation endpoint (**Task B**).

We stream and process **8 Amazon categories** locally rather than the full **571M-review corpus**, balancing performance and reproducibility.

---

## Live Demo

| Service | Purpose |
|---|---|
| **Streamlit App** | **Task A only** — Predict a user's next Amazon review |
| **API Docs** | Test **Task A** and **Task B** endpoints |

### Task distinction

- **Task A — Next Review Prediction** → Available in the **Streamlit UI**
- **Task B — Recommendation Engine** → Test via the API endpoint: `POST /recommend`

The deployed Streamlit app intentionally focuses on **Task A** for simpler interactive evaluation.

---

## Features

| Layer | Status |
|---|---|
| HF streaming → Parquet pipeline | ✅ |
| Cross-category user profiles | ✅ |
| FAISS retrieval (TF-IDF, offline) | ✅ |
| Next-review prediction | ✅ |
| Holdout evaluation | ✅ |
| Streamlit UI (Task A) | ✅ |
| FastAPI API (Task A + Task B) | ✅ |
| Docker deployment | ✅ |
| Groq-powered generation (optional) | ⚙️ |
| Dense embeddings (BGE) | Upgrade path |

---

## How to Test

### Task A — Predict Next Review

Open the deployed **Streamlit app** and provide:

- `user_id`
- `category`

The system returns:

- User review history
- Retrieved similar reviews
- Predicted review title
- Predicted review text
- Predicted rating
- Cross-category user profile

---

### Task B — Recommendation Engine

Use the deployed API docs and test:

```http
POST /recommend
```

Example request:

```json
{
  "user_id": "amz_123",
  "category": "Books"
}
```

Returns ranked product recommendations using:

- User historical behavior
- Similar retrieved reviews
- Cross-category profile signals

---

## Architecture

```text
User History
      ↓
Cross-category Profile Builder
      ↓
FAISS Similarity Retrieval
      ↓
Context Aggregation
      ↓
Prediction / Recommendation Layer
```

### Retrieval Stack

Current implementation:

- **TF-IDF embeddings**
- **FAISS vector retrieval**
- Offline index generation

Planned upgrade path:

- **BGE dense embeddings**
- `sentence-transformers`
- richer agentic reasoning chains

without changing the API contract.

---

## API Contract

### Task A — Predict Next Review

```http
POST /predict
```

Request:

```json
{
  "user_id": "amz_...",
  "category": "Books",
  "target_item_id": null
}
```

Response:

```json
{
  "user_history": [
    {
      "item_id": "...",
      "rating": 5,
      "text": "...",
      "title": "..."
    }
  ],
  "prediction": {
    "rating": 4.5,
    "title": "...",
    "text": "..."
  },
  "retrieved": [...],
  "profile": {...},
  "meta": {
    "mode": "...",
    "retrieval": "..."
  }
}
```

> Pass `target_item_id` during evaluation to avoid holdout leakage.

---

### Task B — Recommendation Endpoint

```http
POST /recommend
```

Returns ranked recommendations and relevance scores for the selected category.

---

## Evaluation

### Task A

Metrics:

- **MAE**
- **RMSE**
- **ROUGE-style text overlap**

### Task B

Metrics:

- **Recall@K**
- **NDCG@K**

Run evaluation:

```powershell
python scripts/evaluate_tasks.py --task all --category Books --max-users 40
```

---

## Local Development

Install dependencies:

```powershell
pip install -r requirements-demilade.txt
```

Build data:

```powershell
python scripts/build_all.py --size 10000
```

Run Streamlit:

```powershell
python -m streamlit run app/frontend/streamlit_app.py
```

Run API:

```powershell
python scripts/run_api.py
```

Reuse cached data:

```powershell
python scripts/build_all.py --skip-fetch
```

---

## Docker

```powershell
python scripts/build_all.py --skip-fetch
docker compose up --build
```

---

## Categories

- Books
- Electronics
- Home_and_Kitchen
- Sports_and_Outdoors
- Video_Games
- Pet_Supplies
- All_Beauty
- Office_Products

---

## Team Split

| Role | Focus |
|---|---|
| **Architect** | BGE embeddings, FAISS tuning, FastAPI extensions, agent-chain architecture |
| **Demilade** | EDA, evaluation, prompts, Streamlit, profiles, data pipeline, Docker |

Integration contract:

```python
app/team_contract.py
```

---

## Repository Layout

```text
BCT-hack/
├── app/
│   ├── api/
│   ├── pipeline/
│   ├── frontend/
│   ├── eval/
│   └── team_contract.py
├── data/
├── scripts/
├── Dockerfile
└── docker-compose.yml
```