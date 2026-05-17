# DSN x BCT LLM Agent Hackathon 3.0

Unified pipeline for **Task A** (User Modeling) and **Task B** (Recommendation)  
built on Amazon Reviews 2023 + Goodreads.

---

## Project structure

```
project-root/
├── data/
│   ├── raw/                  ← downloaded source files (.jsonl.gz)
│   ├── processed/            ← cleaned parquet + reviews.duckdb
│   └── embeddings/           ← FAISS indexes + embedding arrays
├── app/
│   ├── core/                 ← config.py, constants.py
│   ├── data/                 ← loader, preprocess, dataset  ✅ Phase 1
│   ├── embeddings/           ← embedder, vector_store        Phase 2
│   ├── profiles/             ← user_profile, profile_builder Phase 2
│   ├── retrieval/            ← retriever, ranking            Phase 3
│   ├── llm/                  ← prompts*, generator, reasoning Phase 3
│   ├── tasks/                ← task_a, task_b                Phase 4
│   ├── evaluation/           ← metrics*, evaluator*          Phase 4
│   └── utils/                ← logger, helpers
├── notebooks/                ← EDA*, experiments*, evaluation* (teammate)
├── main.py                   ← CLI entry point
└── requirements.txt
```
`*` = teammate ownership

---

## Quickstart

### 1. Environment setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

Then load it anywhere with:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Run the pipeline

```bash
# Full pipeline: download everything + preprocess + store
python main.py run

# Or step by step:
python main.py download                            # step 1: download
python main.py process --skip-download             # step 2: process

# Specific categories only (faster for development)
python main.py run --categories Books

# Check what ended up in the DB
python main.py stats
```

---

## Configuration

All tuneable parameters live in `app/core/config.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `AMAZON_CATEGORIES` | `["Books", "Electronics"]` | Which Amazon subsets to fetch |
| `MIN_USER_REVIEWS` | `5` | Drop users below this threshold |
| `MIN_ITEM_REVIEWS` | `10` | Drop items below this threshold |
| `HOLDOUT_LAST_N` | `1` | How many interactions per user go to test set |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformer model |

---

## Data sources

| Source | URL |
|--------|-----|
| Amazon Reviews 2023 | https://amazon-reviews-2023.github.io |
| Goodreads | https://mengtingwan.github.io/data/goodreads |

---

## Teammate handoff

Your teammate owns these files — do not edit them:

- `notebooks/eda.ipynb` — load `data/processed/reviews.parquet` directly
- `notebooks/evaluation.ipynb`
- `app/llm/prompts.py`
- `app/evaluation/metrics.py`
- `app/evaluation/evaluator.py`

After `python main.py run`, the teammate can immediately start EDA:

```python
import pandas as pd
df = pd.read_parquet("data/processed/reviews.parquet")
df.head()
```
