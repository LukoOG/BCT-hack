# Demilade — your task list

**Your track:** EDA · Prompts · Evaluation · Frontend/demo  
**Teammate track:** Architecture · Embeddings · Retrieval · APIs

---

## Status legend

- [x] Done in repo
- [ ] You do next
- [~] Blocked on teammate

---

## Phase 1 — Setup (do once)

| # | Task | Status | Command / file |
|---|------|--------|----------------|
| 1.1 | Python venv | [ ] | `python -m venv .venv` then activate |
| 1.2 | Install your deps | [ ] | `pip install -r requirements-demilade.txt` |
| 1.3 | Download sample data (not full dataset) | [x] script ready | `python scripts/setup_demilade.py` |
| 1.4 | Confirm plots exist | [x] | `notebooks/outputs/*.png` |
| 1.5 | Open PR / merge to `main` | [ ] | GitHub PR from `demilade-eda-eval` |

---

## Phase 2 — EDA (your deliverable for the team)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | Run full sample (10k–50k rows) | [ ] | `python scripts/fetch_samples.py --size 50000 --force` |
| 2.2 | Re-run EDA | [x] | `python scripts/run_eda.py` |
| 2.3 | Walk through `notebooks/eda.ipynb` | [ ] | Present to teammate |
| 2.4 | **Share §D finding** | [x] sample | ~57% users have 2+ reviews — task viable |
| 2.5 | Write 5 bullets in Implications | [x] | `notebooks/outputs/IMPLICATIONS.md` |
| 2.6 | Section H Books vs Electronics | [x] | `H_books_vs_electronics.png` |

---

## Phase 3 — Prompts

| # | Task | Status | File |
|---|------|--------|------|
| 3.1 | Review system + user templates | [x] | `app/prompts/templates.py` |
| 3.2 | Preview prompts in UI | [x] | Streamlit → **Prompts** tab |
| 3.3 | Test real Claude generation | [ ] | Set `ANTHROPIC_API_KEY` in `.env` |
| 3.4 | Tune prompt after EDA (length, tone) | [ ] | Books vs Electronics differ |
| 3.5 | Hand prompts to teammate for API | [x] | `TEAM_CONTRACT.md` |

---

## Phase 4 — Evaluation

| # | Task | Status | File |
|---|------|--------|------|
| 4.1 | Rating metrics (MAE, RMSE) | [x] | `app/eval/metrics.py` |
| 4.2 | Text metrics (ROUGE) | [x] | `app/eval/metrics.py` |
| 4.3 | Retrieval metrics (Recall@k, NDCG) | [x] | `app/eval/metrics.py` |
| 4.4 | Runner on holdout format | [x] | `app/eval/runner.py` |
| 4.5 | LLM judge prompt | [x] | `app/eval/judge.py` |
| 4.6 | Eval on sample users | [x] | `python scripts/run_eval_on_sample.py` |
| 4.7 | Eval in Streamlit | [x] | **Eval** tab |
| 4.8 | Re-run eval on **real** pipeline | [~] | When teammate wires retrieval + LLM |
| 4.9 | Report numbers in slides/README | [ ] | Copy from `eval_books.json` |

---

## Phase 5 — Frontend / demo

| # | Task | Status | File |
|---|------|--------|------|
| 5.1 | Streamlit app shell | [x] | `app/frontend/streamlit_app.py` |
| 5.2 | User picker from sample | [x] | Uses `*_demo_users.json` |
| 5.3 | Predict tab | [x] | Stub + optional LLM |
| 5.4 | EDA tab (plots + implications) | [x] | Reads `notebooks/outputs/` |
| 5.5 | Eval tab | [x] | Runs `run_eval_on_sample.py` |
| 5.6 | Data status tab | [x] | Shows sample row counts |
| 5.7 | Polish UI (title, errors, loading) | [ ] | Your design pass |
| 5.8 | Demo script for presentation | [ ] | 2-min walkthrough |
| 5.9 | Swap stub → real pipeline | [~] | `app/pipeline/stub.py` → teammate's module |

---

## Phase 6 — Coordination (teammate)

| # | Task | Status |
|---|------|--------|
| 6.1 | Send `TEAM_CONTRACT.md` | [ ] |
| 6.2 | Confirm `predict_next_review(user_id, category)` | [x] defined |
| 6.3 | Agree MVP = Books only first | [ ] |
| 6.4 | Get item metadata into prompts | [~] needs meta fetch or their API |

---

## Quick commands (copy-paste)

```powershell
cd BCT-hack
.venv\Scripts\activate
pip install -r requirements-demilade.txt

python scripts/setup_demilade.py
streamlit run app/frontend/streamlit_app.py
```

Optional LLM:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
streamlit run app/frontend/streamlit_app.py
```

---

## MVP vs stretch

| MVP (ship this) | Stretch |
|-----------------|---------|
| Books sample EDA | Full Books download |
| Streamlit 4 tabs | Goodreads |
| Stub + optional Claude | FAISS retrieval in UI |
| MAE + ROUGE on sample | LLM judge on 50 users |

---

## What NOT to do (wifi / time)

- Do not download full `Books.jsonl.gz` (~multi-GB) on campus wifi
- Do not block on Goodreads until Amazon MVP works
- Do not rebuild embeddings — that's teammate's job
