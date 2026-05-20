# Message to send your architect teammate

Copy-paste (edit names as needed):

---

Hey — Demilade track is ready for you to plug in.

**Branch:** `demilade-eda-eval` (or merge PR to `main`)

**Your integration point:** implement `predict_next_review(user_id, category) -> dict` as documented in `TEAM_CONTRACT.md`. I have a working stub in `app/pipeline/stub.py` — replace the body with retrieval + your LLM call using `app/prompts/templates.py`.

**EDA headline (Books sample, 8k rows):**
- 56% of users have ≥2 reviews → predict-last-review task is viable
- 65% five-star ratings → eval needs more than MAE alone
- Median review ~64 words (Books); truncate embeddings ~400 words

**Eval baseline (stub, 15 users):** MAE 0.73, ROUGE-L 0.69 — see `notebooks/outputs/eval_books.json`. Retrieval Recall@5 is low until your FAISS index is wired.

**Demo:** `powershell -File run.ps1` or `streamlit run app/frontend/streamlit_app.py`

**MVP scope I assumed:** Amazon Books first, then Electronics. Goodreads later.

Can you confirm you're embedding per-review (not per-product aggregate) and when the real pipeline will replace the stub?

---
