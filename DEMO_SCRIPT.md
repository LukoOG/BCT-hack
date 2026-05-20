# 2-minute demo script (presentation)

## Before you start (30 sec)

1. Run: `powershell -File run.ps1` (or the commands in README)
2. Browser opens at `http://localhost:8501`
3. Optional: copy `.env.example` → `.env` and add `ANTHROPIC_API_KEY` for live LLM

---

## Minute 1 — Problem + data

**Say:** "We predict the *next* review a user would write, using their past reviews and similar reviews of the product."

1. Open **Data** tab → show 8k+ sample reviews per category (streamed from Amazon Reviews 2023, not full download)
2. Open **EDA** tab → scroll implications:
   - ~56% of users have 2+ reviews → task is feasible
   - Ratings skewed 5-star → we report MAE *and* per-class metrics
   - Books reviews longer than Electronics → different prompt lengths

---

## Minute 2 — Predict + eval

3. Open **Predict** tab → pick a user from dropdown → **Predict**
   - Show past reviews, predicted rating/text, retrieved similar reviews
   - Note mode: `stub` vs `llm` (if API key set)

4. Open **Prompts** tab → **Build prompt** → show system + user prompt we send to Claude

5. Open **Eval** tab → **Run eval** → show MAE / ROUGE on held-out last review per user
   - Current sample (stub): MAE ~0.73, ROUGE ~0.69 on 15 users
   - Retrieval metrics low until teammate wires FAISS

**Close:** "Contract for integration is in `TEAM_CONTRACT.md` — one function: `predict_next_review(user_id, category)`."

---

## If something breaks

| Issue | Fix |
|-------|-----|
| Wrong folder | `cd` to `BCT-hack` or use `run.ps1` |
| No users in dropdown | `python scripts/setup_demilade.py` |
| streamlit not found | Activate `.venv` first |
