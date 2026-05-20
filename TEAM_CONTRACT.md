# Team contract — Demilade ↔ Architect

## Prediction API (frontend + eval plug-in point)

Implement this in the pipeline module; the Streamlit demo calls it.

```python
def predict_next_review(user_id: str, category: str) -> dict:
    """
    Returns:
        {
          "user_history": [
              {"item_id": str, "rating": int|float, "text": str},
              ...
          ],
          "prediction": {
              "rating": int,
              "title": str,
              "text": str,
          },
          "retrieved": [
              {"item_id": str, "rating": int|float, "text": str},
              ...
          ],
        }
    """
```

**Stub today:** `app/frontend/streamlit_app.py`  
**Wire here when ready:** replace stub body with import from pipeline.

---

## Eval holdout format

`app/eval/runner.evaluate_predictions(holdout_df, predictions)` expects:

| holdout_df column | predictions dict key |
|-------------------|----------------------|
| `user_id` | `user_id` |
| `item_id` | `item_id` |
| `true_rating` | — |
| `true_text` | — |
| — | `pred_rating` |
| — | `pred_text` |
| — | `retrieved_item_ids` (optional list) |

Rename from unified schema: `rating` → `true_rating`, `review_text` → `true_text`.

---

## Prompts

- System + user templates: `app/prompts/templates.py`
- LLM config: `app/core/config.py` → `LLM_MODEL`, `LLM_MAX_TOKENS`

---

## EDA deliverable

- Notebook: `notebooks/eda.ipynb`
- Cached samples: `data/raw/*_reviews_sample.parquet`
- Plots + summary: `notebooks/outputs/` (run `python scripts/run_eda.py`)

**Critical check (§D):** % of users with ≥ `HOLDOUT_LAST_N + 1` reviews — see `eda_summary.json`.

---

## MVP scope (agreed default)

1. Amazon **Books** only
2. Predict last review (rating + text)
3. Metrics: MAE/RMSE + ROUGE; optional LLM judge (`app/eval/judge.py`)
4. Streamlit demo

Stretch: Electronics, Goodreads, richer retrieval eval.
