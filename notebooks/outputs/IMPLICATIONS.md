## Implications for the pipeline (auto-generated from sample EDA)

- **Embedding chunk size (§B):** 95th percentile ≈ 589 words (99th ≈ 873). Consider chunking or truncating at ~400 words for embeddings.

- **Trainable users (§D, Books sample):** 210 / 371 users (56.6%) have ≥ 2 reviews for predict-last-review. **OK for the task** on this sample.

- **Rating skew (§C, Books):** 5.0★=64.6%, 4.0★=22.3%, 3.0★=7.8%. Report MAE **and** per-rating accuracy / macro-F1 in eval.

- **Temporal scope (§F):** 2000-07-21 → 2023-03-12. Consider filtering to 2015+ for demo relevance if older years dominate.

- **Books vs Electronics (§H):** Books n=3,000, Electronics n=3,000 in sample. Compare plots in `notebooks/outputs/H_books_vs_electronics.png`.

**Contract for architect:** embed per-review text; aggregate at retrieval time. `predict_next_review(user_id, category)` — see `TEAM_CONTRACT.md`.