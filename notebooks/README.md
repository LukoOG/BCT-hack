# Notebooks

| File | Purpose |
|------|---------|
| `eda.ipynb` | Exploratory analysis (sections A–H) |
| `evalution.ipynb` | (typo filename — evaluation notebook TBD) |

## Run EDA without Jupyter

From repo root:

```bash
python scripts/fetch_samples.py    # stream HF samples -> data/raw/*.parquet
python scripts/run_eda.py          # plots -> notebooks/outputs/
```

Then open `eda.ipynb` to explore interactively. Plots are already in `outputs/`.
