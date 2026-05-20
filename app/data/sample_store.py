"""Load cached HF samples and demo user lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from app.core import config
from app.core import constants as C


def sample_path(category: str) -> Path:
    return config.DATA_RAW_DIR / f"{category.lower()}_reviews_sample.parquet"


def demo_users_path(category: str) -> Path:
    return config.DATA_RAW_DIR / f"{category.lower()}_demo_users.json"


def has_sample(category: str) -> bool:
    return sample_path(category).exists()


def load_sample(category: str) -> pd.DataFrame:
    path = sample_path(category)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. From repo root run: python scripts/fetch_samples.py"
        )
    return pd.read_parquet(path)


def load_demo_users(category: str, limit: int = 50) -> List[str]:
    path = demo_users_path(category)
    if path.exists():
        users = json.loads(path.read_text(encoding="utf-8"))
        return users[:limit]
    df = load_sample(category)
    counts = df[C.F_USER_ID].value_counts()
    needed = config.HOLDOUT_LAST_N + 1
    return counts[counts >= needed].head(limit).index.astype(str).tolist()


def default_demo_user(category: str) -> Optional[str]:
    users = load_demo_users(category, limit=1)
    return users[0] if users else None
