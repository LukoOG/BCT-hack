"""
serialization.py — Convert numpy / pandas types to plain Python before
                   FastAPI/Pydantic serializes the response.
"""

from __future__ import annotations

import math
from typing import Any


def sanitize(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas types to plain Python primitives.

    Handles:
        np.ndarray          → list (recursively sanitized)
        np.integer          → int
        np.floating         → float  (NaN/Inf → None)
        np.bool_            → bool
        np.str_             → str
        pd.Series           → list
        pd.DataFrame        → list of dicts
        dict                → dict  (keys and values sanitized)
        list / tuple        → list  (elements sanitized)
        float (NaN/Inf)     → None  (JSON has no NaN)
        everything else     → unchanged
    """
    # ── numpy scalars ──────────────────────────────────────────────────────────
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return [sanitize(v) for v in obj.tolist()]

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, np.str_):
            return str(obj)

    except ImportError:
        pass  # numpy not installed — nothing to do

    # ── pandas types ───────────────────────────────────────────────────────────
    try:
        import pandas as pd

        if isinstance(obj, pd.Series):
            return [sanitize(v) for v in obj.tolist()]

        if isinstance(obj, pd.DataFrame):
            return [sanitize(row) for row in obj.to_dict(orient="records")]

        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        if isinstance(obj, pd.NA.__class__):
            return None

    except ImportError:
        pass

    # ── plain Python float edge cases ──────────────────────────────────────────
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    # ── containers ─────────────────────────────────────────────────────────────
    if isinstance(obj, dict):
        return {sanitize(k): sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]

    return obj
