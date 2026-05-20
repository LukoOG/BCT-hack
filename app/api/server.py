"""
FastAPI server for the review predictor — architect integration surface.

Run:
    uvicorn app.api.server:app --reload
    python scripts/run_api.py
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core import config
from app.pipeline import predict_next_review
from app.pipeline.embeddings import active_backend_name
from app.team_contract import EVAL_PREDICTION_KEYS, PREDICT_NEXT_REVIEW_DOC

app = FastAPI(
    title="BCT Review Predictor API",
    description="HTTP wrapper around predict_next_review for architect integrations.",
    version="0.2.0",
)


class PredictRequest(BaseModel):
    user_id: str = Field(..., examples=["amz_AE3TASYGLHHRHUJUDFTKFDMWFIYA"])
    category: str = Field(default="Books", examples=["Books"])
    target_item_id: str | None = Field(
        default=None,
        description="Optional — for eval/holdout runs without leaking review text",
    )


class PredictResponse(BaseModel):
    user_history: list[dict[str, Any]]
    prediction: dict[str, Any]
    retrieved: list[dict[str, Any]]
    profile: Optional[dict[str, Any]] = None
    meta: Optional[dict[str, Any]] = None


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "embedding_backend": active_backend_name(),
        "llm": "on" if os.environ.get("ANTHROPIC_API_KEY") else "off",
    }


@app.get("/contract")
def contract() -> dict[str, Any]:
    return {
        "predict_next_review": PREDICT_NEXT_REVIEW_DOC.strip(),
        "eval_prediction_keys": EVAL_PREDICTION_KEYS,
        "categories": config.AMAZON_CATEGORIES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> dict[str, Any]:
    if body.category not in config.AMAZON_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category {body.category!r}. Choose from {config.AMAZON_CATEGORIES}",
        )
    try:
        return predict_next_review(
            body.user_id,
            body.category,
            target_item_id=body.target_item_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
