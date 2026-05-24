"""
FastAPI server for the review predictor — architect integration surface.

Run:
    uvicorn app.api.server:app --reload
    python scripts/run_api.py
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core import config
from app.core.llm import llm_status
from app.pipeline import predict_next_review, recommend_items
from app.pipeline.embeddings import active_backend_name
from app.tasks.task_b import recommend_items
from app.team_contract import EVAL_PREDICTION_KEYS, PREDICT_NEXT_REVIEW_DOC
from app.api.catalog import router as catalog_router
from app.api.serialization import sanitize

app = FastAPI(
    title="BCT Review Predictor API",
    description="HTTP wrapper around predict_next_review for architect integrations.",
    version="0.2.0",
)


class PredictRequest(BaseModel):
    user_id: str = Field(..., examples=["amz_AFSKPY37N3C43SOI5IEXEK5JSIYA"])
    category: str = Field(default="Pet_Supplies", examples=["Pet_Supplies"])
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


class RecommendRequest(BaseModel):
    user_id: str = Field(..., examples=["amz_AFSKPY37N3C43SOI5IEXEK5JSIYA"])
    category: str = Field(default="Books", examples=["Books", "Electronics"])
    k: int = Field(default=10, ge=1, le=50)
    candidate_item_ids: set[str] | None = Field(
        default=None,
        description="Optional candidate pool for offline ranking/evaluation",
    )


class RecommendResponse(BaseModel):
    recommendations: list[dict[str, Any]]
    profile: Optional[dict[str, Any]] = None
    meta: Optional[dict[str, Any]] = None


@app.get("/health")
def health() -> dict[str, str]:
    llm = llm_status()
    return {
        "status": "ok",
        "embedding_backend": active_backend_name(),
        "llm": llm["llm"],
        "llm_provider": llm["provider"],
        "llm_model": llm["model"],
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
        return sanitize(predict_next_review(
            body.user_id,
            body.category,
            target_item_id=body.target_item_id,
        ))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest) -> dict[str, Any]:
    if body.category not in config.AMAZON_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category {body.category!r}. Choose from {config.AMAZON_CATEGORIES}",
        )
    try:
        return sanitize(recommend_items(
            body.user_id,
            body.category,
            k=body.k,
            seen_item_ids=body.candidate_item_ids,
        ))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

app.include_router(catalog_router)