"""LLM provider detection — Groq (default) or Anthropic."""

from __future__ import annotations

import os

from app.core import config


def llm_provider() -> str:
    return os.environ.get("LLM_PROVIDER", config.LLM_PROVIDER_DEFAULT).strip().lower()


def llm_api_key() -> str:
    if llm_provider() == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return os.environ.get("GROQ_API_KEY", "").strip()


def llm_configured() -> bool:
    return bool(llm_api_key())


def llm_model_name() -> str:
    override = os.environ.get("LLM_MODEL", "").strip()
    if override:
        return override
    if llm_provider() == "anthropic":
        return config.LLM_MODEL_ANTHROPIC
    return config.LLM_MODEL_GROQ


def llm_status() -> dict[str, str]:
    if not llm_configured():
        return {"llm": "off", "provider": llm_provider(), "model": ""}
    return {"llm": "on", "provider": llm_provider(), "model": llm_model_name()}
