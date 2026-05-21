"""
Optional LLM generation — Groq (default) or Anthropic when an API key is set.

Without a key, returns None and callers should use the FAISS heuristic baseline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from app.core.config import LLM_MAX_TOKENS, LLM_TEMPERATURE
from app.core.llm import llm_api_key, llm_configured, llm_model_name, llm_provider
from app.prompts.templates import render_user_prompt, system_prompt_for_category

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_response(raw: str) -> Optional[dict]:
    if not raw:
        return None
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_groq(system: str, user_prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=llm_api_key())
    completion = client.chat.completions.create(
        model=llm_model_name(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    return completion.choices[0].message.content or ""


def _call_anthropic(system: str, user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=llm_api_key())
    msg = client.messages.create(
        model=llm_model_name(),
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return msg.content[0].text if msg.content else ""


def generate_review(
    user_history: list[Mapping],
    item_meta: Mapping,
    retrieved: Optional[list[Mapping]] = None,
    category: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Call Groq or Anthropic to generate a review. Returns None if no API key or on failure.
    """
    if not llm_configured():
        return None

    user_prompt = render_user_prompt(user_history, item_meta, retrieved)
    cat = category or item_meta.get("category") or "Books"
    system = system_prompt_for_category(str(cat))

    try:
        if llm_provider() == "anthropic":
            text = _call_anthropic(system, user_prompt)
        else:
            text = _call_groq(system, user_prompt)
    except ImportError:
        return None
    except Exception:
        return None

    return _parse_json_response(text)
