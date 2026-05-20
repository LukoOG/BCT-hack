"""
Optional LLM generation — uses Anthropic when ANTHROPIC_API_KEY is set.

Without a key, returns None and callers should use the pipeline stub.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Optional

from app.core.config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE
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


def generate_review(
    user_history: list[Mapping],
    item_meta: Mapping,
    retrieved: Optional[list[Mapping]] = None,
    category: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Call Claude to generate a review. Returns None if no API key or on failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    user_prompt = render_user_prompt(user_history, item_meta, retrieved)
    cat = category or item_meta.get("category") or "Books"
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        system=system_prompt_for_category(str(cat)),
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = msg.content[0].text if msg.content else ""
    return _parse_json_response(text)
