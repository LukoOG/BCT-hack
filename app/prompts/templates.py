"""
templates.py — Prompt templates for the review-generation step.

The generation contract:
  Input  : user history + target item metadata + retrieved similar reviews
  Output : JSON {"rating": 1-5, "title": "...", "text": "..."}

Keeping all prompts in one module so they're easy to A/B test and review.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional


_CATEGORY_HINTS = {
    "Books": (
        "Category: Books. Reviews are often longer (median ~60+ words). "
        "Match literary tone; mention plot, writing, or characters when relevant."
    ),
    "Electronics": (
        "Category: Electronics. Reviews are usually shorter (median ~30 words). "
        "Be direct about build quality, features, and value."
    ),
}

SYSTEM_PROMPT = (
    "You are a review-writing assistant. "
    "Given a user's past reviews and similar reviews of a target product, "
    "generate the review this specific user would most likely write next. "
    "Match their tone, typical length, and rating tendencies. "
    "Output JSON ONLY in the shape: "
    '{"rating": <int 1-5>, "title": "<string>", "text": "<string>"}'
)


def system_prompt_for_category(category: str) -> str:
    hint = _CATEGORY_HINTS.get(category, "")
    return f"{SYSTEM_PROMPT}\n\n{hint}" if hint else SYSTEM_PROMPT


USER_PROMPT_TEMPLATE = """## User's recent reviews
{user_history}

## Target product
Title:       {item_title}
Category:    {item_category}
Description: {item_description}

## Similar reviews from other users
{retrieved_reviews}

Generate the review this user would write. JSON only.
"""


def _format_history(user_history: Iterable[Mapping], char_budget: int = 1800) -> str:
    """Compact past-reviews block: star rating + truncated text, one per line."""
    lines, used = [], 0
    for r in user_history:
        rating = r.get("rating", "?")
        text   = (r.get("text") or r.get("review_text") or "")[:300]
        snippet = f"★{rating}  {text}"
        if used + len(snippet) > char_budget:
            break
        lines.append(snippet)
        used += len(snippet)
    return "\n---\n".join(lines) if lines else "(no prior reviews available)"


def _format_retrieved(retrieved: Iterable[Mapping], char_budget: int = 1200) -> str:
    """Compact retrieved-similar-reviews block."""
    lines, used = [], 0
    for r in retrieved:
        rating = r.get("rating", "?")
        text   = (r.get("text") or r.get("review_text") or "")[:200]
        snippet = f"★{rating}  {text}"
        if used + len(snippet) > char_budget:
            break
        lines.append(snippet)
        used += len(snippet)
    return "\n---\n".join(lines) if lines else "(no similar reviews found)"


def render_user_prompt(
    user_history: Iterable[Mapping],
    item_meta: Mapping,
    retrieved: Optional[Iterable[Mapping]] = None,
) -> str:
    """
    Build the user-facing prompt sent alongside SYSTEM_PROMPT.

    user_history : iterable of {"rating", "text" | "review_text"}
    item_meta    : {"title", "category", "description"} (extra keys ignored)
    retrieved    : iterable of similar reviews (same shape as user_history); optional
    """
    return USER_PROMPT_TEMPLATE.format(
        user_history=_format_history(user_history),
        item_title=item_meta.get("title", "(unknown)"),
        item_category=item_meta.get("category", "(unknown)"),
        item_description=(item_meta.get("description") or "")[:500],
        retrieved_reviews=_format_retrieved(retrieved or []),
    )
