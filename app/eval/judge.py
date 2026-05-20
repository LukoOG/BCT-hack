"""
judge.py — LLM-as-judge for qualitative review-generation scoring.

We score three axes on a 1-5 scale:
  - style_match    : does the generated review sound like *this* user?
  - faithfulness   : does it accurately reflect the target product?
  - coherence      : is it well-written prose?

The LLM call itself is wired up later by the runner; this file only owns the
prompt and a thin parser. Keeping the prompt here means evaluation behaviour
is reviewable in version control.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Optional


JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair evaluator of generated product reviews. "
    "You return JSON only, never prose."
)


JUDGE_PROMPT_TEMPLATE = """You are evaluating a generated product review against a ground truth.

## User's past reviews (style reference)
{user_history}

## Actual review the user wrote (ground truth)
Rating: {reference_rating}
Text: {reference_text}

## Generated review (under evaluation)
Rating: {generated_rating}
Text: {generated_text}

Score 1-5 on each axis. Be honest — a 3 is "okay", a 5 is "excellent and indistinguishable from the real review".

- style_match   : does it sound like this specific user wrote it (tone, length, vocabulary)?
- faithfulness  : does it discuss the product accurately and consistently with the reference?
- coherence     : is it well-written and internally consistent?

Return EXACTLY this JSON shape, with no surrounding text:
{{"style_match": <int>, "faithfulness": <int>, "coherence": <int>, "notes": "<one short sentence>"}}
"""


def render_judge_prompt(
    user_history: Iterable[dict],
    reference: dict,
    generated: dict,
    history_char_budget: int = 1500,
) -> str:
    """
    Render the judge prompt.

    user_history : iterable of {"rating": int, "text": str}
    reference    : {"rating": int, "text": str}
    generated    : {"rating": int, "text": str}
    """
    history_lines = []
    used = 0
    for r in user_history:
        snippet = f"  ★{r.get('rating', '?')}  {(r.get('text') or '')[:300]}"
        if used + len(snippet) > history_char_budget:
            break
        history_lines.append(snippet)
        used += len(snippet)

    return JUDGE_PROMPT_TEMPLATE.format(
        user_history="\n".join(history_lines) or "  (no history)",
        reference_rating=reference.get("rating", "?"),
        reference_text=(reference.get("text") or "")[:1000],
        generated_rating=generated.get("rating", "?"),
        generated_text=(generated.get("text") or "")[:1000],
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(raw: str) -> Optional[dict]:
    """
    Extract the JSON object from a judge response. Returns None if unparseable.
    Clamps each numeric score into [1, 5] for safety.
    """
    if not raw:
        return None
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    for key in ("style_match", "faithfulness", "coherence"):
        if key in payload:
            try:
                payload[key] = max(1, min(5, int(payload[key])))
            except (TypeError, ValueError):
                payload[key] = None
    return payload
