"""Anthropic Claude reasoning engine.

Used by the Catalyst Filter: classifies news as Noise vs Structural Trend Change
and assigns a -10 to +10 catalyst score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from src.config import ANALYST_SYSTEM_PROMPT, SETTINGS

MODEL = "claude-sonnet-4-6"


@dataclass
class CatalystVerdict:
    classification: str  # "Noise" or "Structural"
    score: int  # -10..+10
    rationale: str


def is_configured() -> bool:
    return bool(SETTINGS.anthropic_api_key)


def _instruction() -> str:
    return (
        "Classify the following news headline+description as 'Noise' or "
        "'Structural Trend Change'. Ignore general macro news unless it affects "
        "10-year yields. Output JSON only: "
        '{"classification": "Noise"|"Structural", "score": <int -10..10>, '
        '"rationale": "<one sentence>"}.'
    )


def score_catalyst(headline: str, description: str | None, tickers: list[str]) -> CatalystVerdict | None:
    if not is_configured():
        return None

    from anthropic import Anthropic

    client = Anthropic(api_key=SETTINGS.anthropic_api_key)
    user = (
        f"Tickers: {', '.join(tickers) or 'n/a'}\n"
        f"Headline: {headline}\n"
        f"Description: {description or '(none)'}\n\n"
        f"{_instruction()}"
    )
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=[
                {
                    "type": "text",
                    "text": ANALYST_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
    except Exception:
        return None

    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    try:
        # Strip code fences if present.
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        return CatalystVerdict(
            classification=str(parsed.get("classification", "Noise")),
            score=int(parsed.get("score", 0)),
            rationale=str(parsed.get("rationale", "")),
        )
    except (ValueError, TypeError):
        return None
