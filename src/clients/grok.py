"""Grok (xAI) client — X sentiment with Live Search.

Cadence: 2x per day (open + close vibe check).
Uses xAI's OpenAI-compatible chat completions endpoint with the live_search tool
so Grok pulls fresh X posts from verified financial accounts.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.config import SETTINGS

API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-4-latest"


@dataclass
class SentimentReading:
    symbol: str
    score: float  # -1.0 (bearish) to +1.0 (bullish)
    summary: str
    sample_size: int


def is_configured() -> bool:
    return bool(SETTINGS.grok_api_key)


def _system_prompt() -> str:
    return (
        "You are a quantitative sentiment analyst. For the given ticker, search X "
        "for posts from the last 24 hours by VERIFIED financial accounts (analysts, "
        "fund managers, named industry insiders). Ignore retail hype, memes, and "
        "anonymous accounts. Output a JSON object only, no prose: "
        '{"score": <float -1..1>, "summary": "<one sentence>", "n": <int sample size>}.'
    )


def fetch_sentiment(symbol: str) -> SentimentReading | None:
    if not is_configured():
        return None

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": f"Analyze X sentiment for ${symbol}."},
        ],
        "search_parameters": {
            "mode": "on",
            "sources": [{"type": "x"}],
            "max_search_results": 25,
        },
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {SETTINGS.grok_api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=45.0) as c:
            r = c.post(API_URL, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        return None

    try:
        import json

        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return SentimentReading(
            symbol=symbol,
            score=float(parsed.get("score", 0)),
            summary=str(parsed.get("summary", "")),
            sample_size=int(parsed.get("n", 0)),
        )
    except (KeyError, ValueError, TypeError):
        return None
