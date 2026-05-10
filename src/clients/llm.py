"""Anthropic Claude reasoning engine.

Two functions:
  - score_catalyst: per-headline Noise vs Structural classifier (legacy, kept).
  - synthesize_verdict: per-ticker fusion of fundamentals + news + sentiment
    into a single Bullish/Hold/Bearish call with confidence + thesis.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.config import ANALYST_SYSTEM_PROMPT, SETTINGS

MODEL = "claude-sonnet-4-6"


@dataclass
class CatalystVerdict:
    classification: str  # "Noise" or "Structural"
    score: int  # -10..+10
    rationale: str


@dataclass
class TickerVerdict:
    symbol: str
    rating: str  # "Bullish" | "Hold" | "Bearish"
    confidence: float  # 0..1
    catalyst_score: int  # -10..+10 (net of recent news)
    thesis: str  # 1-2 sentence rationale
    drivers: list[str]  # short bullets of the main factors


def is_configured() -> bool:
    return bool(SETTINGS.anthropic_api_key)


def _client():
    from anthropic import Anthropic
    return Anthropic(api_key=SETTINGS.anthropic_api_key)


def _extract_json(text: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _system_block() -> list[dict]:
    return [
        {
            "type": "text",
            "text": ANALYST_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _text(msg) -> str:
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


# ----- legacy per-headline scorer (kept for compatibility) -----

def score_catalyst(headline: str, description: str | None, tickers: list[str]) -> CatalystVerdict | None:
    if not is_configured():
        return None

    instruction = (
        "Classify the following news headline+description as 'Noise' or "
        "'Structural Trend Change'. Ignore general macro news unless it affects "
        "10-year yields. Output JSON only: "
        '{"classification": "Noise"|"Structural", "score": <int -10..10>, '
        '"rationale": "<one sentence>"}.'
    )
    user = (
        f"Tickers: {', '.join(tickers) or 'n/a'}\n"
        f"Headline: {headline}\n"
        f"Description: {description or '(none)'}\n\n"
        f"{instruction}"
    )
    try:
        msg = _client().messages.create(
            model=MODEL,
            max_tokens=300,
            system=_system_block(),
            messages=[{"role": "user", "content": user}],
        )
    except Exception:
        return None

    parsed = _extract_json(_text(msg))
    if not parsed:
        return None
    return CatalystVerdict(
        classification=str(parsed.get("classification", "Noise")),
        score=int(parsed.get("score", 0)),
        rationale=str(parsed.get("rationale", "")),
    )


# ----- per-ticker synthesis -----

def _format_news(news: list[dict]) -> str:
    if not news:
        return "(no recent ticker-tagged news)"
    lines = []
    for n in news[:5]:
        when = n.get("published_at", "")
        title = n.get("title", "").strip()
        desc = (n.get("description") or "").strip()
        if desc:
            desc = desc[:240] + ("…" if len(desc) > 240 else "")
        lines.append(f"- [{when}] {title}\n  {desc}" if desc else f"- [{when}] {title}")
    return "\n".join(lines)


def synthesize_verdict(
    symbol: str,
    role: str,
    price: float | None,
    change_pct: float | None,
    analyst_consensus: float | None,
    analyst_high: float | None,
    analyst_low: float | None,
    news: list[dict],
    sentiment_score: float | None,
    sentiment_summary: str | None,
) -> TickerVerdict | None:
    """Fuse all signals for one ticker into a single Bullish/Hold/Bearish call."""
    if not is_configured():
        return None

    upside = None
    if analyst_consensus and price:
        upside = (analyst_consensus - price) / price * 100

    price_str = f"{price}" if price is not None else "n/a"
    delta_str = f"{change_pct:+.2f}%" if change_pct is not None else "n/a"
    body = f"Ticker: ${symbol}  ({role})\nPrice: {price_str}  Today Δ: {delta_str}\n"

    if analyst_consensus is not None:
        if upside is not None:
            body += (
                f"Analyst consensus: {analyst_consensus}  "
                f"(high {analyst_high}, low {analyst_low})  "
                f"Implied upside: {upside:+.1f}%\n"
            )
        else:
            body += f"Analyst consensus: {analyst_consensus}\n"
    else:
        body += "Analyst consensus: n/a\n"

    body += "\nRecent news:\n" + _format_news(news) + "\n"

    if sentiment_score is not None:
        body += (
            f"\nX sentiment (verified financial accounts): "
            f"{sentiment_score:+.2f}\n"
            f"Sentiment note: {sentiment_summary or '(none)'}\n"
        )
    else:
        body += "\nX sentiment: unavailable\n"

    body += (
        "\nProduce a verdict for the next 1-3 weeks using ALL of the above. "
        "Weight structural catalysts (foundry capacity, HBM, hyperscaler CapEx) over "
        "short-term price action. Output JSON only:\n"
        '{"rating": "Bullish"|"Hold"|"Bearish", '
        '"confidence": <float 0..1>, '
        '"catalyst_score": <int -10..10>, '
        '"thesis": "<1-2 sentences>", '
        '"drivers": ["<short bullet>", "<short bullet>", "<short bullet>"]}'
    )

    try:
        msg = _client().messages.create(
            model=MODEL,
            max_tokens=600,
            system=_system_block(),
            messages=[{"role": "user", "content": body}],
        )
    except Exception:
        return None

    parsed = _extract_json(_text(msg))
    if not parsed:
        return None

    rating = str(parsed.get("rating", "Hold")).strip().title()
    if rating not in ("Bullish", "Hold", "Bearish"):
        rating = "Hold"

    return TickerVerdict(
        symbol=symbol,
        rating=rating,
        confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
        catalyst_score=int(parsed.get("catalyst_score", 0)),
        thesis=str(parsed.get("thesis", "")),
        drivers=[str(d) for d in (parsed.get("drivers") or []) if d],
    )
