"""Grok (xAI) client — X sentiment via the Agent Tools API.

Uses /v1/responses with the x_search tool. The model performs live X searches
and returns a JSON sentiment object. Cadence: 2x/day (open + close vibe check).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

import httpx

from src.config import SETTINGS

API_URL = "https://api.x.ai/v1/responses"
MODEL = "grok-4.3"


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
        "You are a quantitative sentiment analyst. Use the x_search tool to find "
        "posts from the last 24 hours about the requested ticker. Prefer verified "
        "financial accounts (named analysts, fund managers, industry insiders); "
        "discount retail hype, memes, and anonymous accounts. Output a single JSON "
        'object only, no prose: {"score": <float -1..1>, '
        '"summary": "<one sentence>", "n": <int sample size>}.'
    )


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a possibly-fenced text blob."""
    # Strip code fences if Grok wrapped the JSON.
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


def _final_text(payload: dict) -> str:
    """Walk the /v1/responses output array and return the assistant's final text."""
    for block in payload.get("output", []) or []:
        for piece in block.get("content", []) or []:
            if piece.get("type") == "output_text" and piece.get("text"):
                return piece["text"]
    return ""


@dataclass
class HotTicker:
    symbol: str
    side: str  # "Bullish" | "Bearish" | "Mixed"
    thesis: str


@dataclass
class FlowPost:
    symbol: str | None       # ticker, if one is identifiable
    side: str | None         # "Bullish"/"Bearish"/"Calls"/"Puts" if discernible
    summary: str             # the post's text / Grok's interpretation
    image_describes: str | None  # Grok's read of the attached image (chart, OI, etc.)
    has_image: bool
    conviction: int          # 1-10
    posted_at: str | None
    url: str | None


def fetch_hot_chatter() -> list[HotTicker]:
    """Single Grok x_search returning top US tickers being discussed on X.

    Theme-aware and sector-agnostic — Grok identifies the dominant rotations
    of the week (could be AI, energy, defense, biotech, anything) and returns
    the leading bullish/bearish names within those themes. Restricts to verified
    financial accounts; excludes meme/retail noise.
    """
    if not is_configured():
        return []

    system = (
        "You are a buy-side analyst scanning X for the most-discussed individual "
        "US stocks and ETFs in the last 12-24 hours. Use the x_search tool. "
        "First identify the dominant 2-3 sector/rotation themes capital is "
        "flowing into RIGHT NOW (let the chatter decide — could be AI, energy, "
        "defense, healthcare, biotech, financials, materials, consumer, etc.). "
        "Then return the leading bullish/bearish tickers within those themes. "
        "Restrict to verified financial accounts (named analysts, fund managers, "
        "industry insiders, financial journalists). Discount memes and anonymous "
        "retail. Aim for 6-10 entries spanning at least 2 different themes when "
        "the chatter supports it.\n\n"
        "Output JSON only, no prose: "
        '{"hot": [{"ticker": "<TICKER>", "side": "Bullish"|"Bearish"|"Mixed", '
        '"thesis": "<one sentence including which theme it fits>"}, ...]}.'
    )
    body = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "What are the hottest individual US tickers on X right now, "
                    "and what themes are they part of?"
                ),
            },
        ],
        "tools": [{"type": "x_search"}],
    }
    headers = {
        "Authorization": f"Bearer {SETTINGS.grok_api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=120.0) as c:
            r = c.post(API_URL, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        return []

    text = _final_text(data)
    parsed = _extract_json(text) if text else None
    if not parsed:
        return []
    items = parsed.get("hot") or []
    out: list[HotTicker] = []
    for it in items:
        sym = str(it.get("ticker") or "").strip().upper()
        side = str(it.get("side") or "").strip().title()
        thesis = str(it.get("thesis") or "").strip()
        if not sym or not thesis:
            continue
        if side not in ("Bullish", "Bearish", "Mixed"):
            side = "Mixed"
        out.append(HotTicker(symbol=sym, side=side, thesis=thesis))
    return out


FLOWGOD_HANDLE = "FL0WG0D"  # Note: zeros for O's


def fetch_flowgod_flow() -> list[FlowPost]:
    """Pull today's posts from @FL0WG0D on X.

    @FL0WG0D posts chart screenshots / open-interest images with brief text
    commentary. We use Grok's vision to inspect each image and extract what's
    being called out. Drop pure reactions, retweets, and posts without a ticker.
    """
    if not is_configured():
        return []

    today_iso = datetime.now().strftime("%Y-%m-%d")
    system = (
        f"Use the x_search tool with the query "
        f"`from:{FLOWGOD_HANDLE} since:{today_iso}` (mode Latest, limit 50) "
        f"to get every post from @{FLOWGOD_HANDLE} on X published today.\n\n"
        f"@{FLOWGOD_HANDLE} posts a mix of: chart screenshots with brief takes, "
        "open-interest tables, options flow callouts, and short reactions. "
        "For each post:\n"
        "  1. If it has an image, INSPECT it. Describe what's shown — chart of "
        "     which ticker? OI table for which ticker/strike/expiry? Flow row?\n"
        "  2. Identify a stock ticker (from text OR image). If no ticker is "
        "     discernible, SKIP the post.\n"
        "  3. Identify the side / direction if visible: Bullish, Bearish, Calls, "
        "     Puts. If unclear, leave null.\n"
        "  4. Compose a 1-line summary that captures: the ticker, what they're "
        "     pointing at, and any specific numbers (strikes, expiries, $ flow, "
        "     OI counts) visible in text or image.\n\n"
        "DROP these:\n"
        "  - Pure-emoji posts (\"🤣🤣\", \"👀\")\n"
        "  - One-word replies (\"No\", \"Higher\", \"Huh?\")\n"
        "  - Retweets without added value\n"
        "  - Posts with no identifiable ticker\n\n"
        "Conviction 1-10, weighted on:\n"
        "  - Image present with concrete data (chart, OI table): +3\n"
        "  - Explicit dollar amounts visible (e.g. '$20M buy'): +2\n"
        "  - SWEEP / BLOCK / UNUSUAL flagged: +2\n"
        "  - Specific strike + expiry given: +2\n"
        "  - Clear directional call: +1\n"
        "  Cap at 10, floor at 1.\n\n"
        "Output JSON only, no prose:\n"
        '{"flows": [{"ticker": "<TICKER>", '
        '"side": "Bullish"|"Bearish"|"Calls"|"Puts"|null, '
        '"summary": "<one-line description with concrete details>", '
        '"image_describes": "<what the image shows, or null>", '
        '"has_image": <bool>, '
        '"conviction": <1-10>, '
        '"posted_at": "<ISO timestamp or null>", '
        '"url": "<URL of the post or null>"}, ...]}'
    )

    body = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": "Fetch today's @flowgod options flow."},
        ],
        "tools": [{"type": "x_search"}],
    }
    headers = {
        "Authorization": f"Bearer {SETTINGS.grok_api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=120.0) as c:
            r = c.post(API_URL, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        return []

    text = _final_text(data)
    parsed = _extract_json(text) if text else None
    if not parsed:
        return []
    items = parsed.get("flows") or []
    out: list[FlowPost] = []
    for it in items:
        sym = str(it.get("ticker") or "").strip().upper() or None
        if not sym:
            continue
        side = it.get("side")
        if side:
            side = str(side).strip().title()
            if side not in ("Bullish", "Bearish", "Calls", "Puts"):
                side = None
        try:
            conviction = int(it.get("conviction", 5))
        except (TypeError, ValueError):
            conviction = 5
        conviction = max(1, min(10, conviction))
        out.append(
            FlowPost(
                symbol=sym,
                side=side,
                summary=str(it.get("summary") or "").strip(),
                image_describes=(
                    str(it["image_describes"]).strip()
                    if it.get("image_describes") else None
                ),
                has_image=bool(it.get("has_image", False)),
                conviction=conviction,
                posted_at=str(it["posted_at"]).strip() if it.get("posted_at") else None,
                url=str(it["url"]).strip() if it.get("url") else None,
            )
        )
    out.sort(key=lambda p: p.conviction, reverse=True)
    return out


def fetch_sentiment(symbol: str) -> SentimentReading | None:
    if not is_configured():
        return None

    body = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": f"Analyze X sentiment for ${symbol}."},
        ],
        "tools": [{"type": "x_search"}],
    }
    headers = {
        "Authorization": f"Bearer {SETTINGS.grok_api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=90.0) as c:
            r = c.post(API_URL, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError:
        return None

    text = _final_text(data)
    parsed = _extract_json(text) if text else None
    if not parsed:
        return None

    try:
        return SentimentReading(
            symbol=symbol,
            score=float(parsed.get("score", 0)),
            summary=str(parsed.get("summary", "")),
            sample_size=int(parsed.get("n", 0)),
        )
    except (ValueError, TypeError):
        return None
