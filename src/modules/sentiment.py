"""Sentiment Divergence: flag Buy-the-Dip when price is down but Grok sentiment is high."""
from __future__ import annotations

from dataclasses import dataclass

from src.clients.fmp import Quote
from src.clients.grok import SentimentReading


@dataclass
class DivergenceSignal:
    symbol: str
    price_change_pct: float
    sentiment_score: float
    sentiment_summary: str
    flag: str  # "Buy the Dip", "Topping", or "Aligned"


def _flag(price_change_pct: float, sentiment_score: float) -> str:
    # Price down >1% but sentiment among verified accounts >= +0.3 → BTD.
    if price_change_pct < -1.0 and sentiment_score >= 0.3:
        return "Buy the Dip"
    # Price up >1% but sentiment <= -0.3 → smart money fading the rally.
    if price_change_pct > 1.0 and sentiment_score <= -0.3:
        return "Topping"
    return "Aligned"


def detect_divergence(
    quotes: dict[str, Quote], readings: dict[str, SentimentReading]
) -> list[DivergenceSignal]:
    out: list[DivergenceSignal] = []
    for sym, reading in readings.items():
        q = quotes.get(sym)
        if not q:
            continue
        out.append(
            DivergenceSignal(
                symbol=sym,
                price_change_pct=q.change_pct,
                sentiment_score=reading.score,
                sentiment_summary=reading.summary,
                flag=_flag(q.change_pct, reading.score),
            )
        )
    # Surface non-aligned signals first.
    out.sort(key=lambda s: 0 if s.flag == "Aligned" else 1, reverse=True)
    return out
