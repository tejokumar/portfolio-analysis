"""Per-ticker scorecard: gathers fundamentals + news + sentiment, asks Claude
to fuse them into a Bullish / Hold / Bearish verdict.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.clients.fmp import AnalystTarget, Quote
from src.clients.grok import SentimentReading
from src.clients.llm import TickerVerdict, synthesize_verdict
from src.clients.polygon import NewsItem
from src.config import TICKER_ROLES


@dataclass
class Scorecard:
    symbol: str
    role: str
    price: float | None
    change_pct: float | None
    analyst_consensus: float | None
    analyst_upside_pct: float | None
    sentiment_score: float | None
    sentiment_summary: str | None
    news: list[NewsItem]
    verdict: TickerVerdict | None

    @property
    def rating(self) -> str:
        return self.verdict.rating if self.verdict else "—"

    @property
    def confidence(self) -> float | None:
        return self.verdict.confidence if self.verdict else None

    @property
    def catalyst_score(self) -> int | None:
        return self.verdict.catalyst_score if self.verdict else None

    @property
    def thesis(self) -> str:
        return self.verdict.thesis if self.verdict else ""


def _news_for(symbol: str, all_news: list[NewsItem]) -> list[NewsItem]:
    return [n for n in all_news if symbol in (n.tickers or [])]


def _build_one(
    sym: str,
    quotes: dict[str, Quote],
    targets: dict[str, AnalystTarget],
    news: list[NewsItem],
    sentiments: dict[str, SentimentReading],
) -> Scorecard:
    q = quotes.get(sym)
    t = targets.get(sym)
    sent = sentiments.get(sym)
    ticker_news = _news_for(sym, news)

    upside = None
    if t and t.target_consensus and q and q.price:
        upside = (t.target_consensus - q.price) / q.price * 100

    news_payload = [
        {
            "title": n.title,
            "description": n.description,
            "published_at": n.published_at.strftime("%Y-%m-%d %H:%M"),
        }
        for n in ticker_news[:5]
    ]

    verdict = synthesize_verdict(
        symbol=sym,
        role=TICKER_ROLES.get(sym, ""),
        price=q.price if q else None,
        change_pct=q.change_pct if q else None,
        analyst_consensus=t.target_consensus if t else None,
        analyst_high=t.target_high if t else None,
        analyst_low=t.target_low if t else None,
        news=news_payload,
        sentiment_score=sent.score if sent else None,
        sentiment_summary=sent.summary if sent else None,
    )

    return Scorecard(
        symbol=sym,
        role=TICKER_ROLES.get(sym, ""),
        price=q.price if q else None,
        change_pct=q.change_pct if q else None,
        analyst_consensus=t.target_consensus if t else None,
        analyst_upside_pct=upside,
        sentiment_score=sent.score if sent else None,
        sentiment_summary=sent.summary if sent else None,
        news=ticker_news,
        verdict=verdict,
    )


def build_scorecards(
    symbols: list[str],
    quotes: dict[str, Quote],
    targets: dict[str, AnalystTarget],
    news: list[NewsItem],
    sentiments: dict[str, SentimentReading],
    max_workers: int = 8,
    progress_cb=None,
) -> list[Scorecard]:
    """Build a Scorecard per symbol. Parallelized — Claude calls fan out across threads.

    progress_cb: optional callable(done: int, total: int, current_symbol: str)
    """
    total = len(symbols)
    cards: list[Scorecard] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_build_one, s, quotes, targets, news, sentiments): s
            for s in symbols
        }
        for fut in futures:
            sym = futures[fut]
            try:
                cards.append(fut.result())
            except Exception:  # noqa: BLE001
                cards.append(
                    Scorecard(
                        symbol=sym,
                        role=TICKER_ROLES.get(sym, ""),
                        price=None, change_pct=None,
                        analyst_consensus=None, analyst_upside_pct=None,
                        sentiment_score=None, sentiment_summary=None,
                        news=[], verdict=None,
                    )
                )
            done += 1
            if progress_cb:
                progress_cb(done, total, sym)

    rating_order = {"Bullish": 0, "Hold": 1, "Bearish": 2, "—": 3}
    cards.sort(
        key=lambda c: (
            rating_order.get(c.rating, 9),
            -(c.confidence or 0),
        )
    )
    return cards
