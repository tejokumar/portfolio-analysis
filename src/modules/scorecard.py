"""Per-ticker scorecard: gathers fundamentals + news + sentiment, asks Claude
to fuse them into a Bullish / Hold / Bearish verdict. Then combines the verdict
with the user's current holdings to recommend a concrete action (Add / Trim /
Hold / Exit / Initiate / Watch).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.clients.fmp import AnalystTarget, Quote
from src.clients.grok import SentimentReading
from src.clients.llm import TickerVerdict, synthesize_verdict
from src.clients.polygon import NewsItem
from src.clients.snaptrade import Holding
from src.config import REBALANCE_THRESHOLD, TARGET_ALLOCATION, TICKER_ROLES


@dataclass
class Recommendation:
    action: str        # "Add" | "Trim" | "Hold" | "Exit" | "Initiate" | "Watch"
    dollars: float     # signed trade size (+ = buy, − = sell, 0 = hold)
    shares: float      # signed share count
    rationale: str     # 1-line explanation


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
    current_weight: float        # 0-1
    target_weight: float         # 0-1
    current_value: float         # $ held today
    recommendation: Recommendation

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


def _recommend(
    rating: str,
    current_value: float,
    target_value: float,
    current_weight: float,
    target_weight: float,
    price: float | None,
    threshold: float = REBALANCE_THRESHOLD,
) -> Recommendation:
    """Combine a Bullish/Hold/Bearish verdict with the position's drift vs target
    into a concrete action."""
    drift = current_weight - target_weight  # >0 = over, <0 = under
    trade_dollars = target_value - current_value
    trade_shares = (trade_dollars / price) if price else 0.0

    # Bearish — exit or trim regardless of weight.
    if rating == "Bearish":
        if current_value <= 0:
            return Recommendation(
                action="Watch",
                dollars=0.0, shares=0.0,
                rationale="Bearish verdict; not held → skip entry.",
            )
        sell_shares = -(current_value / price) if price else 0.0
        return Recommendation(
            action="Exit" if current_weight > threshold else "Trim",
            dollars=-current_value,
            shares=sell_shares,
            rationale=f"Bearish verdict — reduce/exit ({current_weight:.1%} held).",
        )

    # Bullish or Hold, not held → initiate.
    if current_value <= 0 and target_weight > 0:
        return Recommendation(
            action="Initiate",
            dollars=trade_dollars,
            shares=trade_shares,
            rationale=f"{rating} verdict; not held → initiate to {target_weight:.0%}.",
        )

    # Within rebalance band → hold.
    if abs(drift) <= threshold:
        return Recommendation(
            action="Hold",
            dollars=0.0, shares=0.0,
            rationale=f"Within ±{threshold:.0%} of target ({current_weight:.1%}).",
        )

    # Over-weight.
    if drift > 0:
        if rating == "Bullish":
            return Recommendation(
                action="Hold",
                dollars=0.0, shares=0.0,
                rationale=f"Over-weight {drift:+.1%} but Bullish — let winners run.",
            )
        return Recommendation(
            action="Trim",
            dollars=trade_dollars,
            shares=trade_shares,
            rationale=f"Over-weight {drift:+.1%} — trim back to target.",
        )

    # Under-weight (drift < 0).
    return Recommendation(
        action="Add",
        dollars=trade_dollars,
        shares=trade_shares,
        rationale=f"Under-weight {drift:+.1%} — add to reach target.",
    )


def _news_for(symbol: str, all_news: list[NewsItem]) -> list[NewsItem]:
    return [n for n in all_news if symbol in (n.tickers or [])]


def _build_one(
    sym: str,
    quotes: dict[str, Quote],
    targets: dict[str, AnalystTarget],
    news: list[NewsItem],
    sentiments: dict[str, SentimentReading],
    held_value: float,
    portfolio_total: float,
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

    target_weight = TARGET_ALLOCATION.get(sym, 0.0)
    current_weight = (held_value / portfolio_total) if portfolio_total > 0 else 0.0
    target_value = target_weight * portfolio_total
    rating = verdict.rating if verdict else "Hold"

    recommendation = _recommend(
        rating=rating,
        current_value=held_value,
        target_value=target_value,
        current_weight=current_weight,
        target_weight=target_weight,
        price=q.price if q else None,
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
        current_weight=current_weight,
        target_weight=target_weight,
        current_value=held_value,
        recommendation=recommendation,
    )


def build_scorecards(
    symbols: list[str],
    quotes: dict[str, Quote],
    targets: dict[str, AnalystTarget],
    news: list[NewsItem],
    sentiments: dict[str, SentimentReading],
    holdings: list[Holding],
    max_workers: int = 8,
    progress_cb=None,
) -> list[Scorecard]:
    """Build a Scorecard per symbol. Parallelized — Claude calls fan out across threads."""
    held_by_sym = {h.symbol: h.market_value for h in holdings}
    portfolio_total = sum(h.market_value for h in holdings)

    total = len(symbols)
    cards: list[Scorecard] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _build_one,
                s, quotes, targets, news, sentiments,
                held_by_sym.get(s, 0.0),
                portfolio_total,
            ): s
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
                        current_weight=0.0, target_weight=TARGET_ALLOCATION.get(sym, 0.0),
                        current_value=held_by_sym.get(sym, 0.0),
                        recommendation=Recommendation(
                            "Hold", 0.0, 0.0, "Synthesis failed."
                        ),
                    )
                )
            done += 1
            if progress_cb:
                progress_cb(done, total, sym)

    # Order: actionable first (Initiate, Add, Trim, Exit), then Hold, then Watch.
    action_order = {
        "Initiate": 0, "Add": 1, "Trim": 2, "Exit": 3,
        "Hold": 4, "Watch": 5,
    }
    cards.sort(
        key=lambda c: (
            action_order.get(c.recommendation.action, 9),
            -(c.confidence or 0),
        )
    )
    return cards
