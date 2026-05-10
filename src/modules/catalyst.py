"""Catalyst Filter: send news through Claude to score Noise vs Structural Trend Change."""
from __future__ import annotations

from dataclasses import dataclass

from src.clients.llm import CatalystVerdict, score_catalyst
from src.clients.polygon import NewsItem


@dataclass
class ScoredNews:
    item: NewsItem
    verdict: CatalystVerdict | None


def score_news(items: list[NewsItem], universe: set[str]) -> list[ScoredNews]:
    """Score each news item, restricting tickers to those we hold/target."""
    scored: list[ScoredNews] = []
    for item in items:
        relevant = [t for t in item.tickers if t in universe]
        if not relevant:
            continue
        verdict = score_catalyst(item.title, item.description, relevant)
        scored.append(ScoredNews(item=item, verdict=verdict))

    def sort_key(s: ScoredNews) -> tuple[int, int]:
        # Prioritize Structural over Noise, then by absolute score.
        if not s.verdict:
            return (0, 0)
        struct = 1 if s.verdict.classification.lower().startswith("struct") else 0
        return (struct, abs(s.verdict.score))

    scored.sort(key=sort_key, reverse=True)
    return scored
