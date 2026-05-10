"""Polygon.io news client.

Cadence: every 15-30 min during market hours. Watch for breaking catalysts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx

from src.config import SETTINGS

BASE_URL = "https://api.polygon.io"


@dataclass
class NewsItem:
    id: str
    title: str
    publisher: str
    url: str
    published_at: datetime
    tickers: list[str]
    description: str | None = None


def is_configured() -> bool:
    return bool(SETTINGS.polygon_api_key)


def fetch_news(symbols: list[str], limit_per_symbol: int = 5) -> list[NewsItem]:
    if not is_configured() or not symbols:
        return []
    seen: set[str] = set()
    items: list[NewsItem] = []
    with httpx.Client(timeout=15.0) as c:
        for sym in symbols:
            try:
                r = c.get(
                    f"{BASE_URL}/v2/reference/news",
                    params={
                        "ticker": sym,
                        "limit": limit_per_symbol,
                        "order": "desc",
                        "sort": "published_utc",
                        "apiKey": SETTINGS.polygon_api_key,
                    },
                )
                r.raise_for_status()
                results = (r.json() or {}).get("results", []) or []
            except httpx.HTTPError:
                continue
            for row in results:
                rid = row.get("id")
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                items.append(
                    NewsItem(
                        id=rid,
                        title=row.get("title", ""),
                        publisher=(row.get("publisher") or {}).get("name", ""),
                        url=row.get("article_url", ""),
                        published_at=datetime.fromisoformat(
                            row["published_utc"].replace("Z", "+00:00")
                        ),
                        tickers=row.get("tickers") or [],
                        description=row.get("description"),
                    )
                )
    items.sort(key=lambda n: n.published_at, reverse=True)
    return items
