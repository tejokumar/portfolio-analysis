"""Financial Modeling Prep client — quotes and analyst price targets.

Cadence: 4x per day. Refresh price targets and major data points every ~2h.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.config import SETTINGS

BASE_URL = "https://financialmodelingprep.com/api/v3"
STABLE_URL = "https://financialmodelingprep.com/stable"


@dataclass
class Quote:
    symbol: str
    price: float
    change_pct: float
    market_cap: float | None = None


@dataclass
class AnalystTarget:
    symbol: str
    target_high: float | None
    target_low: float | None
    target_consensus: float | None


def is_configured() -> bool:
    return bool(SETTINGS.fmp_api_key)


def _client() -> httpx.Client:
    return httpx.Client(timeout=15.0)


def fetch_quotes(symbols: list[str]) -> dict[str, Quote]:
    if not is_configured() or not symbols:
        return {}
    joined = ",".join(symbols)
    with _client() as c:
        r = c.get(
            f"{BASE_URL}/quote/{joined}",
            params={"apikey": SETTINGS.fmp_api_key},
        )
        r.raise_for_status()
        rows = r.json() or []

    out: dict[str, Quote] = {}
    for row in rows:
        sym = row.get("symbol")
        if not sym:
            continue
        out[sym] = Quote(
            symbol=sym,
            price=float(row.get("price") or 0),
            change_pct=float(row.get("changesPercentage") or 0),
            market_cap=row.get("marketCap"),
        )
    return out


def fetch_analyst_targets(symbols: list[str]) -> dict[str, AnalystTarget]:
    if not is_configured() or not symbols:
        return {}
    out: dict[str, AnalystTarget] = {}
    with _client() as c:
        for sym in symbols:
            try:
                r = c.get(
                    f"{STABLE_URL}/price-target-consensus",
                    params={"symbol": sym, "apikey": SETTINGS.fmp_api_key},
                )
                r.raise_for_status()
                rows = r.json() or []
            except httpx.HTTPError:
                continue
            if not rows:
                continue
            row = rows[0]
            out[sym] = AnalystTarget(
                symbol=sym,
                target_high=row.get("targetHigh"),
                target_low=row.get("targetLow"),
                target_consensus=row.get("targetConsensus"),
            )
    return out
