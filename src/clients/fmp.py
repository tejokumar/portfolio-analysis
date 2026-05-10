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


@dataclass
class EconomicEvent:
    date: str  # ISO datetime (UTC)
    country: str
    event: str
    impact: str | None  # "Low" | "Medium" | "High"
    actual: str | None
    estimate: str | None
    previous: str | None


@dataclass
class EarningsEvent:
    date: str
    symbol: str
    eps_estimate: float | None
    eps_actual: float | None
    revenue_estimate: float | None
    revenue_actual: float | None
    time: str | None  # "bmo" | "amc" | None


@dataclass
class CompanyProfile:
    symbol: str
    name: str | None
    sector: str | None
    industry: str | None
    market_cap: float | None
    beta: float | None
    description: str | None
    website: str | None
    ceo: str | None
    country: str | None


@dataclass
class Ratios:
    symbol: str
    pe: float | None
    price_to_sales: float | None
    price_to_book: float | None
    ev_to_ebitda: float | None
    debt_to_equity: float | None
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    dividend_yield: float | None
    current_ratio: float | None


def is_configured() -> bool:
    return bool(SETTINGS.fmp_api_key)


def _client() -> httpx.Client:
    return httpx.Client(timeout=15.0)


def fetch_quotes(symbols: list[str]) -> dict[str, Quote]:
    """Fetch real-time quotes via the stable endpoint (per-symbol, free-tier friendly)."""
    if not is_configured() or not symbols:
        return {}
    out: dict[str, Quote] = {}
    with _client() as c:
        for sym in symbols:
            try:
                r = c.get(
                    f"{STABLE_URL}/quote",
                    params={"symbol": sym, "apikey": SETTINGS.fmp_api_key},
                )
                r.raise_for_status()
                rows = r.json() or []
            except httpx.HTTPError:
                continue
            if not rows:
                continue
            row = rows[0]
            out[sym] = Quote(
                symbol=sym,
                price=float(row.get("price") or 0),
                change_pct=float(row.get("changePercentage") or row.get("changesPercentage") or 0),
                market_cap=row.get("marketCap"),
            )
    return out


def fetch_economic_calendar(
    from_date: str, to_date: str, us_high_impact_only: bool = True
) -> list[EconomicEvent]:
    """Fetch the economic calendar for a date range (YYYY-MM-DD).

    Defaults to US high/medium-impact events to keep the briefing focused.
    """
    if not is_configured():
        return []
    try:
        with _client() as c:
            r = c.get(
                f"{STABLE_URL}/economic-calendar",
                params={
                    "from": from_date, "to": to_date,
                    "apikey": SETTINGS.fmp_api_key,
                },
            )
            r.raise_for_status()
            rows = r.json() or []
    except httpx.HTTPError:
        return []

    out: list[EconomicEvent] = []
    for row in rows:
        country = (row.get("country") or "").upper()
        impact = row.get("impact") or row.get("importance")
        if us_high_impact_only:
            if country not in ("US", "UNITED STATES", "USD"):
                continue
            if impact and str(impact).lower() not in ("high", "medium"):
                continue
        out.append(
            EconomicEvent(
                date=row.get("date", ""),
                country=country,
                event=row.get("event") or row.get("name") or "",
                impact=str(impact) if impact else None,
                actual=str(row["actual"]) if row.get("actual") is not None else None,
                estimate=str(row["estimate"]) if row.get("estimate") is not None else None,
                previous=str(row["previous"]) if row.get("previous") is not None else None,
            )
        )
    out.sort(key=lambda e: e.date)
    return out


def fetch_earnings_calendar(from_date: str, to_date: str) -> list[EarningsEvent]:
    """Fetch the earnings calendar (all reporting tickers) for a date range."""
    if not is_configured():
        return []
    try:
        with _client() as c:
            r = c.get(
                f"{STABLE_URL}/earnings-calendar",
                params={
                    "from": from_date, "to": to_date,
                    "apikey": SETTINGS.fmp_api_key,
                },
            )
            r.raise_for_status()
            rows = r.json() or []
    except httpx.HTTPError:
        return []

    out: list[EarningsEvent] = []
    for row in rows:
        out.append(
            EarningsEvent(
                date=row.get("date", ""),
                symbol=row.get("symbol", ""),
                eps_estimate=row.get("epsEstimated"),
                eps_actual=row.get("eps"),
                revenue_estimate=row.get("revenueEstimated"),
                revenue_actual=row.get("revenue"),
                time=(row.get("time") or "").lower() or None,
            )
        )
    out.sort(key=lambda e: (e.date, e.symbol))
    return out


def fetch_profile(symbol: str) -> CompanyProfile | None:
    """Company description, sector, market cap, beta, etc."""
    if not is_configured() or not symbol:
        return None
    try:
        with _client() as c:
            r = c.get(
                f"{STABLE_URL}/profile",
                params={"symbol": symbol, "apikey": SETTINGS.fmp_api_key},
            )
            r.raise_for_status()
            rows = r.json() or []
    except httpx.HTTPError:
        return None
    if not rows:
        return None
    row = rows[0]
    return CompanyProfile(
        symbol=row.get("symbol", symbol),
        name=row.get("companyName"),
        sector=row.get("sector"),
        industry=row.get("industry"),
        market_cap=row.get("mktCap") or row.get("marketCap"),
        beta=row.get("beta"),
        description=row.get("description"),
        website=row.get("website"),
        ceo=row.get("ceo"),
        country=row.get("country"),
    )


def fetch_ratios_ttm(symbol: str) -> Ratios | None:
    """Trailing-twelve-month valuation/profitability ratios."""
    if not is_configured() or not symbol:
        return None
    try:
        with _client() as c:
            r = c.get(
                f"{STABLE_URL}/ratios-ttm",
                params={"symbol": symbol, "apikey": SETTINGS.fmp_api_key},
            )
            r.raise_for_status()
            rows = r.json() or []
    except httpx.HTTPError:
        return None
    if not rows:
        return None
    row = rows[0]
    return Ratios(
        symbol=symbol,
        pe=row.get("priceToEarningsRatioTTM"),
        price_to_sales=row.get("priceToSalesRatioTTM"),
        price_to_book=row.get("priceToBookRatioTTM"),
        ev_to_ebitda=row.get("enterpriseValueMultipleTTM"),
        debt_to_equity=row.get("debtToEquityRatioTTM"),
        gross_margin=row.get("grossProfitMarginTTM"),
        operating_margin=row.get("operatingProfitMarginTTM"),
        net_margin=row.get("netProfitMarginTTM"),
        dividend_yield=row.get("dividendYieldTTM"),
        current_ratio=row.get("currentRatioTTM"),
    )


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
