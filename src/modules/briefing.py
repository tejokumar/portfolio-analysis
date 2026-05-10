"""Pre/Post-market briefings.

Two flavors:
  - "pre"  — ~15 min before US market open (09:15 ET / 06:15 PT). Sets the day:
             overnight news, earnings to watch, economic calendar, hot X chatter.
  - "post" — 1 hr after close (17:00 ET / 14:00 PT). Wraps the day: movers,
             post-close earnings, tomorrow's catalysts, end-of-day sentiment.

Pipeline:
  1. Parallel fan-out: Polygon news (portfolio + broad), FMP economic + earnings
     calendars, Grok hot-chatter.
  2. Single Claude synthesis call → markdown briefing.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from src.clients import fmp, grok, llm, polygon
from src.clients.fmp import EarningsEvent, EconomicEvent
from src.clients.grok import HotTicker
from src.clients.polygon import NewsItem
from src.config import MARKET_SYSTEM_PROMPT, SETTINGS, TICKER_ROLES

Mode = Literal["pre", "post"]


@dataclass
class Briefing:
    mode: Mode
    markdown: str  # full briefing rendered as markdown
    portfolio_news_count: int
    broad_news_count: int
    econ_count: int
    earnings_count: int
    hot_count: int


def _date_range(mode: Mode) -> tuple[str, str]:
    """Calendar window. Stretches across the upcoming few sessions so the user
    sees what's coming, not just one day."""
    today = date.today()
    if mode == "pre":
        return today.isoformat(), (today + timedelta(days=4)).isoformat()
    start = today + timedelta(days=1)
    return start.isoformat(), (start + timedelta(days=4)).isoformat()


def _filter_earnings(events: list[EarningsEvent], portfolio: set[str]) -> list[EarningsEvent]:
    """Portfolio reports always pass. Other names: require analyst coverage so
    we exclude obscure tickers. Cap at 40."""
    portfolio_hits = [e for e in events if e.symbol.upper() in portfolio]
    covered = [
        e for e in events
        if e.symbol.upper() not in portfolio
        and e.eps_estimate is not None
    ]
    return portfolio_hits + covered[:40]


def _fmt_news(items: list[NewsItem]) -> str:
    if not items:
        return "(none)"
    lines = []
    for n in items[:25]:
        when = n.published_at.strftime("%Y-%m-%d %H:%M UTC")
        tickers = ", ".join(n.tickers[:5]) if n.tickers else "—"
        title = (n.title or "").strip()
        desc = (n.description or "").strip()
        if desc:
            desc = desc[:200] + ("…" if len(desc) > 200 else "")
        body = f"- [{when}] [{tickers}] {title}"
        if desc:
            body += f"\n  {desc}"
        lines.append(body)
    return "\n".join(lines)


def _fmt_econ(events: list[EconomicEvent]) -> str:
    if not events:
        return "(none)"
    lines = []
    for e in events:
        impact = f" ({e.impact})" if e.impact else ""
        consensus = ""
        if e.estimate or e.actual or e.previous:
            consensus = (
                f" — actual {e.actual or '—'}, est {e.estimate or '—'}, "
                f"prev {e.previous or '—'}"
            )
        lines.append(f"- {e.date} {e.country} · {e.event}{impact}{consensus}")
    return "\n".join(lines)


def _fmt_earnings(events: list[EarningsEvent], portfolio: set[str]) -> str:
    if not events:
        return "(none)"
    lines = []
    for e in events:
        marker = " ⭐ portfolio" if e.symbol.upper() in portfolio else ""
        eps = f" est EPS {e.eps_estimate}" if e.eps_estimate is not None else ""
        when = f" ({e.time.upper()})" if e.time else ""
        lines.append(f"- {e.date} ${e.symbol}{when}{eps}{marker}")
    return "\n".join(lines)


def _fmt_hot(hot: list[HotTicker]) -> str:
    if not hot:
        return "(no chatter)"
    return "\n".join(
        f"- ${h.symbol} ({h.side}): {h.thesis}" for h in hot
    )


def _fetch_inputs(mode: Mode, portfolio_symbols: list[str]):
    """Parallel fan-out of news + calendars (hot chatter is passed in separately)."""
    portfolio_set = {s.upper() for s in portfolio_symbols}
    cal_from, cal_to = _date_range(mode)

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_port_news = ex.submit(polygon.fetch_news, portfolio_symbols, 5)
        f_broad_news = ex.submit(_fetch_broad_news)
        f_econ = ex.submit(fmp.fetch_economic_calendar, cal_from, cal_to)
        f_earn = ex.submit(fmp.fetch_earnings_calendar, cal_from, cal_to)

        port_news = f_port_news.result()
        broad_news = f_broad_news.result()
        econ = f_econ.result()
        earnings_raw = f_earn.result()

    earnings = _filter_earnings(earnings_raw, portfolio_set)
    return port_news, broad_news, econ, earnings


def _fetch_broad_news() -> list[NewsItem]:
    """Pull a broad market news sweep — uses bellwether ETFs as proxies."""
    return polygon.fetch_news(["SPY", "QQQ", "IWM", "VIX"], limit_per_symbol=4)


def _build_user_prompt(
    mode: Mode,
    portfolio_symbols: list[str],
    portfolio_roles: dict[str, str],
    port_news: list[NewsItem],
    broad_news: list[NewsItem],
    econ: list[EconomicEvent],
    earnings: list[EarningsEvent],
    hot: list[HotTicker],
) -> str:
    today = date.today().isoformat()
    portfolio_set = {s.upper() for s in portfolio_symbols}
    holdings_str = ", ".join(
        f"${s} ({portfolio_roles.get(s, '')})" for s in portfolio_symbols
    )

    if mode == "pre":
        framing = (
            "Write a PRE-MARKET briefing for today's session. The user reads this "
            "around 06:15 PT / 09:15 ET — 15 minutes before the US open. "
            "Frame the day: what's the dominant thesis right now, what could move "
            "the tape, what's specifically relevant to the portfolio."
        )
    else:
        framing = (
            "Write a POST-MARKET briefing — written 1 hour after today's close "
            "(14:00 PT / 17:00 ET). Wrap the day: how it played out, after-hours "
            "earnings news, what tomorrow looks like, end-of-day sentiment "
            "shifts. Frame the next session's risks/opportunities."
        )

    sections = (
        f"Date: {today}\n"
        f"Portfolio holdings (target list): {holdings_str}\n\n"
        f"=== PORTFOLIO-TAGGED NEWS (Polygon, ticker-filtered) ===\n"
        f"{_fmt_news(port_news)}\n\n"
        f"=== BROAD MARKET NEWS (SPY/QQQ/IWM/VIX-tagged) ===\n"
        f"{_fmt_news(broad_news)}\n\n"
        f"=== ECONOMIC CALENDAR (US, high/medium impact) ===\n"
        f"{_fmt_econ(econ)}\n\n"
        f"=== EARNINGS CALENDAR (portfolio + analyst-covered) ===\n"
        f"{_fmt_earnings(earnings, portfolio_set)}\n\n"
        f"=== HOT TICKERS ON X (verified financial accounts) ===\n"
        f"{_fmt_hot(hot)}\n"
    )

    instructions = f"""
{framing}

Output a single Markdown document with these sections (in this order):

## Market Thesis
2-4 sentences on the dominant macro/tape narrative right now. Specific, not generic.

## Top News
Bulleted list of the 5-8 most consequential headlines from the news above. Skip noise.
For each: ticker(s), 1-line summary, and your read on whether it's structural or noise.

## Portfolio Watch
For each portfolio ticker that has news or a near-term catalyst, give a 1-line
implication ("$NVDA: ... → watch the open"). Skip tickers with no relevant signal.

## Economic Calendar
Bullet the high-impact events from the calendar above with PT times in parentheses.
Highlight any that could reprice rates or tech multiples.

## Earnings to Watch
Bullet the most significant reports — portfolio names FIRST (mark with ⭐),
then 3-6 broader names that could move the tape. Include EPS estimates if known.

## Hot on X (Bullish)
Bullets from the X chatter list above where side == "Bullish". Keep theses tight.

## Hot on X (Bearish)
Same for bearish chatter.

## Session Game Plan
3-5 specific, action-oriented bullets the user should keep in mind during the
{'session' if mode == 'pre' else 'next session'}. Be concrete. No fluff.

Rules:
- Use \\$ for dollar signs (so Streamlit/Markdown does NOT treat them as math).
- Convert any ET times to PT in parentheses, e.g., "08:30 ET (05:30 PT)".
- If a section is genuinely empty, write "_No notable events._" and move on.
- Be concise. The whole brief should fit on a phone screen with light scrolling.
"""

    return f"{sections}\n{instructions}"


def generate_briefing(
    mode: Mode,
    portfolio_symbols: list[str],
    hot_chatter: list[HotTicker],
) -> Briefing:
    """Run the full pipeline and return a Briefing with rendered markdown."""
    port_news, broad_news, econ, earnings = _fetch_inputs(mode, portfolio_symbols)
    hot = hot_chatter

    if not llm.is_configured():
        return Briefing(
            mode=mode,
            markdown="_ANTHROPIC_API_KEY not configured — briefing synthesis disabled._",
            portfolio_news_count=len(port_news),
            broad_news_count=len(broad_news),
            econ_count=len(econ),
            earnings_count=len(earnings),
            hot_count=len(hot),
        )

    user = _build_user_prompt(
        mode, portfolio_symbols, TICKER_ROLES,
        port_news, broad_news, econ, earnings, hot,
    )

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=SETTINGS.anthropic_api_key)
        msg = client.messages.create(
            model=llm.MODEL,
            max_tokens=2500,
            system=[
                {
                    "type": "text",
                    "text": MARKET_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as exc:  # noqa: BLE001
        text = f"_Briefing synthesis failed: {type(exc).__name__}: {exc}_"

    return Briefing(
        mode=mode,
        markdown=text,
        portfolio_news_count=len(port_news),
        broad_news_count=len(broad_news),
        econ_count=len(econ),
        earnings_count=len(earnings),
        hot_count=len(hot),
    )
