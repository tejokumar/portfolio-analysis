"""Per-ticker deep analysis.

Fetches everything we have on a single ticker in parallel — FMP quote/target/
profile/ratios, Polygon news, Grok sentiment — and asks Claude to synthesize a
Bullish / Hold / Bearish thesis with concrete details. Reuses any cached
FlowGod / hot-chatter data the caller passes in.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.clients import fmp, grok, llm, polygon
from src.clients.fmp import AnalystTarget, CompanyProfile, Quote, Ratios
from src.clients.grok import FlowPost, HotTicker, SentimentReading
from src.clients.llm import TickerVerdict
from src.clients.polygon import NewsItem
from src.config import MARKET_SYSTEM_PROMPT, SETTINGS


VALID_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def normalize_ticker(raw: str) -> str | None:
    """Uppercase + validate. Returns None if it's clearly not a stock symbol."""
    if not raw:
        return None
    s = raw.strip().upper().lstrip("$")
    if not VALID_TICKER.match(s):
        return None
    return s


@dataclass
class TickerAnalysis:
    symbol: str
    quote: Quote | None
    target: AnalystTarget | None
    profile: CompanyProfile | None
    ratios: Ratios | None
    news: list[NewsItem]
    sentiment: SentimentReading | None
    flow_posts: list[FlowPost]      # filtered to this ticker
    hot_mentions: list[HotTicker]   # filtered to this ticker
    verdict: TickerVerdict | None
    report_markdown: str            # Claude's narrative report
    analyst_upside_pct: float | None


def _fmt_money(x) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"


def _fmt_pct(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x) * 100:+.1f}%" if abs(float(x)) < 5 else f"{float(x):+.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_ratio(x, decimals: int = 2) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _build_analyst_block(symbol: str, t: AnalystTarget | None, q: Quote | None) -> str:
    if not t:
        return f"Analyst targets for ${symbol}: not available."
    upside = None
    if t.target_consensus and q and q.price:
        upside = (t.target_consensus - q.price) / q.price * 100
    parts = [
        f"consensus {t.target_consensus or '—'}",
        f"high {t.target_high or '—'}",
        f"low {t.target_low or '—'}",
    ]
    if upside is not None:
        parts.append(f"implied upside {upside:+.1f}%")
    return "Analyst targets: " + ", ".join(parts)


def _build_profile_block(p: CompanyProfile | None) -> str:
    if not p:
        return "Company profile: not available."
    bits = [p.name]
    if p.sector: bits.append(p.sector)
    if p.industry: bits.append(p.industry)
    if p.market_cap: bits.append(f"market cap {_fmt_money(p.market_cap)}")
    if p.beta is not None: bits.append(f"beta {p.beta:.2f}")
    return "Company: " + " · ".join(b for b in bits if b)


def _build_ratios_block(r: Ratios | None) -> str:
    if not r:
        return "Valuation ratios: not available."
    pairs = [
        ("P/E", r.pe), ("P/S", r.price_to_sales), ("P/B", r.price_to_book),
        ("EV/EBITDA", r.ev_to_ebitda), ("D/E", r.debt_to_equity),
        ("Current ratio", r.current_ratio),
        ("Gross margin", r.gross_margin), ("Op margin", r.operating_margin),
        ("Net margin", r.net_margin), ("Div yield", r.dividend_yield),
    ]
    rendered = []
    for label, val in pairs:
        if val is None:
            continue
        if label in ("Gross margin", "Op margin", "Net margin", "Div yield"):
            rendered.append(f"{label} {_fmt_pct(val)}")
        else:
            rendered.append(f"{label} {_fmt_ratio(val)}")
    return "Ratios (TTM): " + " · ".join(rendered) if rendered else "Ratios: not available."


def _build_news_block(items: list[NewsItem]) -> str:
    if not items:
        return "(no recent news)"
    lines = []
    for n in items[:8]:
        when = n.published_at.strftime("%Y-%m-%d %H:%M UTC")
        title = (n.title or "").strip()
        desc = (n.description or "").strip()[:240]
        body = f"- [{when}] {title}"
        if desc:
            body += f"\n  {desc}"
        lines.append(body)
    return "\n".join(lines)


def _build_flow_block(flows: list[FlowPost]) -> str:
    if not flows:
        return "(no FlowGod activity captured for this ticker today)"
    lines = []
    for f in flows:
        bits = [f"[{f.conviction}/10]"]
        if f.side: bits.append(f.side)
        bits.append(f.summary)
        lines.append("- " + " · ".join(bits))
    return "\n".join(lines)


def _build_chatter_block(hot: list[HotTicker]) -> str:
    if not hot:
        return "(no hot-chatter mentions captured for this ticker)"
    return "\n".join(f"- {h.side}: {h.thesis}" for h in hot)


def _build_user_prompt(a: "TickerAnalysis") -> str:
    upside_str = (
        f"{a.analyst_upside_pct:+.1f}%"
        if a.analyst_upside_pct is not None else "n/a"
    )
    sentiment_block = "X sentiment: unavailable"
    if a.sentiment:
        sentiment_block = (
            f"X sentiment score: {a.sentiment.score:+.2f} "
            f"(sample size {a.sentiment.sample_size})\n"
            f"  → {a.sentiment.summary}"
        )

    return (
        f"Deep-dive analysis for ${a.symbol}.\n\n"
        f"=== COMPANY ===\n"
        f"{_build_profile_block(a.profile)}\n"
        f"{_build_ratios_block(a.ratios)}\n\n"
        f"=== PRICE & TARGETS ===\n"
        f"Price: {a.quote.price if a.quote else 'n/a'} "
        f"({a.quote.change_pct:+.2f}% today)" if a.quote else f"Price: n/a"
    ) + "\n" + (
        f"{_build_analyst_block(a.symbol, a.target, a.quote)}\n"
        f"Implied upside: {upside_str}\n\n"
        f"=== RECENT NEWS (Polygon) ===\n"
        f"{_build_news_block(a.news)}\n\n"
        f"=== X SENTIMENT (Grok, verified accounts) ===\n"
        f"{sentiment_block}\n\n"
        f"=== X CHATTER MENTIONS (today) ===\n"
        f"{_build_chatter_block(a.hot_mentions)}\n\n"
        f"=== OPTIONS FLOW (@FL0WG0D, today) ===\n"
        f"{_build_flow_block(a.flow_posts)}\n\n"
        "Produce a single Markdown analysis with these sections:\n\n"
        "## Verdict\n"
        "One line: **Bullish | Hold | Bearish** with a confidence (0-100%). "
        "Time-frame: next 1-3 weeks.\n\n"
        "## Thesis\n"
        "2-4 sentences. The core argument. Reference specific catalysts, "
        "metrics, or flow.\n\n"
        "## Bull Case\n"
        "Bulleted, 3-5 points. Concrete drivers.\n\n"
        "## Bear Case\n"
        "Bulleted, 3-5 points. Risks the bulls overlook.\n\n"
        "## Valuation Read\n"
        "1-2 sentences on whether the ratios + analyst targets suggest fair, "
        "rich, or cheap. Mention specific multiples.\n\n"
        "## What to Watch\n"
        "3-5 specific upcoming catalysts (earnings dates, macro events, "
        "product launches, OI patterns).\n\n"
        "## Bottom Line\n"
        "1-2 sentence action call. If Bullish → entry zone / position size "
        "thought. If Bearish → stop-out / hedging thought. If Hold → what "
        "would flip your view.\n\n"
        "Rules:\n"
        "- Escape dollar signs as \\$ for Streamlit markdown.\n"
        "- Be concrete. Cite the specific data points above.\n"
        "- Convert any ET times to PT in parens.\n"
    )


def _claude_synthesize(a: "TickerAnalysis") -> tuple[str, TickerVerdict | None]:
    """Call Claude for the long-form report AND a structured verdict for sorting."""
    if not llm.is_configured():
        return ("_ANTHROPIC_API_KEY not configured — synthesis disabled._", None)

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
            messages=[{"role": "user", "content": _build_user_prompt(a)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as exc:  # noqa: BLE001
        text = f"_Synthesis failed: {type(exc).__name__}: {exc}_"

    # Parse the verdict line for a programmatic rating.
    rating = "Hold"
    confidence = 0.5
    m = re.search(r"\*\*(Bullish|Hold|Bearish)\*\*", text)
    if m:
        rating = m.group(1)
    m = re.search(r"(\d{1,3})\s*%", text)
    if m:
        try:
            confidence = max(0.0, min(1.0, int(m.group(1)) / 100))
        except ValueError:
            pass

    verdict = TickerVerdict(
        symbol=a.symbol,
        rating=rating,
        confidence=confidence,
        catalyst_score=0,
        thesis="",
        drivers=[],
    )
    return text, verdict


def analyze(
    symbol: str,
    cached_hot: list[HotTicker] | None = None,
    cached_flows: list[FlowPost] | None = None,
) -> TickerAnalysis:
    """Run the full pipeline for one ticker."""
    sym = symbol.upper()

    with ThreadPoolExecutor(max_workers=6) as ex:
        f_quote = ex.submit(fmp.fetch_quotes, [sym])
        f_target = ex.submit(fmp.fetch_analyst_targets, [sym])
        f_profile = ex.submit(fmp.fetch_profile, sym)
        f_ratios = ex.submit(fmp.fetch_ratios_ttm, sym)
        f_news = ex.submit(polygon.fetch_news, [sym], 10)
        f_sent = ex.submit(grok.fetch_sentiment, sym)

        quote = (f_quote.result() or {}).get(sym)
        target = (f_target.result() or {}).get(sym)
        profile = f_profile.result()
        ratios = f_ratios.result()
        news = f_news.result()
        sentiment = f_sent.result()

    hot_mentions = [
        h for h in (cached_hot or []) if h.symbol.upper() == sym
    ]
    flow_posts = [
        f for f in (cached_flows or []) if f.symbol and f.symbol.upper() == sym
    ]

    upside = None
    if target and target.target_consensus and quote and quote.price:
        upside = (target.target_consensus - quote.price) / quote.price * 100

    analysis = TickerAnalysis(
        symbol=sym,
        quote=quote,
        target=target,
        profile=profile,
        ratios=ratios,
        news=news,
        sentiment=sentiment,
        flow_posts=flow_posts,
        hot_mentions=hot_mentions,
        verdict=None,
        report_markdown="",
        analyst_upside_pct=upside,
    )

    report, verdict = _claude_synthesize(analysis)
    analysis.report_markdown = report
    analysis.verdict = verdict
    return analysis
