"""AI Trend-Following Portfolio Advisor — Streamlit dashboard.

Read-only. Trades are executed manually.

Caching is keyed to **market events**, not arbitrary TTL windows:
  - Holdings: refreshed once per evening (18:00 ET).
  - FMP quotes/targets: once at 09:15 ET pre-market on weekdays.
  - Grok X sentiment: once at 09:15 ET pre-market on weekdays.
  - Polygon news: every 15 min during market hours, plus 09:15 pre-market.
  - Claude verdict synthesis: rebuilt whenever any input above changes.

The page polls every 60 seconds; when a market-clock bucket rolls over, the
relevant fetcher reruns and the UI updates automatically.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

from src import auth, clock
from src.clients import fmp, grok, llm, polygon, snaptrade
from src.clients.fmp import AnalystTarget, Quote
from src.clients.grok import FLOWGOD_HANDLE, FlowPost, HotTicker, SentimentReading
from src.clients.polygon import NewsItem
from src.clients.snaptrade import Holding
from src.config import PORTFOLIO_VALUE, TARGET_ALLOCATION, TICKERS
from src.modules.analyze import TickerAnalysis, analyze, normalize_ticker
from src.modules.briefing import Briefing, generate_briefing
from src.modules.rebalancer import build_rebalance_table
from src.modules.scorecard import Scorecard, build_scorecards

st.set_page_config(page_title="AI Portfolio Advisor", layout="wide", page_icon="📈")


# ---------- error capture ----------

_FETCH_ERRORS: dict[str, str] = {}


def _safe(label: str, fn, default):
    try:
        result = fn()
        _FETCH_ERRORS.pop(label, None)
        return result
    except Exception as exc:  # noqa: BLE001
        _FETCH_ERRORS[label] = f"{type(exc).__name__}: {exc}"
        return default


# ---------- bucket-keyed cached fetchers ----------
# Each fetcher takes a `bucket` string. When it changes, the function reruns.
# Each returns (data, fetched_at) so the UI can show how fresh the data is.

@st.cache_data(ttl=86_400 * 2, show_spinner=False, max_entries=8)
def get_holdings(bucket: str) -> tuple[list[Holding], datetime]:
    data = _safe("SnapTrade", snaptrade.fetch_holdings, [])
    return data, clock.now_et()


@st.cache_data(ttl=86_400 * 2, show_spinner=False, max_entries=8)
def get_quotes(symbols: tuple[str, ...], bucket: str) -> tuple[dict[str, Quote], datetime]:
    data = _safe("FMP quotes", lambda: fmp.fetch_quotes(list(symbols)), {})
    return data, clock.now_et()


@st.cache_data(ttl=86_400 * 2, show_spinner=False, max_entries=8)
def get_targets(symbols: tuple[str, ...], bucket: str) -> tuple[dict[str, AnalystTarget], datetime]:
    data = _safe("FMP targets", lambda: fmp.fetch_analyst_targets(list(symbols)), {})
    return data, clock.now_et()


@st.cache_data(ttl=86_400, show_spinner=False, max_entries=64)
def get_news(symbols: tuple[str, ...], bucket: str) -> tuple[list[NewsItem], datetime]:
    data = _safe("Polygon news", lambda: polygon.fetch_news(list(symbols)), [])
    return data, clock.now_et()


@st.cache_data(ttl=86_400 * 2, show_spinner=False, max_entries=8)
def get_sentiment(symbols: tuple[str, ...], bucket: str) -> tuple[dict[str, SentimentReading], datetime]:
    """Parallel Grok x_search calls — 8-way concurrency."""
    def _run():
        out: dict[str, SentimentReading] = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(grok.fetch_sentiment, s): s for s in symbols}
            for fut, sym in futures.items():
                try:
                    r = fut.result()
                    if r:
                        out[sym] = r
                except Exception:  # noqa: BLE001
                    pass
        return out
    return _safe("Grok sentiment", _run, {}), clock.now_et()


@st.cache_data(ttl=86_400, show_spinner=False, max_entries=128)
def get_hot_chatter(bucket: str) -> tuple[list[HotTicker], datetime]:
    data = _safe("Grok hot-chatter", grok.fetch_hot_chatter, [])
    return data, clock.now_et()


@st.cache_data(ttl=86_400, show_spinner=False, max_entries=256)
def get_flowgod(bucket: str) -> tuple[list[FlowPost], datetime]:
    data = _safe(f"@{FLOWGOD_HANDLE} flow", grok.fetch_flowgod_flow, [])
    return data, clock.now_et()


@st.cache_data(ttl=3_600, show_spinner=False, max_entries=200)
def get_analysis(
    symbol: str,
    nonce: int,
    _hot: list[HotTicker],
    _flows: list[FlowPost],
) -> tuple[TickerAnalysis | None, datetime]:
    """Per-ticker analysis cached for 1 hour. `nonce` is incremented per ticker
    by the manual refresh button, forcing a cache miss + refetch."""
    data = _safe(
        f"Analyze ${symbol}",
        lambda: analyze(symbol, cached_hot=_hot, cached_flows=_flows),
        None,
    )
    return data, clock.now_et()


@st.cache_data(ttl=86_400 * 2, show_spinner=False, max_entries=8)
def get_pre_briefing(bucket: str, _hot: list[HotTicker]) -> tuple[Briefing | None, datetime]:
    def _run():
        return generate_briefing("pre", list(TICKERS), _hot)
    data = _safe("Pre-market briefing", _run, None)
    return data, clock.now_et()


@st.cache_data(ttl=86_400 * 2, show_spinner=False, max_entries=8)
def get_post_briefing(bucket: str, _hot: list[HotTicker]) -> tuple[Briefing | None, datetime]:
    def _run():
        return generate_briefing("post", list(TICKERS), _hot)
    data = _safe("Post-market briefing", _run, None)
    return data, clock.now_et()


@st.cache_data(ttl=86_400, show_spinner=False, max_entries=64)
def get_scorecards(
    symbols: tuple[str, ...],
    bucket: str,
    _quotes: dict[str, Quote],
    _targets: dict[str, AnalystTarget],
    _news: list[NewsItem],
    _sentiments: dict[str, SentimentReading],
    _holdings: list[Holding],
) -> tuple[list[Scorecard], datetime]:
    data = _safe(
        "Claude synthesis",
        lambda: build_scorecards(
            list(symbols), _quotes, _targets, _news, _sentiments, _holdings,
        ),
        [],
    )
    return data, clock.now_et()


# ---------- mock fallback ----------

def mock_holdings() -> list[Holding]:
    drift = {
        "NVDA": +0.04, "MU": -0.03, "TSM": +0.02, "AVGO": -0.025,
        "QQQ": -0.01, "SMH": +0.015, "SPY": 0, "MSFT": +0.005,
        "META": -0.01, "GOOGL": 0, "VRT": +0.02, "CRWD": -0.015,
    }
    out = []
    for sym, tgt in TARGET_ALLOCATION.items():
        weight = max(0.0, tgt + drift.get(sym, 0))
        mv = PORTFOLIO_VALUE * weight
        out.append(Holding(symbol=sym, quantity=mv / 100.0, market_value=mv))
    return out


def mock_quotes() -> dict[str, Quote]:
    base = {
        "SPY": (520, 0.3), "QQQ": (445, 0.5), "SMH": (235, 1.1),
        "NVDA": (118, 2.4), "TSM": (172, 0.9), "MU": (108, -1.8),
        "AVGO": (185, 1.5), "MSFT": (430, 0.2), "META": (510, -0.4),
        "GOOGL": (175, 0.6), "VRT": (95, 1.9), "CRWD": (320, -2.1),
    }
    return {s: Quote(symbol=s, price=p, change_pct=c) for s, (p, c) in base.items()}


# ---------- UI ----------

RATING_EMOJI = {"Bullish": "🟢", "Hold": "🟡", "Bearish": "🔴", "—": "⚪"}
ACTION_EMOJI = {
    "Add": "➕", "Trim": "➖", "Initiate": "🆕", "Hold": "✓",
    "Exit": "🚪", "Watch": "👀",
}


def status_banner() -> None:
    """One compact line — works on phone and desktop."""
    flags = [
        ("SnapTrade", snaptrade.is_configured()),
        ("FMP", fmp.is_configured()),
        ("Polygon", polygon.is_configured()),
        ("Grok", grok.is_configured()),
        ("Claude", llm.is_configured()),
    ]
    parts = [f"{'✓' if ok else '○'} {name}" for name, ok in flags]
    st.caption(" · ".join(parts))


def render_freshness_caption(label: str, fetched: datetime, nxt: datetime) -> None:
    now = clock.now_et()
    st.caption(
        f"⏱ {label} · fetched {clock.fmt_dt(fetched, with_date=False)} "
        f"({clock.fmt_age(fetched, now)}) · next {clock.fmt_dt(nxt, with_date=False)} "
        f"({clock.fmt_until(nxt, now)})"
    )


def _esc(s: str) -> str:
    """Escape $ in markdown so it isn't interpreted as math mode."""
    return s.replace("$", r"\$")


SIDE_EMOJI = {"Bullish": "🟢", "Bearish": "🔴", "Mixed": "🟡"}


def render_chatter(
    hot: list[HotTicker], fetched: datetime, nxt: datetime,
    portfolio_set: set[str],
) -> None:
    st.subheader("X Hot Chatter")
    render_freshness_caption("Chatter", fetched, nxt)
    if not clock.in_active_window():
        st.caption(
            "💤 Polling paused outside market hours. Showing last in-window data."
        )
    st.caption(
        "Grok scans X for the most-discussed US tickers in the last 12-24h, "
        "filtered to verified financial accounts. Refreshes every 15 min "
        "during the 06:15–14:00 PT active window (weekdays)."
    )
    if not hot:
        st.info("No chatter returned. (First refresh may be pending.)")
        return

    bullish = [h for h in hot if h.side == "Bullish"]
    bearish = [h for h in hot if h.side == "Bearish"]
    mixed = [h for h in hot if h.side == "Mixed"]
    st.markdown(
        f"🟢 **{len(bullish)}** Bullish · 🔴 **{len(bearish)}** Bearish · "
        f"🟡 **{len(mixed)}** Mixed"
    )

    def _block(label: str, items: list[HotTicker]) -> None:
        if not items:
            return
        st.markdown(f"### {label}")
        for h in items:
            in_port = " ⭐" if h.symbol.upper() in portfolio_set else ""
            emoji = SIDE_EMOJI.get(h.side, "·")
            st.markdown(f"{emoji} **${h.symbol}**{in_port} — {h.thesis}")

    _block("Bullish chatter", bullish)
    _block("Bearish chatter", bearish)
    _block("Mixed chatter", mixed)


def render_analyze(
    hot: list[HotTicker],
    flows: list[FlowPost],
) -> None:
    st.subheader("🔍 Analyze a Ticker")
    st.caption(
        "Enter any US ticker. The bot pulls FMP quote / target / profile / "
        "ratios, Polygon news, Grok X-sentiment, and any FlowGod or X-chatter "
        "mentions today, then has Claude synthesize a Bullish / Hold / Bearish "
        "thesis with bull/bear cases, valuation read, and bottom-line action. "
        "Each ticker is cached for 1 hour."
    )

    # Visual uppercase + iOS-keyboard autocapitalize for the analyze input.
    # Scoped to inputs that carry our placeholder so the password field is untouched.
    components.html(
        """
        <script>
        const root = window.parent.document;
        const apply = () => {
            root.querySelectorAll('input[placeholder*="NVDA"]').forEach(el => {
                el.setAttribute('autocapitalize', 'characters');
                el.setAttribute('autocorrect', 'off');
                el.setAttribute('spellcheck', 'false');
                el.style.textTransform = 'uppercase';
            });
        };
        apply();
        // Streamlit may re-render — observe for new inputs.
        new MutationObserver(apply).observe(root.body, {childList: true, subtree: true});
        </script>
        """,
        height=0,
    )

    with st.form("analyze_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            raw = st.text_input(
                "Ticker", placeholder="e.g. NVDA, TSLA, PLTR",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Analyze", type="primary", use_container_width=True)

    # Determine which ticker to render. Either just submitted, or sticky from
    # the last submit in this session.
    if submitted:
        sym = normalize_ticker(raw)
        if not sym:
            st.error("Enter a valid US ticker (e.g. NVDA, BRK.B).")
            return
        st.session_state["analyze_current"] = sym
    else:
        sym = st.session_state.get("analyze_current")
        if not sym:
            return

    nonces = st.session_state.setdefault("analyze_nonces", {})
    nonce = nonces.get(sym, 0)

    with st.status(f"📡 Analyzing {sym}…", expanded=True) as s:
        s.write("Fetching FMP, Polygon, and Grok in parallel…")
        s.write("Asking Claude to synthesize…")
        analysis, fetched_at = get_analysis(sym, nonce, hot, flows)
        s.update(label=f"✅ Analysis ready for {sym}", state="complete", expanded=False)

    # Manual refresh — bumps nonce so the next call is a cache miss.
    refresh_col = st.columns([1, 4])[0]
    if refresh_col.button("🔄 Refresh", key=f"refresh_{sym}", use_container_width=True):
        nonces[sym] = nonce + 1
        st.rerun()

    if analysis is None:
        st.error(f"Analysis failed for {sym}. Check the error panel below.")
        return

    # Compact header — single markdown line works on mobile and desktop.
    price_str = (
        f"{_esc(f'${analysis.quote.price:,.2f}')}"
        if analysis.quote else "—"
    )
    delta_str = (
        f"{analysis.quote.change_pct:+.2f}%"
        if analysis.quote and analysis.quote.change_pct is not None else "—"
    )
    upside_str = (
        f"{analysis.analyst_upside_pct:+.1f}%"
        if analysis.analyst_upside_pct is not None else "—"
    )
    sent_str = (
        f"{analysis.sentiment.score:+.2f}"
        if analysis.sentiment else "—"
    )
    consensus_note = (
        f" (consensus {_esc(f'${analysis.target.target_consensus}')})"
        if analysis.target and analysis.target.target_consensus else ""
    )

    st.markdown(
        f"**{analysis.symbol}** · Price {price_str} **{delta_str}** · "
        f"Upside {upside_str}{consensus_note} · "
        f"X-Sent {sent_str} · FlowGod hits **{len(analysis.flow_posts)}**"
    )
    st.caption(
        f"Fetched {clock.fmt_dt(fetched_at)} · "
        f"{len(analysis.news)} news · {len(analysis.hot_mentions)} chatter mentions"
    )

    st.markdown(analysis.report_markdown)

    # Raw data drilldown for the curious.
    with st.expander("📂 Raw data used in this analysis"):
        if analysis.profile:
            st.markdown(
                f"**Company:** {analysis.profile.name} · "
                f"{analysis.profile.sector} / {analysis.profile.industry}  \n"
                f"Market cap: {_esc(_fmt_mcap(analysis.profile.market_cap))} · "
                f"Beta: {analysis.profile.beta or '—'}"
            )
            if analysis.profile.description:
                st.caption(analysis.profile.description[:500] + "…")
        if analysis.news:
            st.markdown("**Recent headlines:**")
            for n in analysis.news[:8]:
                st.markdown(
                    f"- [{n.title}]({n.url})  \n"
                    f"  *{n.publisher} · {clock.fmt_dt(n.published_at)}*"
                )
        if analysis.sentiment:
            st.markdown(f"**Grok X-sentiment note:** {analysis.sentiment.summary}")
        if analysis.flow_posts:
            st.markdown("**FlowGod mentions today:**")
            for f in analysis.flow_posts:
                st.markdown(f"- [{f.conviction}/10] {f.summary}")


def _fmt_mcap(x) -> str:
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


def render_flowgod(
    flows: list[FlowPost], fetched: datetime, nxt: datetime,
    portfolio_set: set[str],
) -> None:
    st.subheader(f"🌊 @{FLOWGOD_HANDLE} Flow")
    render_freshness_caption("FlowGod", fetched, nxt)
    if not clock.in_active_window():
        st.caption(
            "💤 Polling paused outside market hours. Showing last in-window data."
        )
    st.caption(
        f"Today's substantive posts from @{FLOWGOD_HANDLE} on X. "
        "Grok inspects attached images (charts / OI tables) and extracts the "
        "ticker, direction, and dollar/share details. Pure-reaction posts are "
        "filtered out. Refreshes every 10 min during the 06:15–14:00 PT active "
        "window (weekdays)."
    )
    if not flows:
        st.info(
            f"No substantive flow posts captured from @{FLOWGOD_HANDLE} today. "
            "(Pure reactions, replies, and posts without a clear ticker are filtered.)"
        )
        return

    n_high = sum(1 for f in flows if f.conviction >= 7)
    st.markdown(
        f"📝 **{len(flows)}** flow posts · "
        f"🔥 **{n_high}** high-conviction (≥7) · "
        f"sorted by conviction"
    )

    for f in flows:
        in_port = " ⭐" if f.symbol and f.symbol in portfolio_set else ""
        img_badge = " 🖼" if f.has_image else ""
        side_badge = f" · {f.side}" if f.side else ""
        conv_emoji = "🔥" if f.conviction >= 7 else "·"
        sym_display = f"${f.symbol}" if f.symbol else "—"

        # Single-line summary header.
        st.markdown(
            f"{conv_emoji} **[{f.conviction}/10]** **{sym_display}**"
            f"{in_port}{side_badge}{img_badge} — {f.summary}"
        )
        if f.image_describes:
            st.caption(f"🖼 Image: {f.image_describes}")
        meta_bits = []
        if f.posted_at:
            meta_bits.append(f"posted {f.posted_at}")
        if f.url:
            meta_bits.append(f"[view on X]({f.url})")
        if meta_bits:
            st.caption(" · ".join(meta_bits))


def render_briefing(
    label: str, briefing: Briefing | None, fetched: datetime, nxt: datetime,
) -> None:
    st.subheader(label)
    render_freshness_caption(label, fetched, nxt)
    if briefing is None:
        st.info("Briefing pending — first refresh hasn't fired yet.")
        return
    st.caption(
        f"Sources: {briefing.portfolio_news_count} portfolio news · "
        f"{briefing.broad_news_count} broad news · "
        f"{briefing.econ_count} econ events · "
        f"{briefing.earnings_count} earnings · "
        f"{briefing.hot_count} hot tickers"
    )
    st.markdown(briefing.markdown)


def _fmt_action_size(rec) -> str:
    """E.g., '$5,200 (+24 sh)' or '—' when no trade."""
    if rec.action in ("Hold", "Watch"):
        return ""
    if rec.dollars == 0:
        return ""
    sign = "+" if rec.dollars > 0 else "-"
    dollars = abs(rec.dollars)
    parts = [f"{sign}{_esc(f'${dollars:,.0f}')}"]
    if rec.shares:
        parts.append(f"({rec.shares:+.1f} sh)")
    return " ".join(parts)


def render_scorecard_mobile(cards: list[Scorecard], fetched: datetime, nxt: datetime) -> None:
    """Mobile-first verdict view: action recommendation + verdict per ticker."""
    st.subheader("Verdict Scorecard")
    render_freshness_caption("Verdicts", fetched, nxt)

    if not cards:
        st.info("Scorecard unavailable.")
        return

    # Action counts (more useful than verdict-only counts since these are
    # what the user actually does).
    action_counts: dict[str, int] = {}
    for c in cards:
        action_counts[c.recommendation.action] = action_counts.get(c.recommendation.action, 0) + 1

    summary_bits = []
    for act in ("Initiate", "Add", "Trim", "Exit", "Hold", "Watch"):
        n = action_counts.get(act, 0)
        if n:
            summary_bits.append(f"{ACTION_EMOJI.get(act, '·')} **{n}** {act}")
    st.markdown(" · ".join(summary_bits))

    high_conf_bear = [
        c for c in cards
        if c.rating == "Bearish" and (c.confidence or 0) >= 0.7
    ]
    if high_conf_bear:
        st.warning(
            "High-confidence Bearish: "
            + ", ".join(c.symbol for c in high_conf_bear)
        )
    st.caption(
        "Action combines Claude's Bullish/Hold/Bearish verdict with your "
        "current portfolio weight vs target. Sorted by urgency."
    )

    for c in cards:
        v_emoji = RATING_EMOJI.get(c.rating, "⚪")
        a_emoji = ACTION_EMOJI.get(c.recommendation.action, "·")
        conf_str = f" {c.confidence:.0%}" if c.confidence is not None else ""
        action_size = _fmt_action_size(c.recommendation)
        action_label = (
            f"{a_emoji} {c.recommendation.action} {action_size}".strip()
            if action_size else f"{a_emoji} {c.recommendation.action}"
        )
        title = (
            f"{v_emoji} {c.symbol} → {action_label} · "
            f"{c.rating}{conf_str} · {c.role}"
        )
        with st.expander(title, expanded=False):
            # Recommendation block first — what to do.
            st.markdown(
                f"**Action: {a_emoji} {c.recommendation.action}"
                + (f" {action_size}" if action_size else "")
                + "**  \n"
                + c.recommendation.rationale
            )
            st.caption(
                f"Current weight {c.current_weight:.1%} · "
                f"Target {c.target_weight:.1%} · "
                f"Held value {_esc(f'${c.current_value:,.0f}')}"
            )

            # Inline metric line.
            price_str = f"${c.price:,.2f}" if c.price is not None else "—"
            up_str = (
                f"{c.analyst_upside_pct:+.1f}%"
                if c.analyst_upside_pct is not None else "—"
            )
            delta_str = f"{c.change_pct:+.2f}%" if c.change_pct is not None else "—"
            cat_str = f"{c.catalyst_score:+d}" if c.catalyst_score is not None else "—"
            sent_str = (
                f"{c.sentiment_score:+.2f}"
                if c.sentiment_score is not None else "—"
            )
            st.markdown(
                f"**Price** {_esc(price_str)} · **Δ** {delta_str} · "
                f"**Upside** {up_str} · **Catalyst** {cat_str} · "
                f"**X-Sent** {sent_str}"
            )

            if c.thesis:
                st.markdown(f"**Thesis.** {c.thesis}")
            if c.verdict and c.verdict.drivers:
                st.markdown("**Drivers**")
                for d in c.verdict.drivers:
                    st.markdown(f"- {d}")
            if c.sentiment_summary:
                st.caption(f"X-sentiment note: {c.sentiment_summary}")

            if c.news:
                st.markdown("**Recent news**")
                for n in c.news[:5]:
                    st.markdown(
                        f"- [{n.title}]({n.url})  \n"
                        f"  *{n.publisher} · {clock.fmt_dt(n.published_at)}*"
                    )
            else:
                st.caption("No recent ticker-tagged news.")


def _render_rebalance_row(r) -> None:
    """Single-line summary per ticker — weights and trade in one place."""
    emoji = ACTION_EMOJI.get(r.action, "·")
    weights = (
        f"{r.current_weight:.1%} → {r.target_weight:.1%} "
        f"(Δ {r.delta:+.1%})"
    )
    if abs(r.trade_dollars) < 1:
        trade = ""
    elif r.price:
        trade = (
            f" · **{r.action}** {_esc(f'${abs(r.trade_dollars):,.0f}')} "
            f"({r.trade_shares:+.1f} sh)"
        )
    else:
        trade = f" · **{r.action}** {_esc(f'${abs(r.trade_dollars):,.0f}')}"
    st.markdown(f"{emoji} **{r.symbol}** · {r.role} · {weights}{trade}")


def render_allocation(
    holdings: list[Holding],
    quotes: dict[str, Quote],
    holdings_at: datetime, holdings_next: datetime,
    quotes_at: datetime, quotes_next: datetime,
) -> None:
    st.subheader("Allocation & Rebalance")
    render_freshness_caption("Holdings", holdings_at, holdings_next)
    render_freshness_caption("Quotes", quotes_at, quotes_next)

    rows = build_rebalance_table(holdings, quotes)
    if not rows:
        st.info("No holdings to display.")
        return

    total_value = sum(h.market_value for h in holdings)
    off_target_value = sum(r.current_value for r in rows if r.role == "(off-target)")
    actionable = sum(1 for r in rows if r.action in ("Add", "Trim", "Initiate"))
    max_drift = max((abs(r.delta) for r in rows), default=0.0)

    # Compact summary line — works on phone width.
    st.markdown(
        f"**Portfolio** {_esc(f'${total_value:,.0f}')} · "
        f"**{len(holdings)}** positions · "
        f"**Max drift** {max_drift:.1%} · "
        f"**Actionable** {actionable} · "
        f"**Off-target** {_esc(f'${off_target_value:,.0f}')}"
    )
    st.caption(
        f"Target weights applied to your live portfolio total of "
        f"{_esc(f'${total_value:,.0f}')}. "
        "Trade dollars + shares are what to Add/Trim manually."
    )

    st.markdown(f"**All positions ({len(rows)})** — sorted by drift")
    for r in rows:
        _render_rebalance_row(r)


def _tab_refresh_button(tab_id: str) -> None:
    """Small 🔄 button in a tab's top-right. Bumps the per-tab nonce so the next
    cached-fetcher call is a cache miss."""
    nonces = st.session_state.setdefault("tab_nonces", {})
    cols = st.columns([10, 1])
    with cols[1]:
        if st.button("🔄", key=f"refresh_{tab_id}", help="Refresh this tab"):
            nonces[tab_id] = nonces.get(tab_id, 0) + 1
            # Bumping shared inputs too where relevant — handled by caller for
            # dependencies like chatter (used by briefings).
            st.rerun()


def _tab_nonce(tab_id: str) -> int:
    return st.session_state.get("tab_nonces", {}).get(tab_id, 0)


def _swr_render(tab_id: str, fresh_data, render_fn) -> None:
    """Stale-while-revalidate render pattern.

    fresh_data: the newly-fetched data (may be None or empty if fetch failed)
    render_fn(data) → renders the panel for that data

    Pulls the last successfully-rendered data out of session_state. If the
    fresh fetch returned something, uses that and saves it. Otherwise falls
    back to the stored copy so the panel never blanks out on a failed refresh.
    """
    key = f"swr_{tab_id}"
    if fresh_data:
        st.session_state[key] = fresh_data
        render_fn(fresh_data)
        return
    stored = st.session_state.get(key)
    if stored:
        st.warning("Refresh returned no data — showing the last successful fetch.")
        render_fn(stored)
    else:
        st.info("No data yet. Tap 🔄 to retry.")


def _nonced(bucket: str, tab_id: str) -> str:
    """Combine a bucket key with the tab's refresh nonce so manual refresh
    bypasses cache without clearing other tabs."""
    n = _tab_nonce(tab_id)
    return f"{bucket}|n{n}" if n else bucket


def _render_tab_dashboard() -> None:
    _tab_refresh_button("dashboard")
    with st.spinner("Loading dashboard…"):
        try:
            render_dashboard()
        except Exception as exc:  # noqa: BLE001
            stored = st.session_state.get("swr_dashboard_ok")
            if stored:
                st.warning(f"Refresh failed: {exc}. Showing last successful view.")
            else:
                st.error(f"Failed to load dashboard: {exc}")
    st.session_state["swr_dashboard_ok"] = True


def _render_tab_briefing(
    mode: str,        # "pre" | "post"
    base_bucket: str,
    chatter_b: str,
    nxt: datetime,
    label: str,
) -> None:
    _tab_refresh_button(mode)
    nonced_brief = _nonced(base_bucket, mode)
    nonced_chat = _nonced(chatter_b, "chatter_shared")
    with st.spinner(f"Loading {label.lower()}…"):
        hot, _ = get_hot_chatter(nonced_chat)
        if mode == "pre":
            briefing, at = get_pre_briefing(nonced_brief, hot)
        else:
            briefing, at = get_post_briefing(nonced_brief, hot)

    def _render(payload):
        b, t = payload
        render_briefing(label, b, t, nxt)

    _swr_render(mode, (briefing, at) if briefing else None, _render)


def _render_tab_chatter(chatter_b: str, chatter_next: datetime) -> None:
    _tab_refresh_button("chatter")
    nonced = _nonced(chatter_b, "chatter")
    with st.spinner("Loading X chatter…"):
        hot, hot_at = get_hot_chatter(nonced)

    def _render(payload):
        h, h_at = payload
        render_chatter(h, h_at, chatter_next, set(TICKERS))

    _swr_render("chatter", (hot, hot_at) if hot else None, _render)


def _render_tab_flowgod(flowgod_b: str, flowgod_next: datetime) -> None:
    _tab_refresh_button("flowgod")
    nonced = _nonced(flowgod_b, "flowgod")
    with st.spinner("Loading FlowGod posts…"):
        flows, flows_at = get_flowgod(nonced)

    def _render(payload):
        f, f_at = payload
        render_flowgod(f, f_at, flowgod_next, set(TICKERS))

    # Render even when empty — empty is a valid state (e.g., Sunday, no posts).
    if flows or "swr_flowgod" not in st.session_state:
        st.session_state["swr_flowgod"] = (flows, flows_at)
    stored = st.session_state.get("swr_flowgod")
    if stored:
        _render(stored)
    else:
        st.info("No FlowGod posts yet. Tap 🔄 to retry.")


def _render_tab_analyze(chatter_b: str, flowgod_b: str) -> None:
    """Analyze tab has its own internal refresh (per-ticker nonce). No tab-level
    refresh needed since each analysis is keyed by ticker."""
    # Pull shared cached data (cache hits if other tabs have populated).
    hot, _ = get_hot_chatter(chatter_b)
    flows, _ = get_flowgod(flowgod_b)
    render_analyze(hot, flows)


def render_dashboard() -> None:
    """Allocation + verdict scorecard. All fetchers below hit warm caches."""
    holdings_b = clock.holdings_bucket()
    daily_b = clock.daily_premarket_bucket()
    news_b = clock.news_bucket()
    verdict_b = f"{daily_b}|{news_b}"

    holdings_next = clock.next_holdings_refresh()
    daily_next = clock.next_premarket_refresh()
    news_next = clock.next_news_refresh()
    verdict_next = min(daily_next, news_next)

    holdings, holdings_at = get_holdings(holdings_b)
    if not holdings:
        holdings, holdings_at = mock_holdings(), clock.now_et()

    quotes, quotes_at = get_quotes(tuple(TICKERS), daily_b)
    if not quotes:
        quotes, quotes_at = mock_quotes(), clock.now_et()

    render_allocation(
        holdings, quotes,
        holdings_at, holdings_next,
        quotes_at, daily_next,
    )
    st.divider()

    if llm.is_configured():
        targets, _ = get_targets(tuple(TICKERS), daily_b)
        news, _ = get_news(tuple(TICKERS), news_b)
        sentiments, _ = get_sentiment(tuple(TICKERS), daily_b)
        cards, cards_at = get_scorecards(
            tuple(TICKERS), verdict_b, quotes, targets, news, sentiments, holdings,
        )
        if cards:
            render_scorecard_mobile(cards, cards_at, verdict_next)
    else:
        st.warning("ANTHROPIC_API_KEY not set — verdicts disabled.")


def main() -> None:
    auth.require_password()

    # Re-run every 60s so bucket rollovers cause automatic refetches.
    st_autorefresh(interval=60_000, key="market_clock_autorefresh")

    st.markdown("### 📈 AI Portfolio Advisor")
    status_banner()
    st.caption("Read-only. Trades executed manually.")

    # ---- Sidebar ----
    with st.sidebar:
        role_badge = "👤 Owner" if auth.is_owner() else "👋 Guest"
        st.markdown(f"**Signed in:** {role_badge}")
        st.divider()
        if auth.is_owner():
            st.header("Refresh")
            if st.button("🔄 Refresh all on demand", use_container_width=True, type="primary"):
                st.cache_data.clear()
                st.rerun()
            st.caption("Forces a full refetch of every source.")
            st.divider()
        st.markdown(f"**Now:** {clock.fmt_dt(clock.now_et())}")
        st.markdown("**Schedule (PT):**")
        schedule_items = [
            "- Pre-market brief: 06:15 weekdays",
            "- Post-market brief: 14:00 weekdays",
            "- X Hot Chatter: every 15 min (06:15–14:00 weekdays)",
            f"- @{FLOWGOD_HANDLE} flow: every 10 min (06:15–14:00 weekdays)",
        ]
        if auth.is_owner():
            schedule_items[:0] = [
                "- Holdings: 15:00 daily",
                "- FMP / Grok: 06:15 weekdays",
                "- News: 06:15 + every 15 min in-session",
            ]
        st.markdown("\n".join(schedule_items))

    # Bucket keys (instant — just strings). Scheduled refreshes still drive
    # cache invalidation; only the UI blocking is removed.
    chatter_b = clock.chatter_bucket()
    chatter_next = clock.next_chatter_refresh()
    flowgod_b = clock.flowgod_bucket()
    flowgod_next = clock.next_flowgod_refresh()
    pre_b = clock.pre_briefing_bucket()
    pre_next = clock.next_pre_briefing()
    post_b = clock.post_briefing_bucket()
    post_next = clock.next_post_briefing()

    tab_labels = ["🌅 Pre-Market", "🌇 Post-Market", "🔥 X Chatter",
                  "🌊 FlowGod", "🔍 Analyze"]
    if auth.is_owner():
        tab_labels = ["📊 Dashboard"] + tab_labels
        tabs = st.tabs(tab_labels)
        with tabs[0]:
            _render_tab_dashboard()
        tab_pre, tab_post, tab_chat, tab_flow, tab_analyze = tabs[1:]
    else:
        tabs = st.tabs(tab_labels)
        tab_pre, tab_post, tab_chat, tab_flow, tab_analyze = tabs

    with tab_pre:
        _render_tab_briefing("pre", pre_b, chatter_b, pre_next, "Pre-Market Brief")

    with tab_post:
        _render_tab_briefing("post", post_b, chatter_b, post_next, "Post-Market Brief")

    with tab_chat:
        _render_tab_chatter(chatter_b, chatter_next)

    with tab_flow:
        _render_tab_flowgod(flowgod_b, flowgod_next)

    with tab_analyze:
        _render_tab_analyze(chatter_b, flowgod_b)

    # ---- Errors panel ----
    if _FETCH_ERRORS:
        with st.expander(f"⚠️ {len(_FETCH_ERRORS)} data source(s) failed — click to inspect"):
            for label, err in _FETCH_ERRORS.items():
                st.code(f"{label}: {err}", language="text")


if __name__ == "__main__":
    main()
