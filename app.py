"""AI Trend-Following Portfolio Advisor — Streamlit dashboard.

Read-only. No trades are executed. Surfaces:
  - Allocation drift vs target (Rebalancer)
  - Catalyst-scored news (Polygon + Claude)
  - X sentiment divergence (Grok)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.clients import fmp, grok, llm, polygon, snaptrade
from src.clients.fmp import Quote
from src.clients.snaptrade import Holding
from src.config import PORTFOLIO_VALUE, TARGET_ALLOCATION, TICKER_ROLES, TICKERS
from src.modules.catalyst import score_news
from src.modules.rebalancer import build_rebalance_table
from src.modules.sentiment import detect_divergence

st.set_page_config(page_title="AI Portfolio Advisor", layout="wide", page_icon="📈")


# ---------- cached data fetchers ----------
# Cadences from CLAUDE.md, expressed as st.cache_data TTLs in seconds.

@st.cache_data(ttl=86_400, show_spinner="Syncing brokerage holdings…")
def get_holdings() -> list[Holding]:
    return snaptrade.fetch_holdings()


@st.cache_data(ttl=7_200, show_spinner="Fetching quotes…")
def get_quotes(symbols: tuple[str, ...]) -> dict[str, Quote]:
    return fmp.fetch_quotes(list(symbols))


@st.cache_data(ttl=7_200, show_spinner="Fetching analyst targets…")
def get_targets(symbols: tuple[str, ...]) -> dict:
    return fmp.fetch_analyst_targets(list(symbols))


@st.cache_data(ttl=1_800, show_spinner="Pulling latest news…")
def get_news(symbols: tuple[str, ...]) -> list:
    return polygon.fetch_news(list(symbols))


@st.cache_data(ttl=1_800, show_spinner="Scoring catalysts…")
def get_scored_news(symbols: tuple[str, ...]) -> list:
    items = polygon.fetch_news(list(symbols))
    return score_news(items, set(symbols))


@st.cache_data(ttl=43_200, show_spinner="Reading X sentiment…")
def get_sentiment(symbols: tuple[str, ...]) -> dict:
    out = {}
    for s in symbols:
        r = grok.fetch_sentiment(s)
        if r:
            out[s] = r
    return out


# ---------- mock fallback so the dashboard works without keys ----------

def mock_holdings() -> list[Holding]:
    """Generate plausible holdings if SnapTrade isn't configured.

    Slightly off target so the rebalancer has something to surface.
    """
    drift = {
        "NVDA": +0.04, "MU": -0.03, "TSM": +0.02, "AVGO": -0.025,
        "QQQ": -0.01, "SMH": +0.015, "SPY": 0, "MSFT": +0.005,
        "META": -0.01, "GOOGL": 0, "VRT": +0.02, "CRWD": -0.015,
    }
    holdings = []
    for sym, tgt in TARGET_ALLOCATION.items():
        weight = max(0.0, tgt + drift.get(sym, 0))
        mv = PORTFOLIO_VALUE * weight
        holdings.append(Holding(symbol=sym, quantity=mv / 100.0, market_value=mv))
    return holdings


def mock_quotes() -> dict[str, Quote]:
    base = {
        "SPY": (520, 0.3), "QQQ": (445, 0.5), "SMH": (235, 1.1),
        "NVDA": (118, 2.4), "TSM": (172, 0.9), "MU": (108, -1.8),
        "AVGO": (185, 1.5), "MSFT": (430, 0.2), "META": (510, -0.4),
        "GOOGL": (175, 0.6), "VRT": (95, 1.9), "CRWD": (320, -2.1),
    }
    return {
        s: Quote(symbol=s, price=p, change_pct=c, market_cap=None)
        for s, (p, c) in base.items()
    }


# ---------- UI ----------

def status_banner() -> None:
    cols = st.columns(5)
    flags = [
        ("SnapTrade", snaptrade.is_configured()),
        ("FMP", fmp.is_configured()),
        ("Polygon", polygon.is_configured()),
        ("Grok", grok.is_configured()),
        ("Claude", llm.is_configured()),
    ]
    for col, (name, ok) in zip(cols, flags):
        col.metric(name, "✓ live" if ok else "○ mock")


def render_allocation(holdings: list[Holding], quotes: dict[str, Quote]) -> None:
    st.subheader("Allocation & Rebalance")
    rows = build_rebalance_table(holdings, quotes)
    if not rows:
        st.info("No holdings to display.")
        return

    df = pd.DataFrame([r.__dict__ for r in rows])
    total_value = sum(h.market_value for h in holdings)

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Value", f"${total_value:,.0f}")
    c2.metric("Target Value", f"${PORTFOLIO_VALUE:,.0f}")
    actionable = sum(1 for r in rows if r.action in ("Add", "Trim", "Initiate"))
    c3.metric("Actionable", actionable, help="Tickers with drift > 2%")

    display = df[
        [
            "symbol", "role", "target_weight", "current_weight", "delta",
            "current_value", "target_value", "trade_dollars", "trade_shares", "action",
        ]
    ].copy()
    display["target_weight"] = display["target_weight"].map(lambda x: f"{x:.1%}")
    display["current_weight"] = display["current_weight"].map(lambda x: f"{x:.1%}")
    display["delta"] = display["delta"].map(lambda x: f"{x:+.1%}")
    display["current_value"] = display["current_value"].map(lambda x: f"${x:,.0f}")
    display["target_value"] = display["target_value"].map(lambda x: f"${x:,.0f}")
    display["trade_dollars"] = display["trade_dollars"].map(lambda x: f"${x:+,.0f}")
    display["trade_shares"] = display["trade_shares"].map(lambda x: f"{x:+.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_catalysts(symbols: list[str]) -> None:
    st.subheader("Catalyst Feed")
    if not polygon.is_configured():
        st.info("Add `POLYGON_API_KEY` to enable real-time news.")
        return

    if llm.is_configured():
        scored = get_scored_news(tuple(symbols))
        if not scored:
            st.write("No tagged news in the last cycle.")
            return
        for s in scored[:15]:
            v = s.verdict
            badge = "🔵 Structural" if v and v.classification.lower().startswith("struct") else "⚪ Noise"
            score_str = f"{v.score:+d}" if v else "?"
            tickers = ", ".join(t for t in s.item.tickers if t in symbols) or "—"
            with st.expander(f"{badge} [{score_str}] {tickers} — {s.item.title}"):
                st.caption(f"{s.item.publisher} · {s.item.published_at:%Y-%m-%d %H:%M}")
                if v:
                    st.write(v.rationale)
                st.markdown(f"[Read source]({s.item.url})")
    else:
        items = get_news(tuple(symbols))
        st.caption("Add `ANTHROPIC_API_KEY` to enable catalyst scoring.")
        for n in items[:15]:
            tickers = ", ".join(t for t in n.tickers if t in symbols) or "—"
            with st.expander(f"{tickers} — {n.title}"):
                st.caption(f"{n.publisher} · {n.published_at:%Y-%m-%d %H:%M}")
                st.markdown(f"[Read source]({n.url})")


def render_sentiment(symbols: list[str], quotes: dict[str, Quote]) -> None:
    st.subheader("X Sentiment Divergence")
    if not grok.is_configured():
        st.info("Add `GROK_API_KEY` to enable X sentiment.")
        return
    readings = get_sentiment(tuple(symbols))
    signals = detect_divergence(quotes, readings)
    if not signals:
        st.write("No sentiment signals available.")
        return

    for sig in signals:
        emoji = {"Buy the Dip": "🟢", "Topping": "🔴", "Aligned": "⚪"}[sig.flag]
        cols = st.columns([1, 1, 1, 4])
        cols[0].markdown(f"**{sig.symbol}**")
        cols[1].markdown(f"{sig.price_change_pct:+.2f}%")
        cols[2].markdown(f"sentiment {sig.sentiment_score:+.2f}")
        cols[3].markdown(f"{emoji} **{sig.flag}** — {sig.sentiment_summary}")


def render_targets(symbols: list[str]) -> None:
    if not fmp.is_configured():
        return
    st.subheader("Analyst Price Targets")
    targets = get_targets(tuple(symbols))
    quotes = get_quotes(tuple(symbols))
    if not targets:
        st.caption("No analyst data returned.")
        return
    rows = []
    for sym, t in targets.items():
        q = quotes.get(sym)
        upside = (
            ((t.target_consensus - q.price) / q.price * 100)
            if (q and t.target_consensus and q.price)
            else None
        )
        rows.append(
            {
                "symbol": sym,
                "role": TICKER_ROLES.get(sym, ""),
                "price": q.price if q else None,
                "consensus": t.target_consensus,
                "high": t.target_high,
                "low": t.target_low,
                "upside_%": round(upside, 1) if upside is not None else None,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.title("📈 AI Portfolio Advisor")
    st.caption("Read-only. Trades are executed manually. Target: $500k AI/semi tilt.")
    status_banner()

    with st.sidebar:
        st.header("Refresh")
        if st.button("Clear cache & reload", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(
            "Cadences:\n"
            "• Holdings: 24h\n"
            "• Quotes/Targets: 2h\n"
            "• News: 30min\n"
            "• Sentiment: 12h"
        )

    holdings = get_holdings() or mock_holdings()
    quotes = get_quotes(tuple(TICKERS)) or mock_quotes()

    render_allocation(holdings, quotes)
    st.divider()
    render_catalysts(TICKERS)
    st.divider()
    render_sentiment(TICKERS, quotes)
    st.divider()
    render_targets(TICKERS)


if __name__ == "__main__":
    main()
