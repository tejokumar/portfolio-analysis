"""Rebalancer: compare current vs target weights and emit trade actions.

Logic from CLAUDE.md:
- Compare Current_Weight vs Target_Weight per ticker.
- If abs(Delta) > 2%, suggest Trim or Add.
- Output exact dollar amount and shares to execute manually.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.clients.fmp import Quote
from src.clients.snaptrade import Holding
from src.config import REBALANCE_THRESHOLD, TARGET_ALLOCATION


@dataclass
class RebalanceRow:
    symbol: str
    role: str
    target_weight: float
    current_weight: float
    delta: float  # current - target
    current_value: float
    target_value: float
    trade_dollars: float  # +add, -trim
    trade_shares: float
    price: float | None
    action: str  # "Add", "Trim", "Hold", "Initiate"


def _portfolio_value(holdings: list[Holding]) -> float:
    return sum(h.market_value for h in holdings)


def build_rebalance_table(
    holdings: list[Holding],
    quotes: dict[str, Quote],
    target: dict[str, float] = TARGET_ALLOCATION,
) -> list[RebalanceRow]:
    """Build a row per target ticker. Includes tickers held but not in target as Trim-to-zero."""
    from src.config import TICKER_ROLES

    held = {h.symbol: h for h in holdings}
    total = _portfolio_value(holdings)
    if total <= 0:
        return []

    rows: list[RebalanceRow] = []
    seen: set[str] = set()

    for sym, tgt_w in target.items():
        seen.add(sym)
        h = held.get(sym)
        cur_value = h.market_value if h else 0.0
        cur_w = (cur_value / total) if total else 0.0
        delta = cur_w - tgt_w
        target_value = tgt_w * total
        trade_dollars = target_value - cur_value
        price = quotes.get(sym).price if sym in quotes else None
        trade_shares = (trade_dollars / price) if price else 0.0

        if abs(delta) <= REBALANCE_THRESHOLD:
            action = "Hold"
        elif h is None:
            action = "Initiate"
        elif delta > 0:
            action = "Trim"
        else:
            action = "Add"

        rows.append(
            RebalanceRow(
                symbol=sym,
                role=TICKER_ROLES.get(sym, ""),
                target_weight=tgt_w,
                current_weight=cur_w,
                delta=delta,
                current_value=cur_value,
                target_value=target_value,
                trade_dollars=trade_dollars,
                trade_shares=trade_shares,
                price=price,
                action=action,
            )
        )

    # Off-target holdings: anything held that isn't in the target set.
    for sym, h in held.items():
        if sym in seen:
            continue
        cur_w = h.market_value / total
        price = quotes.get(sym).price if sym in quotes else None
        rows.append(
            RebalanceRow(
                symbol=sym,
                role="(off-target)",
                target_weight=0.0,
                current_weight=cur_w,
                delta=cur_w,
                current_value=h.market_value,
                target_value=0.0,
                trade_dollars=-h.market_value,
                trade_shares=(-h.market_value / price) if price else 0.0,
                price=price,
                action="Trim" if cur_w > REBALANCE_THRESHOLD else "Hold",
            )
        )

    rows.sort(key=lambda r: abs(r.delta), reverse=True)
    return rows
