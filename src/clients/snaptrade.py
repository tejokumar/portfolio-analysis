"""SnapTrade read-only client. Pulls IRA/401k holdings.

Cadence: 1x per day (market close) — IRA holdings don't change fast.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import SETTINGS


@dataclass
class Holding:
    symbol: str
    quantity: float
    market_value: float
    average_cost: float | None = None


def is_configured() -> bool:
    s = SETTINGS
    return all(
        [
            s.snaptrade_client_id,
            s.snaptrade_consumer_key,
            s.snaptrade_user_id,
            s.snaptrade_user_secret,
        ]
    )


def fetch_holdings() -> list[Holding]:
    """Return current holdings across all linked accounts.

    Aggregates positions by symbol so the same ticker held in multiple
    accounts is rolled up before rebalancing math runs.
    """
    if not is_configured():
        return []

    from snaptrade_client import SnapTrade  # type: ignore

    s = SETTINGS
    client = SnapTrade(
        client_id=s.snaptrade_client_id,
        consumer_key=s.snaptrade_consumer_key,
    )

    accounts = client.account_information.list_user_accounts(
        user_id=s.snaptrade_user_id,
        user_secret=s.snaptrade_user_secret,
    ).body

    rolled: dict[str, Holding] = {}
    for acct in accounts:
        positions = client.account_information.get_user_account_positions(
            user_id=s.snaptrade_user_id,
            user_secret=s.snaptrade_user_secret,
            account_id=acct["id"],
        ).body

        for pos in positions:
            sym = (pos.get("symbol", {}) or {}).get("symbol", {}).get("symbol")
            if not sym:
                continue
            qty = float(pos.get("units") or 0)
            price = float(pos.get("price") or 0)
            mv = qty * price
            avg = pos.get("average_purchase_price")
            existing = rolled.get(sym)
            if existing:
                rolled[sym] = Holding(
                    symbol=sym,
                    quantity=existing.quantity + qty,
                    market_value=existing.market_value + mv,
                    average_cost=existing.average_cost,
                )
            else:
                rolled[sym] = Holding(
                    symbol=sym,
                    quantity=qty,
                    market_value=mv,
                    average_cost=float(avg) if avg is not None else None,
                )
    return list(rolled.values())
