"""Static configuration: target allocation and env-backed settings."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

PORTFOLIO_VALUE = 500_000.0

# Target weights from CLAUDE.md. Must sum to 1.0.
TARGET_ALLOCATION: dict[str, float] = {
    "SPY": 0.05,
    "QQQ": 0.20,
    "SMH": 0.10,
    "NVDA": 0.12,
    "TSM": 0.10,
    "MU": 0.10,
    "AVGO": 0.08,
    "MSFT": 0.05,
    "META": 0.05,
    "GOOGL": 0.05,
    "VRT": 0.05,
    "CRWD": 0.05,
}

TICKER_ROLES: dict[str, str] = {
    "SPY": "Anchor",
    "QQQ": "Growth Core",
    "SMH": "Sector Momentum",
    "NVDA": "AI King",
    "TSM": "Foundry Monopoly",
    "MU": "Memory Squeeze Play",
    "AVGO": "ASIC/Networking",
    "MSFT": "Hyperscaler",
    "META": "Platforms",
    "GOOGL": "Search/Cloud",
    "VRT": "Data Center Infrastructure",
    "CRWD": "Cybersecurity Tax",
}

# Rebalance threshold — drift larger than this triggers a Trim/Add suggestion.
REBALANCE_THRESHOLD = 0.02

ANALYST_SYSTEM_PROMPT = (
    "You are a senior hedge fund analyst specializing in the 2026 Semiconductor cycle. "
    "Your goal is to protect a $500k principal while riding the AI trend. When analyzing "
    "news for $NVDA, $MU, or $TSM, prioritize 'foundry capacity', 'HBM yield rates', and "
    "'hyperscaler CapEx' over retail hype. Be concise and action-oriented."
)


@dataclass(frozen=True)
class Settings:
    snaptrade_client_id: str | None
    snaptrade_consumer_key: str | None
    snaptrade_user_id: str | None
    snaptrade_user_secret: str | None
    fmp_api_key: str | None
    polygon_api_key: str | None
    grok_api_key: str | None
    anthropic_api_key: str | None

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            snaptrade_client_id=os.getenv("SNAPTRADE_CLIENT_ID"),
            snaptrade_consumer_key=os.getenv("SNAPTRADE_CONSUMER_KEY"),
            snaptrade_user_id=os.getenv("SNAPTRADE_USER_ID"),
            snaptrade_user_secret=os.getenv("SNAPTRADE_USER_SECRET"),
            fmp_api_key=os.getenv("FMP_API_KEY"),
            polygon_api_key=os.getenv("POLYGON_API_KEY"),
            grok_api_key=os.getenv("GROK_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )


SETTINGS = Settings.load()
TICKERS: list[str] = list(TARGET_ALLOCATION.keys())
