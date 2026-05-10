# Portfolio Analysis

AI trend-following portfolio advisor — a read-only Streamlit dashboard for managing a tech/semiconductor-focused IRA/401k portfolio. Provides sentiment analysis, catalyst tracking, and rebalancing suggestions without direct trading access.

See [CLAUDE.md](./CLAUDE.md) for full architecture, target allocation, data cadence, and module design.

## Stack
- Python 3.11+ / Streamlit
- SnapTrade (read-only brokerage)
- Financial Modeling Prep, Polygon.io
- Grok (xAI) for X sentiment
- Claude / Gemini for reasoning

## Setup
1. Copy `.env.example` to `.env` and fill in API keys
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
