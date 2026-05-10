# CLAUDE.md: AI Trend-Following Portfolio Advisor

## Project Overview
A read-only, AI-driven dashboard to manage a $500,000 IRA/401k portfolio focused on 2026 tech and semiconductor trends. The bot provides sentiment analysis, catalyst tracking, and rebalancing suggestions without direct trading access.

## Tech Stack
- **Language:** Python 3.11+
- **Frontend/UI:** Streamlit (for rapid dashboarding)
- **Brokerage Pipe:** SnapTrade API (Read-Only)
- **Financial Data:** Financial Modeling Prep (FMP) & Polygon.io
- **Sentiment/Real-time:** Grok API (xAI) with X Search enabled
- **Reasoning Engine:** Claude 3.5/4.x or Gemini 1.5 Pro

## Target Portfolio Allocation ($500,000)
| Ticker | Weight | Role |
| :--- | :--- | :--- |
| **SPY** | 5% | Anchor |
| **QQQ** | 20% | Growth Core |
| **SMH** | 10% | Sector Momentum |
| **NVDA** | 12% | AI King |
| **TSM** | 10% | Foundry Monopoly |
| **MU** | 10% | Memory Squeeze Play |
| **AVGO** | 8% | ASIC/Networking |
| **MSFT** | 5% | Hyperscaler |
| **META** | 5% | Platforms |
| **GOOGL** | 5% | Search/Cloud |
| **VRT** | 5% | Data Center Infrastructure |
| **CRWD** | 5% | Cybersecurity Tax |

## Data Fetching Cadence (Recommended)
| Service | Frequency | Reason |
| :--- | :--- | :--- |
| **SnapTrade** | 1x per day | IRA holdings don't change fast; sync at Market Close. |
| **FMP (Analyst/Price)** | 4x per day | Update price targets and major data points every 2 hours. |
| **Polygon (News)** | Every 15-30 mins | Monitor for "Breaking" catalysts during market hours. |
| **Grok (X Sentiment)** | 2x per day | Once at Market Open (vibe check) and once at Close. |

## Core Logic Modules

### 1. The Rebalancer (Math)
Compare `Current_Weight` vs. `Target_Weight`.
- **Trigger:** If `abs(Delta) > 2%`, suggest a "Trim" or "Add" action.
- **Output:** Exact dollar amount and shares to trade in Robinhood manually.

### 2. The Catalyst Filter (AI Reasoning)
Input news from Polygon + FMP to an LLM.
- **System Prompt:** "Identify if the news is 'Noise' or a 'Structural Trend Change'. Ignore general macro news unless it affects 10-year yields."
- **Scoring:** Assign a Catalyst Score (-10 to +10).

### 3. The Sentiment Divergence (Grok)
Use Grok's X Search to compare price action to social chatter.
- **Logic:** If Price is DOWN but Grok Sentiment is HIGH among verified financial accounts, flag as a "Buy the Dip" opportunity.

## Security & Privacy
- **No Private Keys in Code:** Use a `.env` file for all API keys.
- **Read-Only:** Ensure SnapTrade tokens are scoped to `read` only.
- **Local Host:** Run the Streamlit app locally or on a private server (e.g., Railway/Heroku with Auth).

## Example AI System Prompt (for Grok/Claude)
> "You are a senior hedge fund analyst specializing in the 2026 Semiconductor cycle. Your goal is to protect a $500k principal while riding the AI trend. When analyzing news for $NVDA, $MU, or $TSM, prioritize 'foundry capacity', 'HBM yield rates', and 'hyperscaler CapEx' over retail hype. Be concise and action-oriented."
