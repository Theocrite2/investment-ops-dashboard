# Financial Analytics Platform

A financial analytics platform combining statistical analysis of thematic asset portfolios, NLP-based geopolitical risk scoring from live news headlines, prediction market intelligence via Polymarket, and a Gemini-powered AI assistant for natural language data queries.

Live demo: https://investment-ops-dashboard.streamlit.app/

---

## What it does

**Tab 1 — Thematic Portfolios**
Five themes: AI & Tech, Crypto, Energy & Commodities, Defence, Traditional/Macro. For each asset: price and return history, rolling volatility, maximum drawdown, returns distribution vs normal distribution, Sharpe ratio, correlation matrix within the theme, and a 15-day ARIMA(1,1,1) price forecast with confidence interval. Every method includes an inline explanation of what it measures, how it's calculated, how to interpret it, and its limitations.

**Tab 2 — Geopolitical Risk**
Daily risk scores across five categories (War & Conflict, Energy Disruption, Trade & Sanctions, Monetary Policy, Tech Regulation) derived from live financial news headlines via the GNews API, scored with TextBlob sentiment analysis. Includes a 30-day heatmap, headline-level detail per category, and individual trend charts.

**Tab 3 — Prediction Markets**
Live Polymarket probabilities on macro and geopolitical events, filtered to exclude sports and irrelevant markets. Includes an AI interpretation feature for asking what a given market probability implies for specific assets.

**Tab 4 — AI Assistant**
Natural language queries answered by Gemini, using live context from asset prices, risk scores, and Polymarket data. Includes a set of pre-built example prompts covering portfolio analysis, geopolitical risk, and prediction market interpretation.

---

## Data

Asset prices are real, fetched daily from Yahoo Finance via yfinance. News headlines are fetched daily via the GNews API. Polymarket probabilities are fetched live from the Polymarket Gamma API. All data refreshes automatically every weekday via GitHub Actions.

---

## Statistical Methods

- **Rolling volatility** — 30-day rolling standard deviation of daily returns, annualised (× √252 × 100)
- **Maximum drawdown** — percentage decline from the rolling peak price
- **Returns distribution** — histogram of daily returns compared against a fitted normal distribution, used to assess fat tails
- **Sharpe ratio** — (annualised return − 2% risk-free rate) / annualised volatility
- **Correlation matrix** — Pearson correlation of daily returns across assets in a theme
- **ARIMA(1,1,1) forecast** — time series forecast on log-transformed prices with a 95% confidence interval

---

## Automation

Daily refresh runs via GitHub Actions every weekday at 18:00 CET (`.github/workflows/daily_prices.yml`). Each run:

1. Updates asset prices from Yahoo Finance
2. Fetches headlines and computes risk scores for all five geopolitical categories
3. Updates Polymarket probabilities
4. Appends NAV history for fund reference data

A one-time backfill workflow (`.github/workflows/seed_prices.yml`) populated 6 months of historical asset prices and 30 days of risk score history at launch.

---

## Project structure

```
investment-ops-dashboard/
├── app.py                          # Streamlit dashboard
├── seed_assets.py                  # One-time asset and price history seed
├── update_prices.py                # Daily refresh: prices, risk scores, Polymarket, NAV
├── fix_prices.py                   # One-off historical price backfill helper
├── requirements.txt
├── .gitignore
├── README.md
└── .github/
    └── workflows/
        ├── daily_prices.yml        # Daily automated refresh
        └── seed_prices.yml         # One-time historical backfill
```

---

## Stack

Python 3.13 | pandas | NumPy | statsmodels (ARIMA) | Supabase (PostgreSQL) | Streamlit | Plotly | yfinance | GNews API | TextBlob | Polymarket Gamma API | Google Gemini API | GitHub Actions

## AI-assisted development

Claude was used to design the statistical methodology explanations, debug the yfinance and GNews API integrations, and structure the Gemini prompt context for the AI assistant.
