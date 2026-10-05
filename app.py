import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
import google.generativeai as genai

load_dotenv()

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))
GEMINI_KEY   = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)
ai_model = genai.GenerativeModel("gemini-3.6-flash")

st.set_page_config(
    page_title="Financial Analytics Platform",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load(table):
    return pd.DataFrame(supabase.table(table).select("*").execute().data)

@st.cache_data(ttl=1800)
def load_prices_for_ticker(ticker):
    data = supabase.table("asset_prices").select("*").eq("ticker", ticker).order("price_date").execute().data
    df = pd.DataFrame(data)
    if not df.empty:
        df["price_date"] = pd.to_datetime(df["price_date"])
        df = df.sort_values("price_date").reset_index(drop=True)
    return df

@st.cache_data(ttl=60)
def load_prices_for_theme(theme, assets_df):
    tickers = assets_df[assets_df["theme"] == theme]["ticker"].tolist()
    if not tickers:
        return pd.DataFrame()
    data = supabase.table("asset_prices").select("ticker,price_date,close,daily_return").in_("ticker", tickers).order("price_date").execute().data
    df = pd.DataFrame(data)
    if not df.empty:
        df["price_date"] = pd.to_datetime(df["price_date"])
    return df

# ── STATISTICAL FUNCTIONS ─────────────────────────────────────────────────────

def compute_rolling_volatility(returns, window=30):
    return returns.rolling(window).std() * np.sqrt(252) * 100

def compute_drawdown(prices):
    roll_max = prices.cummax()
    return (prices - roll_max) / roll_max * 100

def compute_sharpe(returns, risk_free=0.02):
    ann_return = returns.mean() * 252
    ann_vol    = returns.std() * np.sqrt(252)
    if ann_vol == 0:
        return 0.0
    return round((ann_return - risk_free) / ann_vol, 2)

def compute_arima_forecast(prices, steps=15):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        log_prices = np.log(prices.dropna())
        model = ARIMA(log_prices, order=(1, 1, 1))
        fit   = model.fit()
        forecast = fit.get_forecast(steps=steps)
        mean   = np.exp(forecast.predicted_mean)
        ci     = np.exp(forecast.conf_int(alpha=0.05))
        return mean, ci
    except Exception:
        return None, None

def method_expander(title, what, how, interpret, limitations):
    with st.expander(f"About this method: {title}"):
        st.markdown(f"**What it measures:** {what}")
        st.markdown(f"**How it is calculated:** {how}")
        st.markdown(f"**How to interpret:** {interpret}")
        st.markdown(f"**Limitations:** {limitations}")

# ── LOAD BASE DATA ────────────────────────────────────────────────────────────

assets_df      = load("assets")
risk_scores_df = load("risk_scores")
risk_headlines_df = load("risk_headlines")
polymarket_df  = load("polymarket_markets")
funds_df       = load("funds")
breaks_df      = load("reconciliation_breaks")
ssi_df         = load("settlement_instructions")
trades_df      = load("trades")
nav_hist_df    = load("nav_history")
inst_df        = load("instruments")
cps_df         = load("counterparties")

THEMES = ["AI & Tech", "Crypto", "Energy & Commodities", "Defence", "Traditional/Macro"]

# ── PAGE HEADER ───────────────────────────────────────────────────────────────

st.title("Financial Analytics Platform")
st.caption("Thematic portfolios  |  Geopolitical risk  |  Prediction markets  |  Fund operations  |  AI assistant")

st.markdown("""
A financial analytics platform combining statistical analysis of thematic asset portfolios,
NLP-based geopolitical risk scoring, prediction market intelligence, and institutional fund
operations monitoring. Prices updated daily via automated GitHub Actions pipeline.

**Tab 1 — Thematic Portfolios:** Statistical and quantitative analysis per asset — volatility,
drawdown, returns distribution, Sharpe ratio, correlation, and ARIMA price forecast.  
**Tab 2 — Geopolitical Risk:** Daily risk scores across five categories derived from financial
news headlines using FinBERT (domain-specific NLP model).  
**Tab 3 — Prediction Markets:** Live Polymarket probabilities on macro and geopolitical events.    
**Tab 4 — AI Assistant:** Natural language queries interpreted against all live data.
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Thematic Portfolios", "Geopolitical Risk", "Prediction Markets", "Macro", "AI Assistant"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — THEMATIC PORTFOLIOS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.caption("Statistical and quantitative analysis of pre-selected thematic asset portfolios. Select a theme and an asset to view the full analytical breakdown.")

    if assets_df.empty:
        st.warning("No asset data yet. Run seed_assets.py first.")
        st.stop()

    col_theme, col_asset = st.columns([1, 2])
    selected_theme = col_theme.selectbox("Theme", THEMES)
    theme_assets   = assets_df[assets_df["theme"] == selected_theme]
    asset_options  = dict(zip(theme_assets["name"], theme_assets["ticker"]))
    selected_name  = col_asset.selectbox("Asset", list(asset_options.keys()))
    selected_ticker = asset_options[selected_name]

    prices_df = load_prices_for_ticker(selected_ticker)

    if prices_df.empty:
        st.warning("No price data available for this asset.")
    else:
        closes  = prices_df.set_index("price_date")["close"]
        returns = prices_df.set_index("price_date")["daily_return"].dropna()

        # ── Price History ──
        st.subheader("Price & Return History")
        fig_price = px.line(prices_df, x="price_date", y="close",
                            title=f"{selected_name} — Closing Price",
                            labels={"price_date": "Date", "close": "Price"})
        st.plotly_chart(fig_price, use_container_width=True)

        fig_ret = go.Figure(go.Bar(
            x=returns.index, y=returns * 100,
            marker_color=["#C8A0A0" if r < 0 else "#A0B8A0" for r in returns.values]
        ))
        fig_ret.update_layout(title="Daily Returns (%)", xaxis_title="Date", yaxis_title="Return (%)")
        st.plotly_chart(fig_ret, use_container_width=True)

        method_expander(
            "Price & Return History",
            "Daily closing price and percentage daily return over the available history.",
            "Close price from Yahoo Finance. Daily return = (today's close − yesterday's close) / yesterday's close × 100.",
            "Rising price line = appreciation. Large single bars in the returns chart = high-volatility events (earnings, macro announcements, crises).",
            "Historical price does not predict future price. Price data may have gaps on non-trading days or for less liquid assets."
        )

        st.divider()

        # ── Volatility & Drawdown ──
        col_v, col_d = st.columns(2)

        with col_v:
            st.subheader("Rolling Volatility")
            vol = compute_rolling_volatility(returns)
            fig_vol = px.line(x=vol.index, y=vol.values,
                              labels={"x": "Date", "y": "Annualised Volatility (%)"},
                              title="30-Day Rolling Annualised Volatility")
            st.plotly_chart(fig_vol, use_container_width=True)
            method_expander(
                "Rolling Volatility",
                "How much the asset price fluctuates over a 30-day window, expressed as an annualised percentage.",
                "30-day rolling standard deviation of daily returns × √252 (trading days per year) × 100.",
                "Higher % = more volatile. S&P 500 typically 12–20%. Bitcoin typically 60–90%. Spikes indicate periods of market stress.",
                "Backward-looking: reflects past volatility, not future. Volatility regimes change — a low-volatility period can be followed by sudden spikes."
            )

        with col_d:
            st.subheader("Drawdown from Peak")
            dd = compute_drawdown(closes)
            fig_dd = px.area(x=dd.index, y=dd.values,
                             labels={"x": "Date", "y": "Drawdown (%)"},
                             title="Drawdown from Rolling Peak",
                             color_discrete_sequence=["#C8A0A0"])
            st.plotly_chart(fig_dd, use_container_width=True)
            method_expander(
                "Maximum Drawdown",
                "How far the asset has fallen from its highest point at each date.",
                "(Current price − Rolling maximum price) / Rolling maximum price × 100. Always negative or zero.",
                "0% = at an all-time high for the period. −30% = 30% below the peak. Deep drawdowns take longer to recover — a −50% loss requires a +100% gain to break even.",
                "Depends entirely on the time window. A 6-month drawdown chart cannot capture multi-year bear markets."
            )

        st.divider()

        # ── Returns Distribution ──
        st.subheader("Returns Distribution")
        ret_pct = returns * 100
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=ret_pct, nbinsx=50, name="Actual returns",
            marker_color="#4A7CC7", opacity=0.7, histnorm="probability density"
        ))
        # Normal distribution overlay
        mu, sigma = ret_pct.mean(), ret_pct.std()
        x_range = np.linspace(ret_pct.min(), ret_pct.max(), 200)
        normal_y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
        fig_dist.add_trace(go.Scatter(
            x=x_range, y=normal_y, mode="lines", name="Normal distribution",
            line=dict(color="#C45D0B", width=2)
        ))
        fig_dist.update_layout(title="Distribution of Daily Returns vs Normal Distribution",
                               xaxis_title="Daily Return (%)", yaxis_title="Probability Density")
        st.plotly_chart(fig_dist, use_container_width=True)
        method_expander(
            "Returns Distribution",
            "The frequency distribution of daily returns — are extreme moves rare or common compared to a normal distribution?",
            "Histogram of daily returns (%). A normal distribution with the same mean and standard deviation is overlaid for comparison.",
            "Bars that exceed the normal curve at the extremes (fat tails) indicate the asset experiences more extreme moves than a random walk would predict. Crypto typically shows very fat tails. Bonds show thin tails.",
            "The distribution is estimated from the sample period only. Tail behaviour is inherently hard to estimate — rare events are rare by definition. Past tail risk does not bound future tail risk."
        )

        st.divider()

        # ── Sharpe Ratio ──
        st.subheader("Sharpe Ratio")
        sharpe = compute_sharpe(returns)
        col_s1, col_s2 = st.columns([1, 3])
        col_s1.metric("Sharpe Ratio", sharpe,
                      delta="Good" if sharpe > 1 else ("Negative" if sharpe < 0 else "Below 1"))
        col_s2.markdown(f"""
**Interpretation:**
- Above 2.0: excellent risk-adjusted return
- 1.0 to 2.0: good
- 0 to 1.0: return does not compensate well for the risk taken
- Negative: losing money relative to a risk-free investment

Current Sharpe of **{sharpe}** means the asset returns approximately **{sharpe}** units of excess return per unit of volatility.
""")
        method_expander(
            "Sharpe Ratio",
            "Risk-adjusted return — how much excess return the asset generates per unit of risk taken.",
            "Sharpe = (Annualised Return − 2% risk-free rate) / Annualised Volatility. Risk-free rate approximated at 2% (US T-bill proxy).",
            "Higher is better. Comparing two assets: the one with a higher Sharpe delivers more return per unit of risk, making it the more efficient choice in isolation.",
            "Assumes returns are normally distributed (penalises upside and downside equally). Does not account for liquidity risk, concentration risk, or correlation with other holdings. Sensitive to the choice of risk-free rate."
        )

        st.divider()

        # ── Correlation ──
        st.subheader(f"Correlation within {selected_theme}")
        theme_prices = load_prices_for_theme(selected_theme, assets_df)
        if not theme_prices.empty:
            pivot = theme_prices.pivot(index="price_date", columns="ticker", values="daily_return").dropna()
            if pivot.shape[1] > 1:
                corr = pivot.corr()
                fig_corr = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"Correlation Matrix — {selected_theme} (daily returns)"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                method_expander(
                    "Correlation Matrix",
                    "How assets within the theme move relative to each other.",
                    "Pearson correlation coefficient of daily returns between all asset pairs. Range: −1 to +1.",
                    "+1 = move perfectly together (no diversification benefit). 0 = independent. −1 = move opposite (maximum diversification). High correlation within a theme means the assets are exposed to the same risk factors.",
                    "Correlations are unstable and change over time. During market stress, correlations across all assets tend to spike toward +1, reducing diversification precisely when it is needed most."
                )

        st.divider()

        # ── ARIMA Forecast ──
        st.subheader("ARIMA Price Forecast")
        mean_forecast, ci_forecast = compute_arima_forecast(closes)
        if mean_forecast is not None:
            last_date = closes.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=15)
            fig_arima = go.Figure()
            fig_arima.add_trace(go.Scatter(
                x=closes.index[-60:], y=closes.values[-60:],
                mode="lines", name="Historical price", line=dict(color="#4A7CC7")
            ))
            fig_arima.add_trace(go.Scatter(
                x=future_dates, y=mean_forecast.values,
                mode="lines", name="ARIMA forecast", line=dict(color="#C45D0B", dash="dash")
            ))
            fig_arima.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(ci_forecast.iloc[:, 1].values) + list(ci_forecast.iloc[:, 0].values[::-1]),
                fill="toself", fillcolor="rgba(196,93,11,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% confidence interval"
            ))
            fig_arima.update_layout(
                title=f"{selected_name} — 15-Day ARIMA(1,1,1) Forecast",
                xaxis_title="Date", yaxis_title="Price"
            )
            st.plotly_chart(fig_arima, use_container_width=True)
        else:
            st.info("Insufficient data for ARIMA forecast.")

        method_expander(
            "ARIMA Forecast",
            "Statistical time series forecast of the next 15 trading days based on historical price patterns.",
            "ARIMA(1,1,1) — AutoRegressive Integrated Moving Average. The model is fitted on log-transformed prices, differenced once to achieve stationarity. It models each price as a function of the previous price, a lagged forecast error, and a differencing term. Forecasts are transformed back to price level.",
            "Central line = most likely price path under the model. Shaded band = 95% confidence interval. Wider band = higher uncertainty. For volatile assets (crypto), the confidence interval widens rapidly and becomes very wide beyond 5 days.",
            "ARIMA assumes the statistical structure of the series remains constant (stationarity after differencing). It cannot incorporate news events, earnings surprises, policy changes, or any fundamental information. For assets with structural breaks or regime changes, the model is unreliable. This is a statistical model, not financial advice."
        )

        st.error("Disclaimer: ARIMA forecasts are statistical models based on historical patterns only. They do not constitute financial advice and should not be used as the basis for investment decisions.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GEOPOLITICAL RISK
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.caption("Daily risk scores across five geopolitical and macro categories, derived from financial news headlines using FinBERT — a BERT-based model fine-tuned on financial text.")

    CATEGORIES = ["War & Conflict", "Energy Disruption", "Trade & Sanctions", "Monetary Policy", "Tech Regulation"]

    if risk_scores_df.empty:
        st.warning("No risk score data yet. Run the daily price refresh workflow on GitHub Actions.")
    else:
        risk_scores_df["score_date"] = pd.to_datetime(risk_scores_df["score_date"])
        pivot_risk = risk_scores_df.pivot(index="score_date", columns="category", values="risk_score").sort_index().fillna(0.5)

        st.subheader("Risk Score Heatmap — Last 30 Days")
        st.markdown("Scores range from 0 (no negative sentiment) to 1 (maximum negative/risk sentiment). Calculated as the average FinBERT negative-class probability across headlines for that category and day.")

        fig_heat = px.imshow(
            pivot_risk.T,
            color_continuous_scale="Reds",
            zmin=0, zmax=1,
            aspect="auto",
            labels={"color": "Risk Score"},
            title="Geopolitical Risk Heatmap (FinBERT Negative Sentiment)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("About FinBERT and the risk scoring methodology"):
            st.markdown("""
**Model:** ProsusAI/FinBERT — a BERT language model fine-tuned on approximately 10,000 financial news articles from Reuters, The Wall Street Journal, and financial blogs. It classifies text as Positive, Negative, or Neutral with respect to financial market sentiment.

**What is measured:** The daily risk score is the average probability assigned to the "Negative" class across all headlines retrieved for that category. A score of 0.8 means the model assigns an 80% probability of negative financial sentiment to the average headline in that category on that day.

**How to interpret:** Scores above 0.6 indicate elevated negative sentiment — news that a financial analyst would expect to weigh negatively on relevant assets. Sustained high scores across multiple days signal a developing risk theme rather than a one-day event.

**Limitations:** FinBERT is trained on financial market language, not geopolitical analysis. It may score factual reporting on ongoing conflicts differently from crisis escalation. The model does not distinguish between new information and repeated coverage of the same event. Headline selection is based on keyword matching, which may introduce selection bias. The model was trained on data up to approximately 2022 and may not reflect updated financial language conventions.
""")

        st.divider()

        st.subheader("Headlines by Category")
        cat_sel = st.selectbox("Select category", CATEGORIES)

        if not risk_headlines_df.empty:
            risk_headlines_df["headline_date"] = pd.to_datetime(risk_headlines_df["headline_date"])
            cat_headlines = (risk_headlines_df[risk_headlines_df["category"] == cat_sel]
                             .sort_values("headline_date", ascending=False)
                             .head(30))

            if not cat_headlines.empty:
                def hl_colour(row):
                    if row["sentiment_score"] > 0.7: return ["background-color:#C8A0A0"] * len(row)
                    if row["sentiment_score"] > 0.4: return ["background-color:#C8B888"] * len(row)
                    return ["background-color:#A0B8A0"] * len(row)
                st.caption("🔴 High risk sentiment (>0.7)  🟡 Moderate (0.4–0.7)  🟢 Low (<0.4)")
                st.dataframe(
                    cat_headlines[["headline_date","headline","sentiment_score","source"]]
                    .style.apply(hl_colour, axis=1),
                    use_container_width=True
                )
            else:
                st.info("No headlines stored for this category yet.")

        st.subheader("Risk Score Trends")
        for cat in CATEGORIES:
            cat_data = risk_scores_df[risk_scores_df["category"] == cat].sort_values("score_date")
            if not cat_data.empty:
                with st.expander(f"{cat} — trend"):
                    fig_trend = px.line(cat_data, x="score_date", y="risk_score",
                                        title=f"{cat} Risk Score Over Time",
                                        labels={"score_date": "Date", "risk_score": "Risk Score (0–1)"})
                    fig_trend.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTION MARKETS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("🔗 [Browse all markets on Polymarket](https://polymarket.com)")

    with st.expander("What are prediction markets and why do they matter?"):
        st.markdown("""
**Prediction markets** are exchange-like platforms where participants buy and sell contracts that pay $1 if a specific event occurs and $0 if it does not. The market price of a contract converges toward the crowd's estimate of the probability of the event.

**Why they are useful for financial analysis:** Unlike polls or expert forecasts, prediction markets aggregate information from participants who have financial skin in the game. Academic research (Wolfers & Zitzewitz, 2004; Arrow et al., 2008) has shown prediction markets to be better calibrated than expert forecasts on political and economic events.

**Polymarket** is the largest prediction market by volume, operating on the Polygon blockchain. Prices are expressed as probabilities between 0 and 1. A price of 0.72 on "Will the Fed cut rates in 2025?" means the market assigns a 72% probability to that outcome.

**Limitations:** Markets can be thinly traded on niche questions, making prices less reliable. Large participants can temporarily move prices. Markets are only as good as the question specification — poorly defined questions lead to unreliable prices.
""")

    if polymarket_df.empty:
        st.warning("No Polymarket data yet. Trigger the daily price refresh workflow on GitHub Actions.")
    else:
        poly_cats = ["All"] + sorted(polymarket_df["category"].unique().tolist())
        cat_filter = st.selectbox("Filter by category", poly_cats)
        display_poly = polymarket_df if cat_filter == "All" else polymarket_df[polymarket_df["category"] == cat_filter]
        display_poly = display_poly.sort_values("current_prob", ascending=False)

        for _, row in display_poly.iterrows():
            prob = float(row["current_prob"]) if row["current_prob"] else 0.5
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.markdown(f"**{row['title']}**")
            col1.caption(f"Category: {row['category']}")
            col2.metric("Probability", f"{prob:.0%}")
            # Polymarket links removed - unreliable slugs from free API
            st.progress(prob)
            st.divider()

    st.subheader("AI Interpretation of Prediction Market Data")
    poly_question = st.text_input(
        "Ask about a prediction market event and its implications for your portfolio:",
        placeholder="e.g. What does the current Polymarket probability on Fed rate cuts imply for TLT and SPY?"
    )
    if poly_question and not polymarket_df.empty:
        poly_context = polymarket_df[["title","category","current_prob"]].to_string(index=False)
        prompt = f"""
You are a financial analyst. The following are current Polymarket prediction market probabilities:

{poly_context}

Question: {poly_question}

Answer concisely as a financial analyst would, referencing specific market probabilities where relevant.
"""
        with st.spinner("Analysing..."):
            try:
                response = ai_model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")

with tab4:
    st.caption("Yield curve, real yields, liquidity and a semiconductor proxy, cross-checked against gold and AI & Tech exposure. Sourced from FRED and Yahoo Finance.")

    macro_df = load("macro_series")

    if macro_df.empty:
        st.warning("No macro data yet. Run update_prices.py --backfill-macro once, or trigger the daily refresh workflow.")
    else:
        macro_df["series_date"] = pd.to_datetime(macro_df["series_date"])
        macro_pivot = macro_df.pivot_table(index="series_date", columns="series_id", values="value").sort_index()

        st.subheader("Yield Curve & Spreads")
        spread_cols = [c for c in ["DGS10","T10Y2Y","T10Y3M"] if c in macro_pivot.columns]
        if spread_cols:
            fig_yield = go.Figure()
            if "DGS10" in macro_pivot.columns:
                fig_yield.add_trace(go.Scatter(x=macro_pivot.index, y=macro_pivot["DGS10"], mode="lines", name="DGS10 (10Y Yield)", line=dict(color="#4A7CC7")))
            if "T10Y2Y" in macro_pivot.columns:
                fig_yield.add_trace(go.Scatter(x=macro_pivot.index, y=macro_pivot["T10Y2Y"], mode="lines", name="T10Y2Y Spread", yaxis="y2", line=dict(color="#C45D0B")))
            if "T10Y3M" in macro_pivot.columns:
                fig_yield.add_trace(go.Scatter(x=macro_pivot.index, y=macro_pivot["T10Y3M"], mode="lines", name="T10Y3M Spread", yaxis="y2", line=dict(color="#A0B8A0")))
            fig_yield.update_layout(
                title="10Y Yield (left) vs Key Spreads (right)",
                xaxis_title="Date",
                yaxis=dict(title="10Y Yield (%)"),
                yaxis2=dict(title="Spread (%)", overlaying="y", side="right"),
            )
            st.plotly_chart(fig_yield, use_container_width=True)

        method_expander(
            "Yield Curve & Spreads",
            "The 10-year Treasury yield alongside the 10Y-2Y and 10Y-3M spreads — the classic recession-signalling curve.",
            "Daily series from FRED: DGS10, T10Y2Y (10Y minus 2Y), T10Y3M (10Y minus 3M). A negative spread means the curve is inverted.",
            "A spread below zero has historically preceded most US recessions since 1970, typically with a 6 to 24 month lag. Steepening from negative toward zero can reflect either falling short rates (easing) or rising long rates (growth or inflation expectations) — the two imply different things and the chart alone doesn't distinguish them.",
            "A probabilistic historical signal, not a mechanical trigger. Lag length varies widely across cycles and false positives exist."
        )

        st.divider()
        col_ry, col_liq = st.columns(2)

        with col_ry:
            st.subheader("Real Yields vs Gold")
            if "DFII10" in macro_pivot.columns:
                gold_prices = load_prices_for_ticker("GLD")
                fig_ry = go.Figure()
                fig_ry.add_trace(go.Scatter(x=macro_pivot.index, y=macro_pivot["DFII10"],
                                            mode="lines", name="10Y Real Yield (DFII10)", line=dict(color="#C45D0B")))
                if not gold_prices.empty:
                    fig_ry.add_trace(go.Scatter(x=gold_prices["price_date"], y=gold_prices["close"],
                                                mode="lines", name="Gold (GLD)", yaxis="y2", line=dict(color="#D4AF37")))
                fig_ry.update_layout(
                    title="Real Yields vs Gold Price", xaxis_title="Date",
                    yaxis=dict(title="Real Yield (%)"),
                    yaxis2=dict(title="GLD Price ($)", overlaying="y", side="right"),
                )
                st.plotly_chart(fig_ry, use_container_width=True)
            method_expander(
                "Real Yields vs Gold",
                "Whether gold is moving in the historically expected inverse relationship with real (inflation-adjusted) yields.",
                "DFII10, the 10-year TIPS yield from FRED, plotted against GLD's closing price on a secondary axis.",
                "Gold pays no yield, so as real yields rise, the opportunity cost of holding gold rises and gold has historically weakened, and vice versa. Gold rising alongside rising real yields signals it's being bought for a different reason — currency, geopolitical, or monetary-risk hedging.",
                "A historical tendency, not a fixed formula. Has decoupled for extended periods, including recently."
            )

        with col_liq:
            st.subheader("Liquidity Trend")
            if "WALCL" in macro_pivot.columns and "M2SL" in macro_pivot.columns:
                walcl_series = macro_pivot["WALCL"].dropna()
                m2_series    = macro_pivot["M2SL"].dropna()
                if not walcl_series.empty and not m2_series.empty:
                    walcl_norm = walcl_series / walcl_series.iloc[0] * 100
                    m2_norm    = m2_series / m2_series.iloc[0] * 100
                    fig_liq = go.Figure()
                    fig_liq.add_trace(go.Scatter(x=walcl_norm.index, y=walcl_norm.values, mode="lines", name="Fed Balance Sheet (indexed)"))
                    fig_liq.add_trace(go.Scatter(x=m2_norm.index, y=m2_norm.values, mode="lines", name="M2 Money Supply (indexed)"))
                    fig_liq.update_layout(title="Liquidity Proxy (Indexed to 100)", xaxis_title="Date", yaxis_title="Index")
                    st.plotly_chart(fig_liq, use_container_width=True)
                else:
                    st.info("Not enough liquidity data points yet.")
            method_expander(
                "Liquidity Trend",
                "Whether the broader pool of dollar liquidity underpinning asset prices is expanding or contracting.",
                "Fed total assets (WALCL) and M2 money supply (M2SL) from FRED, indexed to 100 at the start of the window so both are comparable on one scale.",
                "A rising line means expanding liquidity — historically supportive for risk assets and crypto. A falling line means tightening conditions.",
                "A simplified two-series proxy, not the full global liquidity picture (ECB, PBoC, BoJ balance sheets are excluded). Directional, not precise."
            )

        st.divider()
        st.subheader("Semiconductor Proxy vs AI & Tech Exposure")
        if "SOXX" in macro_pivot.columns:
            fig_semi = px.line(x=macro_pivot.index, y=macro_pivot["SOXX"],
                               labels={"x": "Date", "y": "SOXX Price ($)"}, title="iShares Semiconductor ETF (SOXX)")
            st.plotly_chart(fig_semi, use_container_width=True)
        method_expander(
            "Semiconductor Proxy",
            "A real-economy read on global chip demand, used as context for the AI & Tech theme.",
            "SOXX (iShares Semiconductor ETF) daily close from Yahoo Finance, tracking a basket of US-listed semiconductor companies.",
            "Chip demand sits upstream of most AI and tech hardware. SOXX strength alongside NVDA/AMD strength in Tab 1 corroborates the AI & Tech move; divergence is worth investigating before treating either move as confirmed.",
            "A market price reflecting investor expectations, not a trade-flow or production number. Export-based data such as South Korean semiconductor shipments would be a stronger complement but has no reliable free source, so it's excluded here."
        )

        st.divider()
        st.subheader("Signal Convergence")
        try:
            spread_now = macro_pivot["T10Y2Y"].dropna().iloc[-1]
            spread_30d = macro_pivot["T10Y2Y"].dropna().iloc[-22] if len(macro_pivot["T10Y2Y"].dropna()) > 22 else spread_now
            liq_now    = macro_pivot["WALCL"].dropna().iloc[-1]
            liq_start  = macro_pivot["WALCL"].dropna().iloc[0]
            ry_now     = macro_pivot["DFII10"].dropna().iloc[-1]
            ry_30d     = macro_pivot["DFII10"].dropna().iloc[-22] if len(macro_pivot["DFII10"].dropna()) > 22 else ry_now

            curve_state = "inverted" if spread_now < 0 else "positive"
            curve_dir   = "steepening" if spread_now > spread_30d else "flattening"
            liq_dir     = "expanding" if liq_now > liq_start else "contracting"
            ry_dir      = "rising" if ry_now > ry_30d else "falling"

            st.info(f"""
The curve is **{curve_state}** and **{curve_dir}** (10Y-2Y at {spread_now:.2f}%). Liquidity is **{liq_dir}** over the displayed window. Real yields are **{ry_dir}** ({ry_now:.2f}%) — the direct driver to weigh against the gold move above.

Rules-based summary of the raw series, not a forecast. Ask the AI Assistant to cross-reference this with the geopolitical risk and Polymarket tabs.
""")
        except Exception:
            st.info("Not enough history yet. Run the backfill once to populate 6 months of data.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.caption("Ask any question about the data in plain English. The assistant has access to current asset prices, risk scores, Polymarket probabilities, and fund operations data.")

    def build_full_context():
        asset_summary = assets_df[["ticker","name","theme","asset_class"]].to_string(index=False)

        # Latest prices per asset
        try:
            latest_prices = (pd.DataFrame(
                supabase.table("asset_prices")
                .select("ticker,price_date,close,daily_return")
                .order("price_date", desc=True)
                .limit(19 * 5)
                .execute().data
            ).groupby("ticker").first().reset_index())
            price_summary = latest_prices[["ticker","price_date","close","daily_return"]].to_string(index=False)
        except Exception:
            price_summary = "Price data unavailable"

        # Risk scores summary
        if not risk_scores_df.empty:
            latest_risk = (risk_scores_df
                          .sort_values("score_date", ascending=False)
                          .groupby("category").first().reset_index()
                          [["category","score_date","risk_score","headline_count"]])
            risk_summary = latest_risk.to_string(index=False)
        else:
            risk_summary = "No risk scores available"

        # Polymarket summary
        if not polymarket_df.empty:
            poly_summary = polymarket_df[["title","category","current_prob"]].head(15).to_string(index=False)
        else:
            poly_summary = "No Polymarket data available"

        try:
            macro_raw = pd.DataFrame(supabase.table("macro_series").select("*").order("series_date", desc=True).limit(60).execute().data)
            macro_summary = macro_raw.sort_values("series_date", ascending=False).groupby("series_id").first().reset_index()[["series_id","series_date","value"]].to_string(index=False) if not macro_raw.empty else "No macro data available"
        except Exception:
            macro_summary = "No macro data available"

        return f"""
You are a financial analyst assistant with access to the following live data.
Answer questions concisely and accurately. Where relevant, reference specific numbers from the data.

THEMATIC ASSETS:
{asset_summary}

LATEST PRICES:
{price_summary}

GEOPOLITICAL RISK SCORES (0=low risk, 1=high risk, FinBERT NLP):
{risk_summary}

PREDICTION MARKET PROBABILITIES (Polymarket):
{poly_summary}

MACRO DATA (FRED yield curve, real yields, liquidity, SOXX semiconductor proxy):
{macro_summary}

"""

    user_question = st.text_input(
        "Ask a question:",
        placeholder="e.g. Which thematic portfolio has the best risk-adjusted return based on Sharpe ratios?"
    )

    if user_question:
        with st.spinner("Analysing..."):
            context = build_full_context()
            prompt  = f"{context}\n\nQuestion: {user_question}\n\nAnswer:"
            try:
                response = ai_model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("**Example questions — click to ask:**")

    example_prompts = [
        "Is it a good time to invest in small caps via the Russell 2000 given current macro conditions?",
        "Which thematic portfolio has the highest risk-adjusted return based on current Sharpe ratios?",
        "Give me a percentage risk estimate of geopolitical events that could impact energy and defence stocks based on current news sentiment.",
        "Where are we in the 4-year Bitcoin cycle based on current prediction market probabilities?",
        "What phase of the US election cycle are we in and which assets are most likely to be impacted?",
        "What does the current Polymarket probability on Fed rate cuts imply for TLT and SPY?",
        "Explain the ARIMA forecast assumptions and what the confidence interval means for a volatile asset like SOL-USD.",
        "Is the correlation between NVDA and AMD high enough that holding both adds no diversification benefit?",
        "Which of the five geopolitical risk categories is showing the strongest deterioration over the last 14 days?",
        "Which fund has the most open reconciliation breaks and what is the total EUR exposure?",
        "What does fat tails in the returns distribution of BTC-USD tell us about tail risk?",
        "Which asset in the portfolio has the highest drawdown from peak right now?",
        "What is the current 10-year minus 2-year yield spread, and is the curve inverted or steepening?",
        "How is gold responding to the current move in real yields?",
        "Is the semiconductor sector confirming or diverging from NVIDIA and AMD's recent performance?",
        "Are liquidity conditions from the Fed balance sheet and M2 currently expanding or contracting?",
    ]

    for prompt_text in example_prompts:
        if st.button(prompt_text, key=prompt_text[:50]):
            with st.spinner("Analysing..."):
                context = build_full_context()
                full_prompt = f"{context}\n\nQuestion: {prompt_text}\n\nAnswer:"
                try:
                    response = ai_model.generate_content(full_prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")