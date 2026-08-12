"""
update_prices_v2.py — daily refresh script.
Replaces update_prices.py. Run by GitHub Actions every weekday at 18:00 CET.

Updates:
  1. Asset prices (new thematic assets)
  2. NLP geopolitical risk scores from NewsAPI + FinBERT
  3. Polymarket market probabilities
  4. Fund NAV history (existing)
  5. Fund instrument prices (existing)

Required env vars:
  SUPABASE_URL, SUPABASE_KEY, NEWSAPI_KEY
"""
import os
import random
import sys
import json
from datetime import date, timedelta

import requests
import yfinance as yf
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase     = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
NEWSAPI_KEY  = os.environ.get("NEWSAPI_KEY", "")
TODAY        = date.today()

# ── 1. ASSET PRICES ──────────────────────────────────────────────────────────

def update_asset_prices():
    print("Updating asset prices...")
    rows = supabase.table("assets").select("ticker").execute().data
    tickers = [r["ticker"] for r in rows]
    if not tickers:
        print("  No assets found.")
        return

    data = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
    updated = 0
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                df = data.xs(ticker, axis=1, level=1).dropna()
            else:
                df = data.dropna()
            if df.empty:
                continue
            row = df.iloc[-1]
            price_date = str(df.index[-1].date())
            prev_close = float(df.iloc[-2]["Close"]) if len(df) > 1 else None
            daily_return = None
            if prev_close and prev_close > 0:
                daily_return = round((float(row["Close"]) - prev_close) / prev_close, 6)
            supabase.table("asset_prices").upsert({
                "ticker":       ticker,
                "price_date":   price_date,
                "open":         round(float(row["Open"]),   4),
                "high":         round(float(row["High"]),   4),
                "low":          round(float(row["Low"]),    4),
                "close":        round(float(row["Close"]),  4),
                "volume":       int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                "daily_return": daily_return,
            }, on_conflict="ticker,price_date").execute()
            updated += 1
        except Exception as e:
            print(f"  Failed {ticker}: {e}")
    print(f"  {updated}/{len(tickers)} asset prices updated.")


# ── 2. NLP RISK SCORES ───────────────────────────────────────────────────────

RISK_CATEGORIES = {
    "War & Conflict":      ["war", "military conflict", "invasion", "troops", "airstrike", "ceasefire"],
    "Energy Disruption":   ["oil disruption", "gas supply", "OPEC", "energy crisis", "pipeline attack"],
    "Trade & Sanctions":   ["sanctions", "trade war", "tariffs", "export ban", "embargo", "trade restrictions"],
    "Monetary Policy":     ["Federal Reserve", "interest rates", "inflation", "ECB rate", "rate hike", "rate cut"],
    "Tech Regulation":     ["AI regulation", "antitrust tech", "semiconductor ban", "chip export", "big tech fine"],
}

FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"


def get_finbert_score(text):
    """Returns risk score 0-1 (higher = more negative/risky sentiment)."""
    try:
        r = requests.post(FINBERT_URL, json={"inputs": text[:512]}, timeout=15)
        if r.status_code == 200:
            result = r.json()
            if isinstance(result, list) and result:
                scores = {item["label"]: item["score"] for item in result[0]}
                return round(scores.get("negative", 0.5), 4)
    except Exception:
        pass
    return 0.5  # neutral fallback


def fetch_headlines(keywords, page_size=10):
    if not NEWSAPI_KEY:
        return []
    query = " OR ".join(f'"{kw}"' for kw in keywords[:3])
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "language": "en", "sortBy": "publishedAt",
                    "pageSize": page_size, "apiKey": NEWSAPI_KEY},
            timeout=10
        )
        if r.status_code == 200:
            return r.json().get("articles", [])
    except Exception:
        pass
    return []


def update_risk_scores():
    print("Updating NLP risk scores...")
    for category, keywords in RISK_CATEGORIES.items():
        articles = fetch_headlines(keywords)
        if not articles:
            print(f"  No headlines for {category}")
            continue

        scores = []
        for article in articles:
            text = f"{article.get('title','')} {article.get('description','')}"
            score = get_finbert_score(text)
            scores.append(score)
            supabase.table("risk_headlines").insert({
                "headline_date":   str(TODAY),
                "category":        category,
                "headline":        article.get("title", "")[:500],
                "sentiment_score": score,
                "source":          article.get("source", {}).get("name", ""),
            }).execute()

        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.5
        supabase.table("risk_scores").upsert({
            "score_date":     str(TODAY),
            "category":       category,
            "risk_score":     avg_score,
            "headline_count": len(scores),
        }, on_conflict="score_date,category").execute()
        print(f"  {category}: {avg_score:.3f} ({len(scores)} headlines)")


# ── 3. POLYMARKET ─────────────────────────────────────────────────────────────

def update_polymarket():
    print("Updating Polymarket probabilities...")
    try:
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": True, "closed": False, "limit": 100,
                    "order": "volume24hr", "ascending": False},
            timeout=15
        )
        if r.status_code != 200:
            print(f"  Polymarket API returned {r.status_code}")
            return

        markets = r.json()
        relevant_keywords = [
            "war", "conflict", "fed", "rate", "bitcoin", "btc", "recession",
            "election", "oil", "china", "russia", "ukraine", "ai", "nvidia",
            "inflation", "ceasefire", "sanctions", "iran", "israel"
        ]

        saved = 0
        for m in markets:
            title = m.get("question", m.get("title", "")).lower()
            if not any(kw in title for kw in relevant_keywords):
                continue
            try:
                outcome_prices = m.get("outcomePrices", "[]")
                if isinstance(outcome_prices, str):
                    outcome_prices = json.loads(outcome_prices)
                prob = float(outcome_prices[0]) if outcome_prices else 0.5
            except Exception:
                prob = 0.5

            category = "Macro"
            if any(w in title for w in ["war","conflict","ukraine","russia","iran","israel","ceasefire"]):
                category = "War & Conflict"
            elif any(w in title for w in ["fed","rate","inflation","recession"]):
                category = "Monetary Policy"
            elif any(w in title for w in ["bitcoin","btc","crypto","eth"]):
                category = "Crypto"
            elif any(w in title for w in ["oil","energy","opec"]):
                category = "Energy"
            elif any(w in title for w in ["election","president","senate"]):
                category = "Politics"

            market_id = m.get("id", m.get("conditionId", str(saved)))
            supabase.table("polymarket_markets").upsert({
                "market_id":    str(market_id),
                "title":        m.get("question", m.get("title", ""))[:300],
                "category":     category,
                "current_prob": round(prob, 4),
                "url":          f"https://polymarket.com/event/{m.get('slug','')}",
                "last_updated": str(TODAY),
            }).execute()
            saved += 1
            if saved >= 20:
                break

        print(f"  {saved} Polymarket markets updated.")
    except Exception as e:
        print(f"  Polymarket failed: {e}")


# ── 4. FUND NAV HISTORY (existing) ──────────────────────────────────────────

def append_nav_today():
    print("Appending NAV history...")
    funds = supabase.table("funds").select("fund_id,nav_per_share,aum_eur").execute().data
    for fund in funds:
        move    = 1 + random.uniform(-0.015, 0.015)
        new_nav = round(fund["nav_per_share"] * move, 4)
        new_aum = round(fund["aum_eur"] * move, 2)
        supabase.table("funds").update({
            "nav_per_share": new_nav, "aum_eur": new_aum, "nav_date": str(TODAY),
        }).eq("fund_id", fund["fund_id"]).execute()
        supabase.table("nav_history").upsert({
            "fund_id": fund["fund_id"], "nav_per_share": new_nav,
            "aum_eur": new_aum, "nav_date": str(TODAY),
        }, on_conflict="fund_id,nav_date").execute()
    print(f"  NAV updated for {len(funds)} funds.")


# ── 5. FUND INSTRUMENT PRICES (existing) ────────────────────────────────────

FUND_TICKERS = {
    "MC.PA": "FR0000121014", "ASML": "NL0010273215", "NESN.SW": "CH0012221716",
    "SAN.PA": "FR0000120578", "SIE.DE": "DE0007236101", "BNP.PA": "FR0000131104",
    "IBGL.L": "IE00B14X4T88", "IEGA.L": "IE00B4WXJJ64", "DBXE.DE": "LU0321463258",
    "CSH2.PA": "FR0010149120", "XEON.DE": "LU0290358497", "EXX5.DE": "DE0002635307",
}

def update_fund_instrument_prices():
    print("Updating fund instrument prices...")
    tickers = list(FUND_TICKERS.keys())
    data = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
    updated = 0
    for ticker, isin in FUND_TICKERS.items():
        try:
            if len(tickers) > 1:
                df = data.xs(ticker, axis=1, level=1).dropna()
            else:
                df = data.dropna()
            price      = round(float(df["Close"].iloc[-1]), 4)
            price_date = str(df.index[-1].date())
            supabase.table("instruments").update({
                "price": price, "price_date": price_date,
            }).eq("isin", isin).execute()
            updated += 1
        except Exception as e:
            print(f"  Failed {ticker}: {e}")
    print(f"  {updated} fund instrument prices updated.")


# ── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    update_asset_prices()
    update_risk_scores()
    update_polymarket()
    append_nav_today()
    update_fund_instrument_prices()
    print("Done.")