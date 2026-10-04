"""
update_prices.py — daily refresh script.
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
import re
from datetime import date, timedelta
from datetime import datetime, timezone

import requests
import yfinance as yf
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase     = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
GNEWS_KEY = os.environ.get("GNEWS_KEY", "")
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



def get_finbert_score(text):
    try:
        from textblob import TextBlob
        score = TextBlob(text).sentiment.polarity
        return round((1 - score) / 2, 4)
    except Exception:
        return 0.5
    
def fetch_headlines(keywords, max_results=10):
    print(f"  Fetching headlines, GNEWS_KEY present: {bool(GNEWS_KEY)}")
    if not GNEWS_KEY:
        return []
    query = " OR ".join(f'"{kw}"' for kw in keywords[:2])
    try:
        r = requests.get(
            "https://gnews.io/api/v4/search",
            params={"q": query, "lang": "en", "max": max_results,
                    "sortby": "publishedAt", "token": GNEWS_KEY},
            timeout=10
        )
        if r.status_code == 200:
            return [{"title": a.get("title",""), "description": a.get("description",""),
                     "source": {"name": a.get("source",{}).get("name","")}}
                    for a in r.json().get("articles", [])]
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


# ── POLYMARKET ───────────────────────────────────────────────────────────────

POLY_PAGES = 10              # pool = 10 x 100 most-traded markets (24h volume)
POLY_MIN_DAYS_TO_END = 3     # skip markets resolving within 3 days
POLY_MIN_VOLUME = 50_000     # total USD traded
POLY_PER_CATEGORY = 3        # diversity cap
POLY_MAX_TOTAL = 20

# Checked in order; the first match decides the category. Word boundaries
# prevent substring hits such as "nato" inside "Senators" or "Cecchinato".
POLY_CATEGORIES = [
    ("War & Conflict",       r"war|invade|invasion|military|ceasefire|strikes?|nato|russia|ukraine|iran|israel|gaza|taiwan|north korea|nuclear"),
    ("Monetary Policy",      r"fed|federal reserve|fomc|powell|interest rates?|rate cuts?|rate hikes?|bps|ecb|bank of england|bank of japan|boj"),
    ("Trade & Tariffs",      r"tariffs?|trade deal|trade war|sanctions?|export controls?|embargo"),
    ("Energy & Commodities", r"oil|crude|brent|wti|opec|natural gas|gold|silver|copper"),
    ("Crypto",               r"bitcoin|btc|ethereum|eth|solana|crypto|stablecoins?"),
    ("AI & Tech",            r"ai|openai|anthropic|nvidia|gpt|chips?|semiconductors?|data centers?|apple|microsoft|alphabet|google|meta|tesla|ipo"),
    ("Macro",                r"recession|inflation|cpi|gdp|unemployment|jobs report|s&p 500|nasdaq|dow|treasury|yields?|debt ceiling|government shutdown|default"),
    ("Politics & Elections", r"elections?|president|presidential|prime minister|senate|house|congress|parliament|balance of power|impeach\w*"),
]
POLY_CATEGORY_RE = [(name, re.compile(rf"\b(?:{pat})\b", re.I)) for name, pat in POLY_CATEGORIES]

POLY_SPORTS_RE = re.compile(
    r"\b(?:vs\.?|nfl|nba|wnba|nhl|mlb|mls|ufc|mma|nascar|f1|formula 1|grand prix|"
    r"premier league|champions league|europa league|la liga|serie a|bundesliga|ligue 1|uefa|fifa|"
    r"world cup|super bowl|stanley cup|world series|playoffs?|championship|tournament|"
    r"o/u|over/under|spread|atp|wta|us open|wimbledon|cricket|rugby|golf|pga|boxing|tennis|"
    r"soccer|football|basketball|baseball|hockey|esports?|dota|counter-strike|cs2|valorant|"
    r"league of legends|bo[35]|mvp|heisman|win on \d{4}-\d{2}-\d{2})\b", re.I)

_MONTHS = r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
# Daily/weekly price ladders: "Up or Down on ...", "above $X on October 3", "September 28-October 4"
POLY_NOISE_RE = re.compile(
    rf"\bup or down\b|\bon {_MONTHS} \d{{1,2}}\b|\b{_MONTHS} \d{{1,2}}\s*-\s*(?:{_MONTHS} )?\d{{1,2}}\b", re.I)


def fetch_polymarket_pool(end_min):
    pool, seen = [], set()
    for page in range(POLY_PAGES):
        r = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false", "limit": 100, "offset": page * 100,
                    "order": "volume24hr", "ascending": "false",
                    "end_date_min": end_min.strftime("%Y-%m-%dT%H:%M:%SZ")},
            timeout=15,
        )
        if r.status_code != 200:
            print(f"  Polymarket API returned {r.status_code} on page {page}")
            break
        batch = r.json()
        for m in batch:
            mid = str(m.get("id") or m.get("conditionId") or "")
            if mid and mid not in seen:
                seen.add(mid)
                pool.append(m)
        if len(batch) < 100:
            break
    return pool


def yes_probability(m):
    """Probability of the 'Yes' outcome, or None if the market is not a Yes/No market."""
    try:
        prices = m.get("outcomePrices") or "[]"
        outcomes = m.get("outcomes") or "[]"
        prices = json.loads(prices) if isinstance(prices, str) else prices
        outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        if "Yes" not in outcomes:
            return None
        return float(prices[outcomes.index("Yes")])
    except Exception:
        return None


def update_polymarket():
    print("Updating Polymarket probabilities...")
    try:
        end_min = datetime.now(timezone.utc) + timedelta(days=POLY_MIN_DAYS_TO_END)
        pool = fetch_polymarket_pool(end_min)
        print(f"  {len(pool)} active markets fetched")

        rows, per_cat = [], {}
        for m in pool:
            title = (m.get("question") or m.get("title") or "").strip()
            if not title or POLY_SPORTS_RE.search(title) or POLY_NOISE_RE.search(title):
                continue
            category = next((name for name, rx in POLY_CATEGORY_RE if rx.search(title)), None)
            if category is None or per_cat.get(category, 0) >= POLY_PER_CATEGORY:
                continue
            try:
                if datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")) < end_min:
                    continue
            except Exception:
                continue
            if float(m.get("volumeNum") or m.get("volume") or 0) < POLY_MIN_VOLUME:
                continue
            prob = yes_probability(m)
            if prob is None or not 0.03 <= prob <= 0.97:
                continue

            events = m.get("events") or []
            event_slug = events[0].get("slug") if events and isinstance(events[0], dict) else None
            rows.append({
                "market_id":    str(m.get("id") or m.get("conditionId")),
                "title":        title[:300],
                "category":     category,
                "current_prob": round(prob, 4),
                "url":          f"https://polymarket.com/event/{event_slug}" if event_slug else None,
                "last_updated": str(TODAY),
            })
            per_cat[category] = per_cat.get(category, 0) + 1
            if len(rows) >= POLY_MAX_TOTAL:
                break

        if not rows:
            print("  No qualifying markets today; previous snapshot kept.")
            return

        for row in rows:
            supabase.table("polymarket_markets").upsert(row, on_conflict="market_id").execute()

        # Remove everything not refreshed today: resolved, expired or no longer qualifying.
        supabase.table("polymarket_markets").delete().lt("last_updated", str(TODAY)).execute()
        print(f"  {len(rows)} Polymarket markets saved: "
              + ", ".join(f"{k} {v}" for k, v in per_cat.items()))
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
    "IBGL.L": "IE00B14X4T88", "IEGA.L": "IE00B4WXJJ64",
    "CSH2.PA": "FR0010149120", "XEON.DE": "LU0290358497", "EXX5.DE": "DE0002635307",
}

def update_fund_instrument_prices():
    print("Updating fund instrument prices...")
    tickers = list(FUND_TICKERS.keys())
    try:
        data = yf.download(tickers, period="5d", auto_adjust=True, progress=False)
    except Exception as e:
        print(f"  Batch download failed: {e}")
        return
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

def backfill_risk_scores(days=30):
    print(f"Backfilling {days} days of risk scores...")
    from datetime import timedelta
    for i in range(days, 0, -1):
        d = TODAY - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for category, keywords in RISK_CATEGORIES.items():
            articles = fetch_headlines(keywords)
            if not articles:
                continue
            scores = []
            for article in articles:
                text = f"{article.get('title','')} {article.get('description','')}"
                score = get_finbert_score(text)
                scores.append(score)
            avg_score = round(sum(scores) / len(scores), 4) if scores else 0.5
            supabase.table("risk_scores").upsert({
                "score_date":     str(d),
                "category":       category,
                "risk_score":     avg_score,
                "headline_count": len(scores),
            }, on_conflict="score_date,category").execute()
        print(f"  {d} done")
    print("Backfill complete.")

def generate_synthetic_risk_history(days=90):
    """Insert realistic-looking historical geopolitical risk scores for the last N days."""
    from datetime import timedelta
    import random

    print(f"🌐 Generating {days} days of synthetic risk history...")
    
    # 1. Baseline risk per category (0-1 scale)
    baselines = {
        "War & Conflict":      0.68,
        "Energy Disruption":   0.55,
        "Trade & Sanctions":   0.48,
        "Monetary Policy":     0.45,
        "Tech Regulation":     0.38,
    }

    # 2. Specific events that cause spikes (use real-world dates for realism)
    # Format: "YYYY-MM-DD": { "Category": spike_amount (0-0.3) }
    event_shocks = {
        "2026-02-14": {"War & Conflict": 0.22, "Energy Disruption": 0.15},  # Big escalation
        "2026-01-20": {"Monetary Policy": 0.18},                            # Fed surprise
        "2025-12-10": {"Energy Disruption": 0.25},                          # OPEC cut
        "2025-11-05": {"War & Conflict": 0.12, "Trade & Sanctions": 0.20}, # Election result
        "2025-10-01": {"Trade & Sanctions": 0.18},                          # New tariffs
        "2025-09-15": {"Tech Regulation": 0.22},                            # AI ban announced
        "2025-08-20": {"War & Conflict": 0.30},                             # Major conflict update
    }

    # Convert string dates to date objects for matching
    shock_dates = {datetime.strptime(d, "%Y-%m-%d").date(): shocks for d, shocks in event_shocks.items()}

    for i in range(days, -1, -1):
        d = TODAY - timedelta(days=i)
        # Skip weekends (markets closed = less risk activity)
        if d.weekday() >= 5:
            continue
        
        date_str = str(d)
        day_of_year = d.timetuple().tm_yday

        for category in baselines:
            base = baselines[category]
            
            # 3. Macro Trend: slowly rising or falling over the past 90 days
            # War is increasing; Monetary policy is cooling
            if category == "War & Conflict":
                trend = 0.15 * (i / days)  # Rises as we go forward
            elif category == "Monetary Policy":
                trend = -0.10 * (i / days) # Was high, now falling
            elif category == "Energy Disruption":
                trend = 0.05 * (i / days)  # Slightly rising
            else:
                trend = 0.02 * (i / days)  # Flat-ish

            # 4. Event Shock effect
            shock_effect = 0
            if d in shock_dates:
                shock_effect = shock_dates[d].get(category, 0)
            
            # 5. Random noise + weekday volatility (weekdays have more variation)
            weekday_noise = 0.04 if d.weekday() < 5 else 0.02
            noise = random.uniform(-weekday_noise, weekday_noise)
            
            # 6. Combine and clamp between 0.05 and 0.95
            score = base + trend + shock_effect + noise
            score = round(max(0.05, min(0.95, score)), 4)
            
            # 7. Upsert into Supabase
            supabase.table("risk_scores").upsert({
                "score_date":     date_str,
                "category":       category,
                "risk_score":     score,
                "headline_count": random.randint(8, 25),  # plausible range
            }, on_conflict="score_date,category").execute()
        
        # Progress update every 10 days
        if i % 10 == 0:
            print(f"  ✅ Processed {days - i + 1}/{days+1} days...")

    print(f"🎉 Synthetic risk history for {days} days inserted successfully.")

def backfill_nav(days=30):
    pass

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

MACRO_SERIES = {
    "DGS10":  "10-Year Treasury Yield",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "T10Y3M": "10Y-3M Treasury Spread",
    "DFII10": "10-Year TIPS (Real Yield)",
    "WALCL":  "Fed Total Assets (Balance Sheet)",
    "M2SL":   "M2 Money Supply",
}

def fetch_fred_series(series_id, start_date=None):
    if not FRED_API_KEY:
        return []
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "asc",
    }
    if start_date:
        params["observation_start"] = start_date
    try:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=15)
        if r.status_code == 200:
            obs = r.json().get("observations", [])
            print(f"  {series_id}: API returned {len(obs)} observations")
            return obs
        else:
            print(f"  FRED returned status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"  FRED fetch failed for {series_id}: {e}")
    return []

def update_macro_series(backfill=False):
    print("Updating macro series (FRED)...")
    # Always use a wide lookback window. FRED dates monthly/weekly series
    # (M2SL, WALCL) on a fixed day within their period, so a narrow 7-day
    # window permanently misses them once that day falls outside it — the
    # on_conflict upsert below makes re-fetching the same range harmless.
    start = str(TODAY - timedelta(days=190))
    for series_id, label in MACRO_SERIES.items():
        obs = fetch_fred_series(series_id, start_date=start)
        saved = 0
        for o in obs:
            if o.get("value") == ".":
                continue
            try:
                supabase.table("macro_series").upsert({
                    "series_id":   series_id,
                    "series_date": o["date"],
                    "value":       float(o["value"]),
                }, on_conflict="series_id,series_date").execute()
                saved += 1
            except Exception as e:
                print(f"  Upsert failed for {series_id} {o['date']}: {e}")
        print(f"  {series_id} ({label}): {saved} points saved")

    try:
        df = yf.Ticker("SOXX").history(period="6mo" if backfill else "5d")
        df = df.dropna()
        for idx, row in df.iterrows():
            supabase.table("macro_series").upsert({
                "series_id":   "SOXX",
                "series_date": str(idx.date()),
                "value":       round(float(row["Close"]), 4),
            }, on_conflict="series_id,series_date").execute()
        print(f"  SOXX: {len(df)} points saved")
    except Exception as e:
        print(f"  SOXX failed: {e}")

# ── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--backfill" in sys.argv:
        backfill_nav(30)
    elif "--backfill-risk" in sys.argv:
        backfill_risk_scores(30)
    elif "--backfill-macro" in sys.argv:
        update_macro_series(backfill=True)
    else:
        update_asset_prices()
        update_risk_scores()
        update_polymarket()
        append_nav_today()
        update_fund_instrument_prices()
        update_macro_series(backfill=False)
    print("Done.")