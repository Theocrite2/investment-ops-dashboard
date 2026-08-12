import os
import pandas as pd
import yfinance as yf
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

tickers = [r["ticker"] for r in sb.table("assets").select("ticker").execute().data]

for ticker in tickers:
    try:
        raw = yf.Ticker(ticker).history(period="6mo")
        raw = raw.dropna()
        dates  = raw.index.tolist()
        opens  = raw["Open"].tolist()
        highs  = raw["High"].tolist()
        lows   = raw["Low"].tolist()
        closes = raw["Close"].tolist()
        vols   = raw["Volume"].tolist()

        for i in range(len(dates)):
            dr = round((closes[i] - closes[i-1]) / closes[i-1], 6) if i > 0 else None
            sb.table("asset_prices").upsert({
                "ticker":       ticker,
                "price_date":   str(dates[i].date()),
                "open":         round(float(opens[i]),  4),
                "high":         round(float(highs[i]),  4),
                "low":          round(float(lows[i]),   4),
                "close":        round(float(closes[i]), 4),
                "volume":       int(vols[i]) if not pd.isna(vols[i]) else 0,
                "daily_return": dr,
            }, on_conflict="ticker,price_date").execute()
        print(f"OK {ticker} ({len(dates)} rows)")
    except Exception as e:
        print(f"FAIL {ticker}: {e}")