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
        closes = raw["Close"].tolist()
        for i, (idx, row) in enumerate(raw.iterrows()):
            dr = round((closes[i] - closes[i-1]) / closes[i-1], 6) if i > 0 else None
            sb.table("asset_prices").upsert({
                "ticker":       ticker,
                "price_date":   str(idx.date()),
                "open":         round(float(row["Open"]), 4),
                "high":         round(float(row["High"]), 4),
                "low":          round(float(row["Low"]),  4),
                "close":        round(float(row["Close"]), 4),
                "volume":       int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                "daily_return": dr,
            }, on_conflict="ticker,price_date").execute()
        print(f"OK {ticker} ({len(raw)} rows)")
    except Exception as e:
        print(f"FAIL {ticker}: {e}")