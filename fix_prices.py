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
        df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
        df = df.dropna()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        closes = df["Close"].values.flatten()
        for i, (idx, row) in enumerate(df.iterrows()):
            dr = round((float(closes[i]) - float(closes[i-1])) / float(closes[i-1]), 6) if i > 0 else None
            sb.table("asset_prices").upsert({
                "ticker":       ticker,
                "price_date":   str(idx.date()),
                "open":         round(float(row["Open"]), 4),
                "high":         round(float(row["High"]), 4),
                "low":          round(float(row["Low"]),  4),
                "close":        round(float(closes[i]),   4),
                "volume":       int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                "daily_return": dr,
            }, on_conflict="ticker,price_date").execute()
        print(f"OK {ticker} ({len(df)} rows)")
    except Exception as e:
        print(f"FAIL {ticker}: {e}")