"""
seed_assets.py — run once to populate assets and 6 months of price history.
Run AFTER creating the new tables in Supabase.
"""
import os
from datetime import date
import pandas as pd
import yfinance as yf
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

ASSETS = [
    # (ticker, name, theme, asset_class, currency)
    ("NVDA",     "NVIDIA Corporation",          "AI & Tech",             "Equity", "USD"),
    ("MSFT",     "Microsoft Corporation",       "AI & Tech",             "Equity", "USD"),
    ("META",     "Meta Platforms",              "AI & Tech",             "Equity", "USD"),
    ("PLTR",     "Palantir Technologies",       "AI & Tech",             "Equity", "USD"),
    ("AMD",      "Advanced Micro Devices",      "AI & Tech",             "Equity", "USD"),
    ("BTC-USD",  "Bitcoin",                     "Crypto",                "Crypto", "USD"),
    ("ETH-USD",  "Ethereum",                    "Crypto",                "Crypto", "USD"),
    ("SOL-USD",  "Solana",                      "Crypto",                "Crypto", "USD"),
    ("USO",      "US Oil Fund ETF",             "Energy & Commodities",  "ETF",    "USD"),
    ("GLD",      "SPDR Gold Shares",            "Energy & Commodities",  "ETF",    "USD"),
    ("CPER",     "US Copper Index Fund",        "Energy & Commodities",  "ETF",    "USD"),
    ("LMT",      "Lockheed Martin",             "Defence",               "Equity", "USD"),
    ("RHM.DE",   "Rheinmetall AG",              "Defence",               "Equity", "EUR"),
    ("BA",       "Boeing Company",              "Defence",               "Equity", "USD"),
    ("SPY",      "S&P 500 ETF",                 "Traditional/Macro",     "ETF",    "USD"),
    ("TLT",      "iShares 20+ Yr Treasury ETF", "Traditional/Macro",     "ETF",    "USD"),
    ("EFA",      "iShares MSCI EAFE ETF",       "Traditional/Macro",     "ETF",    "USD"),
    ("IWM",      "Russell 2000 ETF",            "Traditional/Macro",     "ETF",    "USD"),
    ("EURUSD=X", "EUR/USD",                     "Traditional/Macro",     "FX",     "USD"),
]


def insert_assets():
    print("Inserting assets...")
    for a in ASSETS:
        supabase.table("assets").upsert({
            "ticker": a[0], "name": a[1], "theme": a[2],
            "asset_class": a[3], "currency": a[4]
        }).execute()
    print(f"  {len(ASSETS)} assets inserted.")


def fetch_and_insert_prices():
    print("Fetching 6 months of price history from yfinance...")
    tickers = [a[0] for a in ASSETS]

    data = yf.download(tickers, period="6mo", auto_adjust=True, progress=False)

    inserted = 0
    failed = 0
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                df = data.xs(ticker, axis=1, level=1).dropna()
            else:
                df = data.dropna()

            closes = df["Close"].values
            for i, (idx, row) in enumerate(df.iterrows()):
                daily_return = None
                if i > 0 and closes[i - 1] > 0:
                    daily_return = round((float(closes[i]) - float(closes[i - 1])) / float(closes[i - 1]), 6)

                supabase.table("asset_prices").upsert({
                    "ticker":       ticker,
                    "price_date":   str(idx.date()),
                    "open":         round(float(row["Open"]),   4),
                    "high":         round(float(row["High"]),   4),
                    "low":          round(float(row["Low"]),    4),
                    "close":        round(float(row["Close"]),  4),
                    "volume":       int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                    "daily_return": daily_return,
                }, on_conflict="ticker,price_date").execute()
                inserted += 1
        except Exception as e:
            print(f"  Failed {ticker}: {e}")
            failed += 1

    print(f"  {inserted} price rows inserted, {failed} tickers failed.")


if __name__ == "__main__":
    insert_assets()
    fetch_and_insert_prices()
    print("Done.")