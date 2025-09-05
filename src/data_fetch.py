import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("../data/raw")

def fetch_data(ticker="AAPL", start="2015-01-01", end="2023-12-31"):
    """Download historical stock data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(DATA_DIR / f"{ticker}.csv")
    return df

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(fetch_data("AAPL").head())
