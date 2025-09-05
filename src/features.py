import pandas as pd
import ta  # technical analysis library

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical indicators and lag features."""
    df = df.copy()

    # Technical Indicators
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()
    df["ema_20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["bollinger_hband"] = ta.volatility.BollingerBands(df["Close"]).bollinger_hband()
    df["bollinger_lband"] = ta.volatility.BollingerBands(df["Close"]).bollinger_lband()

    # Lag features
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)

    # Calendar features
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.dropna()
    return df

if __name__ == "__main__":
    sample = pd.read_csv("../data/raw/AAPL.csv", index_col=0, parse_dates=True)
    print(add_features(sample).head())
