import pandas as pd
import numpy as np
from train import prepare_data, get_model

def backtest(model_name="xgb", threshold=0.55, transaction_cost=0.001):
    X, y, df = prepare_data()
    model = get_model(model_name)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    df["signal"] = np.where(probs > threshold, 1, -1)  # long/short

    # Strategy returns
    df["return"] = df["Close"].pct_change()
    df["strategy"] = df["signal"].shift(1) * df["return"]
    df["strategy_after_cost"] = df["strategy"] - transaction_cost

    cum_return = (1 + df["strategy_after_cost"]).cumprod()
    print(f"Final return: {cum_return.iloc[-1]:.2f}x")

    return df, cum_return

if __name__ == "__main__":
    df, cum_return = backtest("xgb")
    cum_return.plot(title="Strategy vs Buy-and-Hold")
