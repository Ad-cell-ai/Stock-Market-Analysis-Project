import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from features import add_features
from models import get_model

def prepare_data(filepath="../data/raw/AAPL.csv"):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = add_features(df)

    # Labels: 1 if next-day return positive, else 0
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y, df

def train_model(model_name="xgb"):
    X, y, df = prepare_data()
    tscv = TimeSeriesSplit(n_splits=5)

    acc_scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = get_model(model_name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        acc_scores.append(acc)

    print(f"{model_name} avg accuracy: {sum(acc_scores)/len(acc_scores):.3f}")

if __name__ == "__main__":
    train_model("xgb")
