import streamlit as st
import pandas as pd
from train import prepare_data, get_model
import matplotlib.pyplot as plt

st.title("📈 Stock Market Analysis Dashboard")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
X, y, df = prepare_data(f"../data/raw/{ticker}.csv")
model = get_model("xgb")
model.fit(X, y)
probs = model.predict_proba(X)[:, 1]

df["Predicted_Prob"] = probs
df["Signal"] = (probs > 0.55).astype(int)

st.line_chart(df[["Close"]])
st.line_chart(df[["Predicted_Prob"]])

st.write("Latest Signals:")
st.dataframe(df.tail(10))
