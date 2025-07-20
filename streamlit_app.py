# crypto_tracker_predictor.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === Section 1: Data Fetching ===

def get_historical_data(coin_id='bitcoin', vs_currency='usd', days='90'):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# === Section 2: Feature Engineering ===

def add_technical_indicators(df):
    df['close'] = df['price']
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = BollingerBands(close=df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # 1 = buy, 0 = sell
    df.dropna(inplace=True)
    return df

# === Section 3: Model Training ===

def train_model(df):
    features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, report

# === Section 4: Streamlit UI ===

st.title("🔍 Crypto Tracker & Buy/Sell Predictor")
coin = st.selectbox("Select a coin:", ['bitcoin', 'ethereum', 'solana'])
days = st.slider("Days of history to fetch:", min_value=30, max_value=365, value=90)

with st.spinner("Fetching data and calculating indicators..."):
    df = get_historical_data(coin, days=days)
    df = add_technical_indicators(df)
    model, report = train_model(df)

st.subheader("📊 Price & Indicators Preview")
st.line_chart(df[['price', 'bb_upper', 'bb_lower']])

st.subheader("🤖 Model Evaluation")
st.json(report)

st.subheader("📈 Latest Prediction Suggestion")
latest = df.iloc[-1:]
pred = model.predict(latest[['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']])
if pred[0] == 1:
    st.success("💰 Suggestion: BUY")
else:
    st.error("🚨 Suggestion: SELL")
