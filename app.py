import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
import tensorflow as tf
import plotly.graph_objs as go
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from newsapi import NewsApiClient
import time

# --- SETUP ---
st.set_page_config(page_title="📈 Stock Forecast", layout="wide")
st.image("Banner.png", use_column_width=True)
st.title("📊 Stock Price Prediction & Forecasting with Technical Indicators")

# --- INPUT ---
ticker = st.text_input("🔎 Enter Stock Ticker:", "")
if not ticker:
    st.warning("Please enter a valid stock ticker symbol to continue.")
    st.stop()

st.sidebar.title("⚙️ Model Configuration")
time_step = st.sidebar.slider("⏳ Time Step", 30, 100, 60, 5)
forecast_days = st.sidebar.slider("📆 Forecast Days", 5, 30, 10, 1)

# --- Live Stock Price ---
try:
    live_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    st.metric(label=f"📌 Real-time Price ({ticker})", value=f"${live_price:.2f}")
except:
    st.warning("Could not fetch real-time price.")

# --- DISCLAIMER ---
st.warning("📌 **Note:** These predictions are based on past trends, technical indicators, and historical prices. They should not be used for financial decisions without professional advice.")


# --- DATES ---
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5 * 365)

@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# --- LOAD DATA ---
data_state = st.text("📡 Loading stock data...")
try:
    data = load_data(ticker, start_date, end_date)
    if data.empty:
        st.error("❌ Invalid ticker or no data found.")
        st.stop()
except Exception as e:
    st.error(f"❌ Data loading failed: {e}")
    st.stop()
data_state.text("✅ Data loaded!")

# --- PREPROCESS ---
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().astype(float)
ohlc_data = data[['Open', 'High', 'Low', 'Close']].copy()
close_series = data["Close"].squeeze()
data["SMA_20"] = SMAIndicator(close=close_series, window=20).sma_indicator()
data["RSI"] = RSIIndicator(close=close_series, window=14).rsi()
macd = MACD(close=close_series)
data["MACD"] = macd.macd_diff()
bb = BollingerBands(close=close_series)
data["BB_High"] = bb.bollinger_hband()
data["BB_Low"] = bb.bollinger_lband()
data.dropna(inplace=True)

# --- SCALE DATA ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i])
        y.append(dataset[i, 3])  # Close
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, time_step)
n_features = X.shape[2]
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- GPU CONFIG ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tips = [
    "📈 Tip: This tool learns from past stock trends to make predictions.",
    "🔍 Tip: Predictions are not 100% accurate — use them as a guide.",
    "🧠 Tip: The model analyzes patterns from historical prices and volumes.",
    "📅 Tip: The longer the history, the better the model learns patterns.",
    "💡 Tip: Always consider market news and events alongside predictions.",
    "🔄 Tip: We are training the model. It may take a few moments...",
    "📊 Tip: Technical indicators like RSI and MACD help improve forecasting.",
    "⚠️ Tip: This prediction doesn't consider sudden market crashes or news.",
    "📰 Tip: Check the latest news — it may affect future stock movement.",
    "⏳ Tip: Please wait... the model is learning from thousands of data points.",
]


# --- LSTM MODEL ---
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(time_step, n_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
es = EarlyStopping(patience=10, restore_best_weights=True)

# --- TRAINING ---
# --- TRAINING WITH ROTATING TIPS ---
with st.spinner("🔁 Training model..."):
    tip_container = st.empty()
    tip_index = 0
    max_epochs = 50
    total_time = max_epochs * 0.3  # approx 0.3s per epoch on average
    tips_shown = 0
    tip_interval = 5  # seconds

    start_time = time.time()
    next_tip_time = start_time

    # Training in a separate thread-like loop to rotate tips while training
    for epoch in range(max_epochs):
        if time.time() >= next_tip_time:
            tip_container.info(tips[tip_index % len(tips)])
            tip_index += 1
            next_tip_time = time.time() + tip_interval
        model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)

# --- PREDICT & INVERT ---
def invert_close_only(preds, scaler, index=3):
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:, index] = preds.ravel()
    return scaler.inverse_transform(dummy)[:, index]

y_pred = model.predict(X_test, verbose=0)
y_test_actual = invert_close_only(y_test.reshape(-1, 1), scaler)
y_pred_actual = invert_close_only(y_pred, scaler)

# --- FORECAST ---
last_seq = scaled_data[-time_step:].copy()
future_scaled_preds = []
for _ in range(forecast_days):
    input_seq = last_seq.reshape(1, time_step, n_features)
    pred = model.predict(input_seq, verbose=0)[0][0]
    dummy_row = np.zeros_like(last_seq[-1])
    dummy_row[3] = pred
    inverse_pred = scaler.inverse_transform([dummy_row])[0][3]
    new_row = last_seq[-1].copy()
    new_row[3] = pred
    last_seq = np.vstack([last_seq[1:], new_row])
    future_scaled_preds.append(inverse_pred)

# --- DATES ---
test_dates = data.index[-len(y_test_actual):]
future_dates = pd.bdate_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=forecast_days)

# --- PLOTS ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual, name="Actual Close", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_dates, y=y_pred_actual, name="Predicted Close", line=dict(color='orange')))
fig.add_trace(go.Scatter(x=future_dates, y=future_scaled_preds, name="Forecast", line=dict(color='green', dash='dash')))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20", line=dict(color='purple')))
fig.add_trace(go.Scatter(x=data.index, y=data["BB_High"], name="BB High", line=dict(color='gray'), opacity=0.3))
fig.add_trace(go.Scatter(x=data.index, y=data["BB_Low"], name="BB Low", line=dict(color='gray'), opacity=0.3))
fig.update_layout(
    title=f"{ticker} Stock Forecast & Indicators",
    xaxis_title="Date",
    yaxis_title="Price",
    height=600,
    template='plotly_dark' if st.get_option("theme.base") == "dark" else "plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# --- RSI ---
with st.expander("📉 RSI Chart"):
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="teal")))
    fig_rsi.update_layout(title="RSI (14)", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_rsi, use_container_width=True)

# --- MACD ---
with st.expander("📈 MACD Chart"):
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD Histogram", line=dict(color="orange")))
    st.plotly_chart(fig_macd, use_container_width=True)

# --- TABLES ---
forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Close Price": np.round(future_scaled_preds, 2)})
predicted_df = pd.DataFrame({
    "Date": test_dates,
    "Actual Close": y_test_actual.ravel(),
    "Predicted Close": y_pred_actual.ravel()
})
indicator_df = data[["Close", "SMA_20", "RSI", "MACD", "BB_High", "BB_Low"]].copy()

st.subheader("🔮 Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

st.subheader("📊 Prediction Results")
st.dataframe(predicted_df.tail(100).style.format("{:.2f}"), use_container_width=True)

# --- DOWNLOADS ---
def convert_df(df): return df.to_csv(index=False).encode("utf-8")
st.download_button("📁 Download Forecast CSV", convert_df(forecast_df), f"{ticker}_forecast.csv", "text/csv")
st.download_button("📁 Download Prediction CSV", convert_df(predicted_df), f"{ticker}_prediction.csv", "text/csv")
st.download_button("📁 Download Indicators CSV", convert_df(indicator_df), f"{ticker}_indicators.csv", "text/csv")

# --- METRICS ---
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
st.success(f"📉 RMSE: {rmse:.2f}")
st.success(f"📉 MAE : {mae:.2f}")

# --- NEWS SENTIMENT ---
with st.expander("📰 Latest News Sentiment"):
    try:
        newsapi = NewsApiClient(api_key='f91aaadec6f54629b7d25613589389c8')
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
        for article in articles['articles']:
            st.markdown(f"**[{article['title']}]({article['url']})**  \n*{article['source']['name']} - {article['publishedAt']}*\n{article['description']}\n")
    except Exception as e:
        st.warning("🔒 Could not fetch news. Check NewsAPI key or quota.")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 40px; margin-bottom: 10px;">
<div style='text-align: center; font-size: 14px; color: gray;'>
    © 2025 | Developed by <b>Adil Abbasi</b> | Built with Streamlit, Plotly, LSTM, TA-Lib, and NewsAPI
</div>
""", unsafe_allow_html=True)
