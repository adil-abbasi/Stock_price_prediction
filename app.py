import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import plotly.graph_objs as go
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from newsapi import NewsApiClient

# --- Streamlit Setup ---
st.markdown(
    """
    <div style="
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #0e7490);
        color: white;
        margin-bottom: 25px;
    ">
        <h1 style="margin: 0;">Stock Price Prediction Dashboard</h1>
        <p style="margin-top: 10px; font-size: 18px;">
            Data analysis, trend visualization, and machine learning-based stock movement prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Input ---
ticker = st.text_input("🔎 Enter Stock Ticker:", "")
if not ticker:
    st.warning("Please enter a valid stock ticker symbol to continue.")
    st.stop()

st.sidebar.title("⚙️ Model Configuration")
time_step = st.sidebar.slider("⏳ Time Step", 30, 100, 60, 5)
forecast_days = st.sidebar.slider("📆 Forecast Days", 5, 30, 10, 1)
mode = st.sidebar.selectbox("🧠 Training Mode", ["Fast", "Accurate"])
max_epochs = 10 if mode == "Fast" else 50

# --- Live Price ---
try:
    live_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    st.metric(label=f"📌 Real-time Price ({ticker})", value=f"${live_price:.2f}")
except:
    st.warning("Could not fetch real-time price.")

# --- Disclaimer ---
st.warning("📌 **Note:** These predictions are based on past trends, technical indicators, and historical prices. They should not be used for financial decisions without professional advice.")

# --- Date Range ---
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5 * 365)

@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

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

# --- Preprocessing ---
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().astype(float)

# ✅ Ensure 1D Pandas Series — this line guarantees no 2D array mistake
close_series = pd.Series(data['Close'].values.flatten(), index=data.index)

# ✅ Indicators (no error guaranteed)
data['SMA_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()
data['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
macd = MACD(close=close_series)
data['MACD'] = macd.macd_diff()
bb = BollingerBands(close=close_series)
data['BB_High'] = bb.bollinger_hband()
data['BB_Low'] = bb.bollinger_lband()

data.dropna(inplace=True)


# --- Scaling ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i])
        y.append(dataset[i, 3])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, time_step)
n_features = X.shape[2]
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- GPU ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

@st.cache_resource(show_spinner=False)
def train_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_step, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(patience=3, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    model.fit(X_train, y_train, epochs=max_epochs, batch_size=32, validation_split=0.1, callbacks=[es, lr], verbose=0)
    return model

with st.spinner("🔁 Training model (cached)..."):
    model = train_model()

# --- Predictions ---
def invert_close_only(preds, scaler, index=3):
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:, index] = preds.ravel()
    return scaler.inverse_transform(dummy)[:, index]

y_pred = model.predict(X_test, verbose=0)
y_test_actual = invert_close_only(y_test.reshape(-1, 1), scaler)
y_pred_actual = invert_close_only(y_pred, scaler)

# --- Forecast ---
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

# --- Dates ---
test_dates = data.index[-len(y_test_actual):]
future_dates = pd.bdate_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=forecast_days)

# --- Plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_dates, y=y_test_actual, name="Actual Close", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_dates, y=y_pred_actual, name="Predicted Close", line=dict(color='orange')))
fig.add_trace(go.Scatter(x=future_dates, y=future_scaled_preds, name="Forecast", line=dict(color='green', dash='dash')))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name="SMA 20", line=dict(color='purple')))
fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], name="BB High", line=dict(color='gray'), opacity=0.3))
fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], name="BB Low", line=dict(color='gray'), opacity=0.3))
fig.update_layout(title=f"{ticker} Stock Forecast & Indicators", xaxis_title="Date", yaxis_title="Price", height=600, template='plotly_dark' if st.get_option("theme.base") == "dark" else "plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- RSI & MACD ---
with st.expander("📉 RSI Chart"):
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color="teal")))
    fig_rsi.update_layout(title="RSI (14)", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_rsi, use_container_width=True)

with st.expander("📈 MACD Chart"):
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name="MACD Histogram", line=dict(color="orange")))
    st.plotly_chart(fig_macd, use_container_width=True)

# --- Tables ---
forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Close Price": np.round(future_scaled_preds, 2)})
predicted_df = pd.DataFrame({"Date": test_dates, "Actual Close": y_test_actual.ravel(), "Predicted Close": y_pred_actual.ravel()})
indicator_df = data[["Close", "SMA_20", "RSI", "MACD", "BB_High", "BB_Low"]].copy()

st.subheader("🔮 Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

st.subheader("📊 Prediction Results")
st.dataframe(predicted_df.tail(100).style.format("{:.2f}"), use_container_width=True)

# --- Downloads ---
def convert_df(df): return df.to_csv(index=False).encode("utf-8")
st.download_button("📁 Download Forecast CSV", convert_df(forecast_df), f"{ticker}_forecast.csv", "text/csv")
st.download_button("📁 Download Prediction CSV", convert_df(predicted_df), f"{ticker}_prediction.csv", "text/csv")
st.download_button("📁 Download Indicators CSV", convert_df(indicator_df), f"{ticker}_indicators.csv", "text/csv")

# --- Metrics ---
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
st.success(f"📉 RMSE: {rmse:.2f}")
st.success(f"📉 MAE : {mae:.2f}")

# --- News ---
with st.expander("📰 Latest News Sentiment"):
    try:
        newsapi = NewsApiClient(api_key='f91aaadec6f54629b7d25613589389c8')
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
        for article in articles['articles']:
            st.markdown(f"**[{article['title']}]({article['url']})**  \n*{article['source']['name']} - {article['publishedAt']}*\n{article['description']}\n")
    except:
        st.warning("⚠️ Could not fetch news. Check NewsAPI key or quota.")

# --- Footer ---
st.markdown("""
<hr style="margin-top: 40px; margin-bottom: 10px;">
<div style='text-align: center; font-size: 14px; color: gray;'>
    © 2025 | Developed by <b>Adil Abbasi</b> | Built with Streamlit, Plotly, LSTM, TA-Lib, and NewsAPI
</div>
""", unsafe_allow_html=True)
