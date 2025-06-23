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
import matplotlib.pyplot as plt
import tensorflow as tf
import io

# Page setup
st.set_page_config(page_title="📈 Stock Price Forecast", layout="wide")

# Banner and title
st.image("Banner.png", use_column_width=True)
st.title("📊 Stock Price Prediction & Forecasting")
st.markdown("Enter a valid stock ticker below (e.g., **AAPL**, **LUCK.KA**, **GOOG**) to begin prediction.")

# Sidebar input
ticker = st.text_input("🔎 Enter Stock Ticker:", "")
if not ticker:
    st.warning("Please enter a valid stock ticker symbol to continue.")
    st.stop()

# Sidebar hyperparameters
st.sidebar.title("🔧 Configuration")
time_step = st.sidebar.slider("Time Step", 30, 100, 60, step=5)
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 10, step=1)

# Dates
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5 * 365)

@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

data_load_state = st.text("📡 Loading stock data...")
try:
    data = load_data(ticker, start_date, end_date)
    if data.empty:
        st.error("❌ Invalid ticker symbol or no data found. Please check and try again.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

data_load_state.text("✅ Data loaded successfully!")

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().astype(float)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Dataset preparation
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

# GPU config (optional)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# LSTM Model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(time_step, n_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
es = EarlyStopping(patience=10, restore_best_weights=True)

with st.spinner("⏳ Training model..."):
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)

# Inverse scale helper
def invert_close_only(preds, scaler, index=3):
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:, index] = preds.ravel()
    return scaler.inverse_transform(dummy)[:, index]

# Predict
y_pred = model.predict(X_test, verbose=0)
y_test_actual = invert_close_only(y_test.reshape(-1, 1), scaler)
y_pred_actual = invert_close_only(y_pred, scaler)

# Forecast future
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

# Dates
test_dates = data.index[-len(y_test_actual):]
future_dates = pd.bdate_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=forecast_days)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_dates, y_test_actual, label="Actual", color="blue")
ax.plot(test_dates, y_pred_actual, label="Predicted", color="orange")
ax.plot(future_dates, future_scaled_preds, label="Forecast", color="green", linestyle='--')
ax.set_title(f"{ticker} Stock Price Forecast", fontsize=16)
ax.set_xlabel("Date")
ax.set_ylabel("Price (PKR)" if ".KA" in ticker else "Price (USD)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# 📅 Forecast Table
forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Close Price": np.round(future_scaled_preds, 2)})
st.subheader("🔮 Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

# 📈 Predicted vs Actual Table
predicted_df = pd.DataFrame({
    "Date": test_dates,
    "Actual Close": y_test_actual.ravel(),
    "Predicted Close": y_pred_actual.ravel()
})
st.subheader("📊 Prediction Results")
st.dataframe(predicted_df.tail(100).style.format("{:.2f}"), use_container_width=True)

# 📥 Download CSV buttons
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

st.download_button("📁 Download Forecast CSV", convert_df(forecast_df), file_name=f"{ticker}_forecast.csv", mime="text/csv")
st.download_button("📁 Download Prediction CSV", convert_df(predicted_df), file_name=f"{ticker}_prediction.csv", mime="text/csv")

# ✅ Metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
st.success(f"📉 RMSE: {rmse:.2f}")
st.success(f"📉 MAE : {mae:.2f}")

# Footer
st.markdown("""
<hr style="margin-top: 40px; margin-bottom: 10px;">
<div style='text-align: center; font-size: 14px; color: gray;'>
    © 2025 | Developed by <b>Adil Abbasi</b> | Built with Streamlit, LSTM & YFinance | All Rights Reserved.
</div>
""", unsafe_allow_html=True)
