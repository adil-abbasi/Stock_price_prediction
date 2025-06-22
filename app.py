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

# Theme Switcher
st.set_page_config(page_title="Stock Forecast App", layout="wide")
theme = st.sidebar.radio("Choose Theme:", options=["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stApp {
            background-color: #0e1117;
        }
        </style>
    """, unsafe_allow_html=True)

# Banner image
st.image("Banner.png", use_column_width=True)

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, LUCK.KA):", value="LUCK.KA")
time_step = st.sidebar.slider("Time Step", min_value=30, max_value=100, value=60, step=5)
forecast_days = st.sidebar.slider("Forecast Days", min_value=5, max_value=30, value=10, step=1)

# Download data
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5 * 365)

data_load_state = st.text("📡 Loading data...")
data = yf.download(ticker, start=start_date, end=end_date)
data_load_state.text("✅ Data loaded successfully!")

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
data = data.astype(float)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Dataset creation
def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i])
        y.append(dataset[i, 3])  # Close price
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, time_step)
n_features = X.shape[2]

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# GPU optional config
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        st.warning(e)

# Model
model = Sequential([
    Bidirectional(LSTM(100, return_sequences=True), input_shape=(time_step, n_features)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
es = EarlyStopping(patience=10, restore_best_weights=True)

with st.spinner("⏳ Training model..."):
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)

# Invert scale
def invert_close_only(preds, scaler, index=3):
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:, index] = preds.ravel()
    return scaler.inverse_transform(dummy)[:, index]

# Predict
y_pred = model.predict(X_test)
y_test_actual = invert_close_only(y_test.reshape(-1, 1), scaler)
y_pred_actual = invert_close_only(y_pred, scaler)

# Forecast future
last_seq = scaled_data[-time_step:].copy()
future_scaled_preds = []
for _ in range(forecast_days):
    input_seq = last_seq.reshape(1, time_step, n_features)
    pred = model.predict(input_seq)[0][0]
    dummy_row = np.zeros_like(last_seq[-1])
    dummy_row[3] = pred
    inverse_pred = scaler.inverse_transform([dummy_row])[0][3]
    new_row = last_seq[-1].copy()
    new_row[3] = pred
    last_seq = np.vstack([last_seq[1:], new_row])
    future_scaled_preds.append(inverse_pred)

# Plot
test_dates = data.index[-len(y_test_actual):]
future_dates = pd.bdate_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=forecast_days)

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

# Forecast Table
forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted Close Price": future_scaled_preds})
st.subheader("📅 Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
st.success(f"✅ RMSE: {rmse:.2f}")
st.success(f"✅ MAE : {mae:.2f}")

# Footer
st.markdown("""
<hr style="margin-top: 40px; margin-bottom: 10px;">
<div style='text-align: center; font-size: 14px; color: gray;'>
    © 2025 | Developed by <b>Adil Abbasi</b> | Built with Streamlit & TensorFlow LSTM. All Rights Reserved.
</div>
""", unsafe_allow_html=True)
