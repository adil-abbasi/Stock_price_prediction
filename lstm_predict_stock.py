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

# Optional GPU config
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Parameters
ticker = "LUCK.KA"
time_step = 60
forecast_days = 10

# Download data
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5 * 365)
data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
data = data.astype(float)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Dataset creation
def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i])
        y.append(dataset[i, 3])  # Close
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, time_step)
n_features = X.shape[2]

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build model
model = Sequential([
    Bidirectional(LSTM(100, return_sequences=True), input_shape=(time_step, n_features)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
es = EarlyStopping(patience=10, restore_best_weights=True)

# Train
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[es], verbose=1)

# Invert scaling
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
    new_row = last_seq[-1].copy()
    new_row[3] = pred  # predicted close
    last_seq = np.vstack([last_seq[1:], new_row])
    future_scaled_preds.append(pred)

future_preds_actual = invert_close_only(np.array(future_scaled_preds).reshape(-1, 1), scaler)

# Plot
test_dates = data.index[-len(y_test_actual):]
future_dates = pd.bdate_range(start=test_dates[-1] + pd.Timedelta(days=1), periods=forecast_days)

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_actual, label="Actual", color="blue")
plt.plot(test_dates, y_pred_actual, label="Predicted", color="orange")
plt.plot(future_dates, future_preds_actual, label="Forecast", color="green", linestyle='--')
plt.title(f"{ticker} Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE : {mae:.2f}")
