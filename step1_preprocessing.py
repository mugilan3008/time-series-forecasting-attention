import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load CSV
df = pd.read_csv("data/power.csv")

print("CSV Columns:")
print(df.columns)
print(df.head())

# Datetime handling (IMPORTANT FIX)
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df.set_index('datetime', inplace=True)

# Target column
data = df[['global_active_power']]

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Sliding window function
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(scaled_data, 1)
print("samples:",X.shape)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)
print("y shape:", y.shape)