import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("data/power.csv")
df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M")
df["hour"] = df["datetime"].dt.hour

features = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity"
    "hour"
]

df = df[features].dropna()

# Scale data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# -------- TIME SERIES SEQUENCES --------
TIME_STEPS = 5

X, y = [], []
for i in range(len(scaled) - TIME_STEPS):
    X.append(scaled[i:i+TIME_STEPS])
    y.append(scaled[i+TIME_STEPS, 0])  # target

X = np.array(X)
y = np.array(y)

# Train-test split (time based)
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Save
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("Preprocessing done")
print("X_train shape:", X_train.shape)  # MUST be 3D
