import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load CSV
df = pd.read_csv("data/power.csv")

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)
df = df.sort_values("datetime")

# Select features (multivariate)
features = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity"
]

data = df[features].values

# Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Train-test split (time based)
split = int(len(scaled) * 0.8)
train = scaled[:split]
test = scaled[split:]

# X = all features, y = target (Global_active_power)
X_train = train
y_train = train[:, 0]

X_test = test
y_test = test[:, 0]

# Save files (IMPORTANT)
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

print("Preprocessing completed")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)