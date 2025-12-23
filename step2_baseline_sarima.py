import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load preprocessed data
df = pd.read_csv("data/power.csv")

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Target variable
series = df['global_active_power']

# Train-test split
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# SARIMA model
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24)
)

model_fit = model.fit(disp=False)

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Plot
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="SARIMA Forecast")
plt.legend()
plt.title("Baseline SARIMA Forecast")
plt.show()