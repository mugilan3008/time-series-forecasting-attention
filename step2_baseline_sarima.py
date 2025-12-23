import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load processed data
train = np.load("data/train.npy")
test = np.load("data/test.npy")

# Target column (global_active_power)
y_train = train[:, 0]
y_test = test[:, 0]

# SARIMA model (baseline)
model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0)   # safe for small data
)

results = model.fit(disp=False)

# Forecast
forecast = results.forecast(steps=len(y_test))

# Evaluation
mae = mean_absolute_error(y_test, forecast)
rmse = np.sqrt(mean_squared_error(y_test, forecast))

print("SARIMA MAE:", mae)
print("SARIMA RMSE:", rmse)