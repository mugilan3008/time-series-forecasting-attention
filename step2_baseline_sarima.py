import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

# SARIMA model
model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0)
)

results = model.fit(disp=False)

# Forecast
forecast = results.forecast(steps=len(y_test))

# Evaluation
mae = mean_absolute_error(y_test, forecast)
rmse = mean_squared_error(y_test, forecast) ** 0.5

print("SARIMA MAE:", mae)
print("SARIMA RMSE:", rmse)