import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------
# Load test data
# ----------------------------------------
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# ----------------------------------------
# Load trained Attention LSTM model
# ----------------------------------------
model = load_model("data/attention_lstm_model.keras")

# ----------------------------------------
# Make predictions
# ----------------------------------------
pred = model.predict(X_test)

# ----------------------------------------
# Evaluation metrics
# ----------------------------------------
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("DL MAE:", mae)
print("DL RMSE:", rmse)

print("Training and evaluation completed successfully")