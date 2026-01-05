import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

model = load_model("data/attention_lstm_model.keras")

checkpoint = ModelCheckpoint(
    "data/best_model.keras",
    save_best_only=True,
    monitor="val_loss"
)

logger = CSVLogger("data/training_log.csv")

# Training with basic hyperparameter tuning
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[checkpoint, logger]
)

# Load best model
model = load_model("data/best_model.keras")

# Prediction
pred = model.predict(X_test)

# Metrics
dl_mae = mean_absolute_error(y_test, pred)
dl_rmse = np.sqrt(mean_squared_error(y_test, pred))

# SARIMA metrics (manual entry from step2 output)
sarima_mae = 0.12
sarima_rmse = 0.18

results = pd.DataFrame({
    "Model": ["SARIMA", "Attention LSTM"],
    "MAE": [sarima_mae, dl_mae],
    "RMSE": [sarima_rmse, dl_rmse]
})

results.to_csv("data/final_results.csv", index=False)
print(results)

print("Training, evaluation and comparison completed")
