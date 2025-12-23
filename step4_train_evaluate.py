import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Multiply, Softmax
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/power.csv")
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df.set_index('datetime', inplace=True)

data = df[['global_active_power']]

# -----------------------------
# Scaling
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------
# Create sequences
# -----------------------------
def create_sequences(data, time_steps=1):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

TIME_STEPS = 1
X, y = create_sequences(scaled_data, TIME_STEPS)
X = X.reshape((X.shape[0], X.shape[1], 1))

# -----------------------------
# Attention block
# -----------------------------
def attention_block(inputs):
    score = Dense(1)(inputs)
    attention_weights = Softmax(axis=1)(score)
    context = Multiply()([inputs, attention_weights])
    return context

# -----------------------------
# Build model
# -----------------------------
input_layer = Input(shape=(X.shape[1], 1))
lstm_out = LSTM(50, return_sequences=True)(input_layer)
attention_out = attention_block(lstm_out)
lstm_out2 = LSTM(50)(attention_out)
output = Dense(1)(lstm_out2)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# -----------------------------
# Train model
# -----------------------------
model.fit(X, y, epochs=20, batch_size=4, verbose=0)

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(X)

# Inverse transform
y_actual = scaler.inverse_transform(y.reshape(-1, 1))
y_predicted = scaler.inverse_transform(predictions)

# -----------------------------
# Evaluation metrics
# -----------------------------
mae = mean_absolute_error(y_actual, y_predicted)
rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))

print("MAE :", mae)
print("RMSE:", rmse)

# -----------------------------
# Plot Actual vs Predicted
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_actual, label="Actual", marker='o')
plt.plot(y_predicted, label="Predicted", marker='x')
plt.title("Actual vs Predicted Power Consumption")
plt.xlabel("Time Steps")
plt.ylabel("Global Active Power")
plt.legend()
plt.grid(True)
plt.show()