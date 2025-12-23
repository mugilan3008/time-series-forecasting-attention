import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Permute, Multiply, Flatten
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Load and preprocess data
# -----------------------------
df = pd.read_csv("data/power.csv")

df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df.set_index('datetime', inplace=True)

data = df[['global_active_power']]

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

print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# Attention Mechanism
# -----------------------------
from tensorflow.keras.layers import Softmax

def attention_block(inputs):
    # inputs shape: (batch_size, time_steps, hidden_units)

    score = Dense(1)(inputs)              # (batch, time_steps, 1)
    attention_weights = Softmax(axis=1)(score)  # (batch, time_steps, 1)

    context = Multiply()([inputs, attention_weights])
    return context

# -----------------------------
# Build Attention LSTM Model
# -----------------------------
input_layer = Input(shape=(X.shape[1], 1))
lstm_out = LSTM(50, return_sequences=True)(input_layer)

attention_out = attention_block(lstm_out)

lstm_out2 = LSTM(50)(attention_out)
output = Dense(1)(lstm_out2)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

model.summary()

# -----------------------------
# Train Model
# ----------------