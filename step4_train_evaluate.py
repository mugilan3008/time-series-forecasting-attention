import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# -------------------------------------------------
# Load preprocessed training data
# -------------------------------------------------
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

print("Original X_train shape:", X_train.shape)

# -------------------------------------------------
# Create time-series sequences
# -------------------------------------------------
TIME_STEPS = 3

def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)

print("X_train final shape:", X_train_seq.shape)
print("y_train final shape:", y_train_seq.shape)

# -------------------------------------------------
# Attention-based LSTM Model (SAFE VERSION)
# -------------------------------------------------
inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

lstm_out = LSTM(64, return_sequences=True)(inputs)

# Attention approximation using pooling (NO Lambda)
attention = GlobalAveragePooling1D()(lstm_out)

output = Dense(1)(attention)

model = Model(inputs, output)
model.compile(optimizer="adam", loss="mse")

# Save model (training done in step 4)
model.save("data/attention_lstm_model.h5")

print("Attention LSTM model created and saved successfully")