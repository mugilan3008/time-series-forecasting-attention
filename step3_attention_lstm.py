import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from tensorflow.keras.models import Model

# -------------------------------------------------
# Load data created from preprocessing step
# -------------------------------------------------
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

print("X_train shape before sequencing:", X_train.shape)

# -------------------------------------------------
# Convert data into time-series sequences
# -------------------------------------------------
TIME_STEPS = 5

def create_sequences(X, y, steps):
    X_seq = []
    y_seq = []

    for i in range(len(X) - steps):
        X_seq.append(X[i:i + steps])
        y_seq.append(y[i + steps])

    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)

print("X_train shape after sequencing:", X_train_seq.shape)
print("y_train shape after sequencing:", y_train_seq.shape)

# -------------------------------------------------
# Build Attention-based LSTM model
# -------------------------------------------------
inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

# LSTM layer
lstm_output = LSTM(64, return_sequences=True)(inputs)

# Simple attention using average across time steps
attention_output = Lambda(lambda x: tf.reduce_mean(x, axis=1))(lstm_output)

# Final prediction layer
output = Dense(1)(attention_output)

model = Model(inputs, output)
model.compile(optimizer="adam", loss="mse")

# -------------------------------------------------
# Save model (training handled in next step)
# -------------------------------------------------
model.save("data/attention_lstm_model.h5")

print("Attention LSTM model created and saved successfully")