import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from tensorflow.keras.losses import MeanSquaredError

# --------------------------------------------------
# Load training data only to identify input dimensions
# --------------------------------------------------
X_train = np.load("data/X_train.npy")

# --------------------------------------------------
# Define input layer based on time steps and features
# --------------------------------------------------
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# --------------------------------------------------
# First LSTM layer to capture temporal patterns
# --------------------------------------------------
lstm_out = LSTM(
    units=64,
    return_sequences=True
)(inputs)

# --------------------------------------------------
# Attention mechanism to focus on important time steps
# --------------------------------------------------
attention_output = Attention()(
    [lstm_out, lstm_out]
)

# --------------------------------------------------
# Second LSTM layer for sequence summarization
# --------------------------------------------------
lstm_final = LSTM(
    units=32
)(attention_output)

# --------------------------------------------------
# Output layer for forecasting the target value
# --------------------------------------------------
output = Dense(1)(lstm_final)

# --------------------------------------------------
# Build and compile the Attention-based LSTM model
# --------------------------------------------------
model = Model(inputs=inputs, outputs=output)

model.compile(
    optimizer="adam",
    loss=MeanSquaredError()
)

# --------------------------------------------------
# Save only the model architecture (training in next step)
# --------------------------------------------------
model.save("data/attention_lstm_model.keras")

print("Attention-based LSTM model architecture saved successfully")