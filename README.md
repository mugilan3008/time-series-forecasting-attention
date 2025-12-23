Advanced Time Series Forecasting using SARIMA and Attention LSTM

Project Overview
This project is done to understand time series forecasting using two different approaches.
First, a SARIMA model is implemented as a traditional statistical baseline.
Next, an LSTM model with Attention mechanism is built to see whether deep learning can capture patterns better than SARIMA.
The main idea of this project is comparison, not just accuracy.

Dataset
The dataset contains time-based power consumption values.
Since this is time series data, random splitting is avoided.
Steps followed:
Missing values are handled
Data is scaled
Data is split using time order into train and test sets
This helps prevent data leakage.

SARIMA Model
SARIMA is used as a baseline model because it works well for:
linear trends
seasonality
The model is trained using historical data and tested on unseen data.
Performance is measured using MAE and RMSE.
This gives a reference point for comparison with the deep learning model.

Attention-based LSTM Model
After SARIMA, a deep learning model is built using LSTM with Attention.
Why Attention?
In time series, all past values are not equally important
Attention helps the model focus on more relevant time steps
Model details:
LSTM layer with sequence output
Attention layer to weight time steps
Final LSTM + Dense layer for prediction
The model is trained for 10 epochs with a batch size of 32.

Evaluation
Both SARIMA and LSTM models are evaluated on the same test data.
Metrics used:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
These metrics help compare the performance of both models clearly.

Results
SARIMA performs well for simple and seasonal patterns.
The Attention-based LSTM shows better ability to learn complex temporal relationships.
This comparison shows how deep learning can improve forecasting when data patterns are non-linear.

Key Learnings
Time-based split is very important in time series
SARIMA is a strong baseline model
Attention improves LSTM performance and interpretability
Proper evaluation is more important than just building a model

How to Run
pip install -r requirements.txt
python step1_preprocessing.py
python step2_baseline_sarima.py
python step3_attention_lstm.py
python step4_train_evaluate.py

project structure

advanced_time_series/
│
├── data/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
│
├── step1_preprocessing.py
├── step2_baseline_sarima.py
├── step3_attention_lstm.py
├── step4_train_evaluate.py
├── report.txt
├── requirements.txt
└── README.md