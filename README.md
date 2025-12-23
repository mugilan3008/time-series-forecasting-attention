# Advanced Time Series Forecasting with Attention-based LSTM

## 📌 Project Overview
This project focuses on *advanced time series forecasting* using both
traditional statistical models and deep learning techniques.

A *baseline SARIMA model* is implemented for comparison, followed by an
*Attention-based LSTM neural network* to improve forecasting accuracy.
The project demonstrates a complete end-to-end pipeline suitable for
*academic evaluation and interview discussion*.

---

## 🎯 Objectives
- Perform time series preprocessing and scaling
- Build a baseline forecasting model using SARIMA
- Design an advanced Attention-based LSTM model
- Compare actual vs predicted values
- Evaluate performance using error metrics

---

## 🗂️ Project Structure
advanced_time_series/ ├── data/ │   └── power.csv ├── step1_preprocessing.py ├── step2_baseline_sarima.py ├── step3_attention_lstm.py ├── step4_train_evaluate.py ├── report.txt ├── README.md └── requirements.txt

---

## ⚙️ Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Statsmodels (SARIMA)
- TensorFlow / Keras (LSTM + Attention)

---

## 🔄 Workflow Description

### Step 1: Data Preprocessing
- Load time series data
- Handle datetime indexing
- Apply Min-Max scaling
- Create sliding window sequences

### Step 2: Baseline Model (SARIMA)
- Implement SARIMA model using statsmodels
- Generate baseline forecasts
- Visualize actual vs forecasted values

### Step 3: Attention-based LSTM Model
- Build LSTM layers using TensorFlow/Keras
- Apply attention mechanism to focus on important time steps
- Train the deep learning model on time series sequences

### Step 4: Training and Evaluation
- Generate predictions on test data
- Evaluate using MAE and RMSE
- Plot actual vs predicted values

---

## 📊 Evaluation Metrics
- *MAE (Mean Absolute Error)*
- *RMSE (Root Mean Square Error)*

These metrics are used to assess forecasting accuracy.

---

## How to Run the Project

1. Install dependencies:
```bash
pip install -r requirements.txt
```


Run preprocessing:
python step1_preprocessing.py

Run baseline SARIMA model:
python step2_baseline_sarima.py

Run Attention-based LSTM:
python step3_attention_lstm.py

Train and evaluate:
python step4_train_evaluate.py

## Conclusion

In this project, an advanced time series forecasting pipeline was implemented using both
a traditional statistical model (SARIMA) and a deep learning model (Attention-based LSTM).

The baseline SARIMA model provides a classical reference for forecasting performance,
while the Attention-based LSTM captures temporal dependencies more effectively.

Experimental results show that the Attention-based LSTM model achieves lower error
values (MAE and RMSE) compared to the SARIMA model, demonstrating its superiority
in modeling complex time series patterns.

This project highlights the effectiveness of deep learning and attention mechanisms
for real-world energy consumption forecasting tasks.

## Future Work

- Increase the dataset size to improve generalization.
- Include additional features such as sub-metering values and reactive power.
- Apply advanced attention mechanisms such as Transformer models.
- Perform hyperparameter tuning for improved forecasting accuracy.

## Author

Name: Mugilan  
Project Type: Advanced Time Series Forecasting  
Model Used: SARIMA, Attention-based LSTM  
Purpose: Academic Project / Interview Demonstration