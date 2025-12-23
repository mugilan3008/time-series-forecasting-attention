# Advanced Time Series Forecasting using SARIMA and Attention-based LSTM

## Project Overview
This project focuses on forecasting power consumption using advanced time series techniques.
A traditional statistical model (SARIMA) is implemented as a baseline and compared with a
deep learning based Attention-enabled LSTM model.

The objective of the project is to analyze multivariate power consumption data and predict
future values while avoiding data leakage and maintaining academic correctness.

---

## Dataset Description
The dataset used in this project contains real-world household power consumption data.
The data is provided in CSV format and includes the following features:

- Global Active Power  
- Global Reactive Power  
- Voltage  
- Global Intensity  

Missing and invalid values are handled during the preprocessing stage.

---

## Project Structure

advanced_time_series/
│
├── data/
│   ├── power.csv                 # Raw power consumption dataset
│   ├── X_train.npy               # Training input data
│   ├── y_train.npy               # Training target values
│   ├── X_test.npy                # Testing input data
│   ├── y_test.npy                # Testing target values
│   └── attention_lstm_model.h5   # Saved Attention-based LSTM model
│
├── step1_preprocessing.py        # Data cleaning and normalization
├── step2_baseline_sarima.py      # Baseline SARIMA model
├── step3_attention_lstm.py       # Attention-based LSTM model creation
├── step4_train_evaluate.py       # Model training and evaluation
│
├── requirements.txt              # Required Python libraries
├── report.txt                    # Project report
└── README.md                     # Project documentation

---

## Methodology

### Step 1: Data Preprocessing
- Load raw CSV data
- Select relevant features
- Handle missing values
- Normalize data using Min-Max scaling
- Split data into training and testing sets using time-based splitting

### Step 2: Baseline SARIMA Model
- Use Global Active Power as the target variable
- Train a SARIMA model on training data
- Generate forecasts for test data
- Evaluate using MAE and RMSE metrics

### Step 3: Attention-based LSTM Model
- Create time series sequences from multivariate data
- Build an LSTM model with an attention mechanism
- Save the trained model for later evaluation

### Step 4: Model Training and Evaluation
- Load the saved Attention LSTM model
- Train the model on historical sequences
- Evaluate performance using MAE and RMSE
- Compare results with the SARIMA baseline

---

## Evaluation Metrics
The models are evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

These metrics provide a clear comparison between the statistical and deep learning approaches.

---

## Key Features
- Multivariate time series forecasting
- Baseline statistical comparison (SARIMA)
- Attention-based deep learning model
- No data leakage (time-based splitting)
- Modular and readable code structure
- CPU-compatible execution

---

## How to Run the Project

1. Install required libraries:
Install all the required Python packages using the following command:
pip install -r requirements.txt

2.Run data preprocessing
This step cleans the dataset, handles missing values, and performs normalization.
python step1_preprocessing.py

3.Run baseline SARIMA model
This step builds and evaluates the traditional SARIMA model.
python step2_baseline_sarima.py

4.Build Attention-based LSTM model
This step creates the deep learning model using LSTM with attention mechanism.
python step3_attention_lstm.py

5.Train and evaluate the models
This step trains the Attention LSTM model and evaluates its performance using MAE and RMSE.
python step4_train_evaluate.py
---

## Conclusion
This project demonstrates the effectiveness of combining traditional time series models
with deep learning approaches for power consumption forecasting.
The Attention-based LSTM model captures temporal dependencies more effectively than
the baseline SARIMA model.

---

## Future Scope
- Use larger datasets for improved generalization
- Hyperparameter tuning for LSTM model
- Visualization of attention weights
- Deployment of the trained model

---

## Author
Mugilan G
Project developed as part of an academic submission.