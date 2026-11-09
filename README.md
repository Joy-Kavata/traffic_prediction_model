# Traffic Congestion and Volume Predictor

An end-to-end machine learning web application built to predict traffic volume and institutional transit congestion levels based on temporal parameters. Developed as part of an academic software engineering and machine learning project.

## Project Overview
Urban traffic management requires accurate forecasting of vehicle volume to mitigate bottlenecks and optimize transportation infrastructure. This system leverages an optimized XGBoost regression model trained on historical traffic datasets, integrated into a clean, interactive web dashboard built with Streamlit.

## Key Features
* **Real-Time Predictive Modeling:** Instantaneous estimation of traffic volume based on user-defined temporal inputs.
* **Smart Lag Estimation:** Automated derivation of historical context features (1-hour lag, 24-hour lag, and 3-hour rolling means) to maintain machine learning pipeline integrity without requiring manual lookup inputs.
* **Dynamic Congestion Classification:** Automatically categorizes predicted vehicle volume into distinct operational states (Low, Medium, High congestion) with clear advisory warnings.
* **Academic Aesthetic:** Designed with a clean, professional dashboard interface optimized for institutional reporting and project presentations.

## Technology Stack
* **Language & Core Libraries:** Python, Pandas, NumPy, Scikit-Learn
* **Machine Learning Model:** XGBoost Regressor
* **Model Serialization:** Joblib
* **Web Application Framework:** Streamlit
* **Version Control & Hosting:** Git, GitHub, Streamlit Community Cloud

## Repository Structure
```text
├── app.py                     # Main Streamlit web application script
├── requirements.txt           # Python dependency configuration file
├── traffic_xgboost_model.pkl  # Serialized trained XGBoost regression model
├── Traffic_Prediction_Model.ipynb # Google Colab data preprocessing and training notebook
└── Metro_Interstate_Traffic_Volume.csv # Historical training dataset
```
Local Installation and Setup
If you wish to run or evaluate this application locally, follow these steps:

Clone the repository:
```text
git clone [https://github.com/Joy-Kavata/traffic_prediction_model.git](https://github.com/Joy-Kavata/traffic_prediction_model.git)
cd traffic_prediction_model
```
Install dependencies
```text
pip install -r requirements.txt
```

[Find the Live App Here](https://trafficpredictionmodel-fhu7bqtwj4gmqmxvu36nkk.streamlit.app/)

[Github Repo](https://github.com/Joy-Kavata/traffic_prediction_model)
