import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load('traffic_xgboost_model.pkl')

st.title("🚦 Traffic Congestion & Volume Predictor")
st.write("Predicting traffic volume based on temporal and weather features for your university project.")

# User inputs for the web interface
hour = st.slider("Select Hour of Day", 0, 23, 12)
day_of_week = st.selectbox("Day of Week (0=Monday, 6=Sunday)", [0, 1, 2, 3, 4, 5, 6])
is_weekend = 1 if day_of_week >= 5 else 0
month = st.slider("Select Month", 1, 12, 6)

# Cyclical encoding computation (matching your training pipeline)
hour_sin = np.sin(2 * np.pi * hour / 24.0)
hour_cos = np.cos(2 * np.pi * hour / 24.0)

# Input fields for historical features
traffic_lag_1h = st.number_input("Traffic Volume 1 Hour Ago", value=3000.0)
traffic_lag_24h = st.number_input("Traffic Volume Yesterday Same Time", value=3200.0)
rolling_mean_3h = st.number_input("3-Hour Rolling Mean Traffic", value=3100.0)

# Create input DataFrame for the model
input_data = pd.DataFrame([[
    hour, day_of_week, month, is_weekend, hour_sin, hour_cos, 
    traffic_lag_1h, traffic_lag_24h, rolling_mean_3h
]], columns=[
    'hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos', 
    'traffic_lag_1h', 'traffic_lag_24h', 'rolling_mean_3h'
])

# Prediction button
if st.button("Predict Traffic Volume"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Traffic Volume: **{int(prediction)} vehicles**")
    
    if prediction > 4500:
        st.error("Congestion Level: **HIGH** 🔴")
    elif prediction > 2500:
        st.warning("Congestion Level: **MEDIUM** 🟡")
    else:
        st.success("Congestion Level: **LOW** 🟢")
