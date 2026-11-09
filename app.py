import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Traffic Congestion & Volume Predictor",
    layout="centered"
)

# Load the trained model safely
@st.cache_resource
def load_model():
    return joblib.load('traffic_xgboost_model.pkl')

model = load_model()

# App Header
st.title("Traffic Congestion & Volume Predictor")
st.markdown("An end-to-end machine learning web application predicting traffic volume and congestion levels for institutional transport analysis.")
st.divider()

# --- SIDEBAR FOR USER INPUTS ---
st.sidebar.header("Control Panel")
st.sidebar.markdown("Select temporal parameters to simulate traffic conditions:")

# 1. Hour selection
hour = st.sidebar.slider("Hour of Day", 0, 23, 8)

# 2. Human-readable Day of Week selection
day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, 
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}
selected_day = st.sidebar.selectbox("Day of Week", list(day_mapping.keys()))
day_of_week = day_mapping[selected_day]
is_weekend = 1 if day_of_week >= 5 else 0

# 3. Human-readable Month selection
month_mapping = {
    "January": 1, "February": 2, "March": 3, "April": 4, 
    "May": 5, "June": 6, "July": 7, "August": 8, 
    "September": 9, "October": 10, "November": 11, "December": 12
}
selected_month = st.sidebar.selectbox("Month", list(month_mapping.keys()))
month = month_mapping[selected_month]

# --- SMART LAG ESTIMATION ---
if is_weekend:
    base_traffic = 1200 + 800 * np.sin(np.pi * hour / 12)
else:
    if 7 <= hour <= 9:
        base_traffic = 4800 + np.random.randint(-100, 100)  # Morning rush peak
    elif 16 <= hour <= 18:
        base_traffic = 5100 + np.random.randint(-100, 100)  # Evening rush peak
    elif 10 <= hour <= 15:
        base_traffic = 3300 + np.random.randint(-100, 100)  # Midday steady flow
    else:
        base_traffic = 1100 + np.random.randint(-50, 50)    # Night / early morning

traffic_lag_1h = base_traffic + np.random.randint(-50, 50)
traffic_lag_24h = base_traffic + np.random.randint(-100, 100)
rolling_mean_3h = base_traffic + np.random.randint(-30, 30)

# Cyclical feature engineering
hour_sin = np.sin(2 * np.pi * hour / 24.0)
hour_cos = np.cos(2 * np.pi * hour / 24.0)

# --- STRICT FEATURE ORDERING ---
# Ensure columns match the exact sequence your model was trained on
feature_columns = [
    'hour', 'day_of_week', 'month', 'is_weekend', 
    'hour_sin', 'hour_cos', 
    'traffic_lag_1h', 'traffic_lag_24h', 'rolling_mean_3h'
]

input_data = pd.DataFrame([[
    hour, day_of_week, month, is_weekend, 
    hour_sin, hour_cos, 
    traffic_lag_1h, traffic_lag_24h, rolling_mean_3h
]], columns=feature_columns)

# --- MAIN DASHBOARD DISPLAY ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected Scenario")
    st.write(f"**Time:** {hour:02d}:00 hours")
    st.write(f"**Day:** {selected_day}")
    st.write(f"**Month:** {selected_month}")
    st.write(f"**Weekend Status:** {'Weekend' if is_weekend else 'Weekday'}")

with col2:
    st.subheader("Historical Context (Auto-Estimated)")
    st.caption("Estimated dynamically to maintain pipeline integrity:")
    st.text(f"1-Hour Lag: {int(traffic_lag_1h)} vehicles")
    st.text(f"24-Hour Lag: {int(traffic_lag_24h)} vehicles")
    st.text(f"3-Hour Rolling Mean: {int(rolling_mean_3h)} vehicles")

st.divider()

# Prediction Action
if st.button("Run Prediction", use_container_width=True):
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Estimated Traffic Volume: **{int(prediction):,} vehicles**")
    
    # Congestion threshold logic
    if prediction > 4500:
        st.error("**Congestion Status: HIGH** — Expect significant delays and bottlenecks.")
    elif prediction > 2500:
        st.warning("**Congestion Status: MEDIUM** — Moderate traffic flow observed.")
    else:
        st.success("**Congestion Status: LOW** — Traffic is moving smoothly and freely.")
