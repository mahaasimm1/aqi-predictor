import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import pandas as pd
from datetime import datetime

from frontend.components.aqi_gauge import render_aqi_gauge
from frontend.components.forecast_chart import render_forecast_chart
from frontend.components.pollutant_chart import render_pollutant_chart
from frontend.components.shap_plot import render_shap_plot
from frontend.components.alert_banner import render_alert_banner
from frontend.components.history_chart import render_history_chart
from config.settings import FLASK_PORT

FLASK_BASE_URL = os.getenv("FLASK_BASE_URL", f"http://localhost:{FLASK_PORT}")


# Fetch data from Flask API with error handling

def fetch(endpoint, params=None):
    try:
        response = requests.get(f"{FLASK_BASE_URL}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch {endpoint}: {e}")
        return None


# Page config

st.set_page_config(
    page_title="AQI Predictor — Karachi",
    page_icon="🌫",
    layout="wide"
)

st.title("Air Quality Index Predictor — Karachi")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Auto-refresh every 60 minutes
refresh_interval = st.sidebar.slider("Auto-refresh (minutes)", 10, 120, 60)
st.sidebar.info(f"Dashboard refreshes every {refresh_interval} minutes")

if st.sidebar.button("Refresh Now"):
    st.rerun()

# Fetch data
with st.spinner("Loading predictions..."):
    predict_data = fetch("/predict")
    history_data = fetch("/history", params={"days": 7})

if not predict_data:
    st.error("Could not load prediction data. Make sure the Flask backend is running.")
    st.stop()

# Alert banner (shown at top if hazardous)
render_alert_banner(predict_data)

st.markdown("---")

# Top row: current AQI + forecast
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current AQI")
    render_aqi_gauge(predict_data)

with col2:
    st.subheader("3-Day Forecast")
    render_forecast_chart(predict_data)

st.markdown("---")

# Middle row: pollutants + SHAP
col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("Pollutant Breakdown")
    if history_data and history_data.get("data"):
        latest = history_data["data"][-1]
        render_pollutant_chart(latest)
    else:
        st.info("No pollutant data available.")

with col4:
    st.subheader("Feature Importance (SHAP)")
    render_shap_plot(predict_data)

st.markdown("---")

# Bottom: historical trend
st.subheader("Historical AQI — Last 7 Days")
if history_data and history_data.get("data"):
    render_history_chart(history_data)
else:
    st.info("No historical data available.")

st.markdown("---")
st.caption("Data source: OpenWeatherMap API | Model: Ridge Regression | Store: MongoDB Atlas")