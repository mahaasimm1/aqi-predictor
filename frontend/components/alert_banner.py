import streamlit as st
from config.settings import AQI_ALERT_THRESHOLD


def render_alert_banner(predict_data):
    current_aqi = predict_data["current_aqi"]
    forecasts = predict_data.get("forecasts", [])

    if current_aqi > AQI_ALERT_THRESHOLD:
        st.error(
            f"HAZARD ALERT: Current AQI is {int(current_aqi)} — {predict_data['current_category']}. "
            f"Avoid outdoor activities. Wear a mask if going outside."
        )

    future_alerts = [f for f in forecasts if f.get("alert")]
    if future_alerts and current_aqi <= AQI_ALERT_THRESHOLD:
        hours = [f["horizon_hours"] for f in future_alerts]
        st.warning(
            f"FORECAST ALERT: AQI predicted to exceed {AQI_ALERT_THRESHOLD} "
            f"in {min(hours)} hours. Plan accordingly."
        )