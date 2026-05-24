import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone


def render_forecast_chart(predict_data):
    forecasts = predict_data["forecasts"]
    current_aqi = predict_data["current_aqi"]
    current_timestamp = predict_data.get("current_timestamp", "")

    try:
        base_time = datetime.fromisoformat(current_timestamp).replace(tzinfo=timezone.utc)
    except Exception:
        base_time = datetime.now(timezone.utc)

    times = [base_time] + [base_time + timedelta(hours=f["horizon_hours"]) for f in forecasts]
    values = [current_aqi] + [f["predicted_aqi"] for f in forecasts]
    categories = [predict_data["current_category"]] + [f["category"] for f in forecasts]

    df = pd.DataFrame({
        "time": times,
        "aqi": values,
        "category": categories
    })

    # Color map
    color_map = {
        "Good": "#00e400",
        "Moderate": "#ffff00",
        "Unhealthy for Sensitive Groups": "#ff7e00",
        "Unhealthy": "#ff0000",
        "Very Unhealthy": "#8f3f97",
        "Hazardous": "#7e0023"
    }

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["aqi"],
        mode="lines",
        name="Predicted AQI",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y:.0f}<extra></extra>"
    ))

    # Colored markers by category
    for cat, color in color_map.items():
        mask = df["category"] == cat
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df[mask]["time"],
                y=df[mask]["aqi"],
                mode="markers",
                name=cat,
                marker=dict(color=color, size=6),
                hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y:.0f}<br>" + cat + "<extra></extra>"
            ))

    # Threshold lines
    fig.add_hline(y=50, line_dash="dot", line_color="#00e400", opacity=0.5,
                  annotation_text="Good/Moderate (50)")
    fig.add_hline(y=100, line_dash="dot", line_color="#ffff00", opacity=0.5,
                  annotation_text="Moderate/Sensitive (100)")
    fig.add_hline(y=150, line_dash="dash", line_color="#ff0000", opacity=0.7,
                  annotation_text="Unhealthy threshold (150)")

    fig.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="Predicted AQI",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Day summary cards
    st.markdown("**72-Hour Summary**")
    col1, col2, col3 = st.columns(3)

    day1 = [f for f in forecasts if 1 <= f["horizon_hours"] <= 24]
    day2 = [f for f in forecasts if 25 <= f["horizon_hours"] <= 48]
    day3 = [f for f in forecasts if 49 <= f["horizon_hours"] <= 72]

    def day_summary(day_forecasts, label):
        if not day_forecasts:
            return
        avg = round(sum(f["predicted_aqi"] for f in day_forecasts) / len(day_forecasts))
        peak = round(max(f["predicted_aqi"] for f in day_forecasts))
        cats = [f["category"] for f in day_forecasts]
        dominant = max(set(cats), key=cats.count)
        color = color_map.get(dominant, "#888")
        st.markdown(
            f"""
            <div style="background:{color};border-radius:8px;padding:12px;
            color:{'#000' if dominant in ['Good','Moderate'] else '#fff'};text-align:center;">
                <b>{label}</b><br>
                Avg: {avg} | Peak: {peak}<br>
                <small>{dominant}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col1:
        day_summary(day1, "Day 1 (Next 24h)")
    with col2:
        day_summary(day2, "Day 2 (24-48h)")
    with col3:
        day_summary(day3, "Day 3 (48-72h)")