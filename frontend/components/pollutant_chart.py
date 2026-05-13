import streamlit as st
import plotly.graph_objects as go


def render_pollutant_chart(latest_record):
    pollutants = {
        "PM2.5": latest_record.get("pm2_5", 0),
        "PM10": latest_record.get("pm10", 0),
        "NO2": latest_record.get("no2", 0),
        "O3": latest_record.get("o3", 0),
        "CO": latest_record.get("co", 0),
        "SO2": latest_record.get("so2", 0),
    }

    fig = go.Figure(go.Bar(
        x=list(pollutants.keys()),
        y=list(pollutants.values()),
        marker_color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"],
        text=[f"{v:.1f}" for v in pollutants.values()],
        textposition="outside"
    ))

    fig.update_layout(
        yaxis_title="Concentration (μg/m³)",
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)