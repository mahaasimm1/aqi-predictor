import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from models.model_utils import get_aqi_color

def render_forecast_chart(predict_data):
    forecasts = predict_data["forecasts"]
    current_aqi = predict_data["current_aqi"]

    labels = ["Now"] + [f"+{f['horizon_hours']}h" for f in forecasts]
    values = [current_aqi] + [f["predicted_aqi"] for f in forecasts]
    colors = [get_aqi_color(v) for v in values]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels,
        y=values,
        mode="lines+markers+text",
        text=[str(int(v)) for v in values],
        textposition="top center",
        line=dict(color="#1f77b4", width=2),
        marker=dict(color=colors, size=14, line=dict(width=2, color="#333")),
    ))

    fig.add_hline(y=150, line_dash="dash", line_color="red",
                  annotation_text="Unhealthy threshold (150)")

    fig.update_layout(
        xaxis_title="Time Horizon",
        yaxis_title="Predicted AQI",
        yaxis=dict(range=[0, max(max(values) + 30, 200)]),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(len(forecasts))
    for i, (col, f) in enumerate(zip(cols, forecasts)):
        with col:
            st.metric(
                label=f"+{f['horizon_hours']}h",
                value=int(f["predicted_aqi"]),
                delta=int(f["predicted_aqi"] - current_aqi)
            )