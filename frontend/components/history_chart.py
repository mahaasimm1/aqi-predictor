import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_history_chart(history_data):
    records = history_data["data"]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["aqi"],
        mode="lines",
        name="Actual AQI",
        line=dict(color="#1f77b4", width=2),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.1)"
    ))

    fig.add_hline(y=50, line_dash="dot", line_color="#00e400",
                  annotation_text="Good")
    fig.add_hline(y=100, line_dash="dot", line_color="#ffff00",
                  annotation_text="Moderate")
    fig.add_hline(y=150, line_dash="dot", line_color="#ff7e00",
                  annotation_text="Unhealthy (Sensitive)")

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)