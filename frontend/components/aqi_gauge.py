import streamlit as st
from models.model_utils import get_aqi_category, get_aqi_color


def render_aqi_gauge(predict_data):
    aqi = predict_data["current_aqi"]
    category = predict_data["current_category"]
    color = predict_data["current_color"]
    timestamp = predict_data["current_timestamp"]

    st.markdown(
        f"""
        <div style="
            background-color: {color};
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            color: {'#000' if aqi <= 100 else '#fff'};
        ">
            <h1 style="font-size: 64px; margin: 0;">{int(aqi)}</h1>
            <h3 style="margin: 0;">{category}</h3>
            <p style="margin: 4px 0 0 0; font-size: 12px;">{timestamp}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    aqi_levels = [
        ("Good", "0-50", "#00e400"),
        ("Moderate", "51-100", "#ffff00"),
        ("Unhealthy (Sensitive)", "101-150", "#ff7e00"),
        ("Unhealthy", "151-200", "#ff0000"),
        ("Very Unhealthy", "201-300", "#8f3f97"),
        ("Hazardous", "301+", "#7e0023"),
    ]

    for label, rng, c in aqi_levels:
        active = "font-weight: bold; border: 2px solid #333;" if label.split()[0] in category else ""
        st.markdown(
            f'<div style="background:{c}; padding:4px 8px; border-radius:4px; '
            f'margin-bottom:3px; color:{"#000" if c in ["#00e400","#ffff00"] else "#fff"}; {active}">'
            f'{label} ({rng})</div>',
            unsafe_allow_html=True
        )