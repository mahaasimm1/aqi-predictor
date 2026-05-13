import streamlit as st
import plotly.graph_objects as go


def render_shap_plot(predict_data):
    shap_data = predict_data.get("shap_importance", [])

    if not shap_data or all(d["shap_value"] == 0 for d in shap_data):
        st.info("SHAP values not available for the current prediction.")
        return

    top = shap_data[:10]
    features = [d["feature"] for d in top]
    values = [d["shap_value"] for d in top]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color="#3498db",
        text=[f"{v:.4f}" for v in values],
        textposition="outside"
    ))

    fig.update_layout(
        xaxis_title="Mean |SHAP value|",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)