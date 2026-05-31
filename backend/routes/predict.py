from flask import Blueprint, jsonify, current_app
from models.model_utils import (
    load_latest_features, generate_forecast,
    generate_shap_values, get_aqi_category, get_aqi_color
)
from config.settings import PREDICTIONS_COLLECTION, AQI_ALERT_THRESHOLD
from datetime import datetime, timezone

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict")
def predict():
    db = current_app.config["DB"]
    artifact = current_app.config["ARTIFACT"]
    model_doc = current_app.config["MODEL_DOC"]
    feature_cols = model_doc["feature_cols"]

    # Load at least 50 rows so lag_24h / rolling_24h are fully populated
    X, latest_df = load_latest_features(feature_cols, db=db, n=50)

    # Generate 72-step (3-day) forecast
    forecasts = generate_forecast(artifact, latest_df, feature_cols)

    current_aqi = float(latest_df["aqi"].iloc[-1])
    current_timestamp = latest_df["timestamp"].iloc[-1]

    for f in forecasts:
        f["category"] = get_aqi_category(f["predicted_aqi"])
        f["color"] = get_aqi_color(f["predicted_aqi"])
        f["alert"] = f["predicted_aqi"] > AQI_ALERT_THRESHOLD

    # SHAP — only for the latest observed row
    _, shap_df = generate_shap_values(artifact, X[-1:], feature_cols, background_X=X)
    shap_data = shap_df.to_dict(orient="records") if shap_df is not None else []

    db[PREDICTIONS_COLLECTION].insert_one({
        "timestamp": datetime.now(timezone.utc),
        "current_aqi": current_aqi,
        "forecasts": forecasts,
        "model_type": model_doc["model_type"]
    })

    return jsonify({
        "city": "karachi",
        "current_aqi": current_aqi,
        "current_category": get_aqi_category(current_aqi),
        "current_color": get_aqi_color(current_aqi),
        "current_timestamp": (current_timestamp.isoformat()
                              if hasattr(current_timestamp, "isoformat")
                              else str(current_timestamp)),
        "forecasts": forecasts,
        "shap_importance": shap_data,
        "alert": current_aqi > AQI_ALERT_THRESHOLD,
        "model_type": model_doc.get("model_type", "unknown"),
    })