import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import shap
import gridfs
from pymongo import MongoClient
from config.settings import (
    MONGO_URI, MONGO_DB_NAME, MODELS_COLLECTION,
    FEATURES_COLLECTION, FORECAST_HOURS
)


# MongoDB connection

def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


# Load the best model artifact from MongoDB GridFS

def load_best_model(db=None):
    if db is None:
        db = get_db()

    fs = gridfs.GridFS(db)
    model_doc = db[MODELS_COLLECTION].find_one({"is_best": True})

    if not model_doc:
        raise ValueError("No best model found in MongoDB. Run training pipeline first.")

    gridfs_id = model_doc["gridfs_id"]
    model_bytes = fs.get(gridfs_id).read()
    artifact = pickle.loads(model_bytes)

    return artifact, model_doc


# Load latest features from MongoDB for inference

def load_latest_features(feature_cols, db=None, n=50):
    if db is None:
        db = get_db()

    collection = db[FEATURES_COLLECTION]
    records = list(collection.find(
        {},
        sort=[("timestamp", -1)],
        limit=n
    ))

    if not records:
        raise ValueError("No features found in MongoDB. Run feature pipeline first.")

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Fill missing feature columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols].values, df


# Run prediction using loaded model artifact

def predict(artifact, X):
    model = artifact["model"]
    scaler = artifact["scaler"]

    if scaler is not None:
        X = scaler.transform(X)

    model_type = type(model).__name__

    if model_type in ["Sequential"]:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        preds = model.predict(X).flatten()
    else:
        preds = model.predict(X)

    return preds


# Generate 3-day forecast by running prediction iteratively
# Each step uses the previous prediction as the new lag feature

def generate_forecast(artifact, latest_features_df, feature_cols):
    forecasts = []
    current_row = latest_features_df[feature_cols].values[-1:].copy()
    current_aqi = float(latest_features_df["aqi"].iloc[-1])
    prev_aqi = current_aqi
    aqi_history = list(latest_features_df["aqi"].values[-24:])

    for horizon in FORECAST_HOURS:
        row = current_row.copy()

        # Update lag features with recent predictions
        for col in feature_cols:
            idx = feature_cols.index(col)
            if col == "aqi_lag_1h" and len(aqi_history) >= 1:
                row[0][idx] = aqi_history[-1]
            elif col == "aqi_lag_2h" and len(aqi_history) >= 2:
                row[0][idx] = aqi_history[-2]
            elif col == "aqi_lag_3h" and len(aqi_history) >= 3:
                row[0][idx] = aqi_history[-3]
            elif col == "aqi_lag_6h" and len(aqi_history) >= 6:
                row[0][idx] = aqi_history[-6]
            elif col == "aqi_lag_12h" and len(aqi_history) >= 12:
                row[0][idx] = aqi_history[-12]
            elif col == "aqi_lag_24h" and len(aqi_history) >= 24:
                row[0][idx] = aqi_history[-24]
            elif col == "aqi_rolling_3h" and len(aqi_history) >= 3:
                row[0][idx] = round(sum(aqi_history[-3:]) / 3, 2)
            elif col == "aqi_rolling_6h" and len(aqi_history) >= 6:
                row[0][idx] = round(sum(aqi_history[-6:]) / 6, 2)
            elif col == "aqi_rolling_12h" and len(aqi_history) >= 12:
                row[0][idx] = round(sum(aqi_history[-12:]) / 12, 2)
            elif col == "aqi_rolling_24h" and len(aqi_history) >= 24:
                row[0][idx] = round(sum(aqi_history[-24:]) / 24, 2)
            elif col == "aqi_change_rate":
                row[0][idx] = round((aqi_history[-1] - prev_aqi) / max(prev_aqi, 1), 4)
            elif col == "hour":
                from datetime import datetime, timezone, timedelta
                future_time = datetime.now(timezone.utc) + timedelta(hours=horizon)
                row[0][idx] = future_time.hour
            elif col == "day_of_week":
                from datetime import datetime, timezone, timedelta
                future_time = datetime.now(timezone.utc) + timedelta(hours=horizon)
                row[0][idx] = future_time.weekday()
            elif col == "month":
                from datetime import datetime, timezone, timedelta
                future_time = datetime.now(timezone.utc) + timedelta(hours=horizon)
                row[0][idx] = future_time.month

        preds = predict(artifact, row)
        predicted_aqi = round(float(preds[0]), 2)
        predicted_aqi = max(0, min(500, predicted_aqi))

        forecasts.append({
            "horizon_hours": horizon,
            "predicted_aqi": predicted_aqi
        })

        prev_aqi = aqi_history[-1]
        aqi_history.append(predicted_aqi)
        current_row = row

    return forecasts


# Generate SHAP values for the latest prediction
# Only supported for tree-based and linear models

def generate_shap_values(artifact, X, feature_cols, background_X=None):
    model = artifact["model"]
    scaler = artifact["scaler"]
    model_type = type(model).__name__

    if scaler is not None:
        X_input = scaler.transform(X)
        bg_input = scaler.transform(background_X) if background_X is not None else X_input
    else:
        X_input = X
        bg_input = background_X if background_X is not None else X_input

    try:
        if model_type == "RandomForestRegressor":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)

        elif model_type == "Ridge":
            if background_X is None:
                return None, None
            explainer = shap.LinearExplainer(model, bg_input)
            shap_values = explainer.shap_values(X_input)

        else:
            return None, None

        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "shap_value": np.abs(shap_values[-1])
        }).sort_values("shap_value", ascending=False)

        return shap_values, shap_df

    except Exception as e:
        print(f"SHAP generation failed: {e}")
        return None, None


# Get AQI category label from numeric AQI

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# Get hex color for AQI value

def get_aqi_color(aqi):
    if aqi <= 50:
        return "#00e400"
    elif aqi <= 100:
        return "#ffff00"
    elif aqi <= 150:
        return "#ff7e00"
    elif aqi <= 200:
        return "#ff0000"
    elif aqi <= 300:
        return "#8f3f97"
    else:
        return "#7e0023"