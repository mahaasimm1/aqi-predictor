import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import requests
import shap
import gridfs
from pymongo import MongoClient
from datetime import datetime, timezone, timedelta
from config.settings import (
    MONGO_URI, MONGO_DB_NAME, MODELS_COLLECTION,
    FEATURES_COLLECTION, LAT, LON,
    OPENMETEO_AIR_QUALITY_URL, OPENMETEO_WEATHER_URL,
    LAG_HOURS, ROLLING_WINDOWS
)

# Total hours to forecast: 3 days = 72 steps of 1h each
FORECAST_STEPS = 72


def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_best_model(db=None):
    if db is None:
        db = get_db()

    fs = gridfs.GridFS(db)
    model_doc = db[MODELS_COLLECTION].find_one({"is_best": True})

    if not model_doc:
        raise ValueError("No best model found in MongoDB. Run training pipeline first.")

    model_bytes = fs.get(model_doc["gridfs_id"]).read()
    artifact = pickle.loads(model_bytes)
    return artifact, model_doc


# ---------------------------------------------------------------------------
# Feature loading from MongoDB
# ---------------------------------------------------------------------------

def load_latest_features(feature_cols, db=None, n=50):
    """Load the n most recent feature rows from MongoDB."""
    if db is None:
        db = get_db()

    records = list(db[FEATURES_COLLECTION].find(
        {}, sort=[("timestamp", -1)], limit=n
    ))

    if not records:
        raise ValueError("No features found in MongoDB. Run feature pipeline first.")

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols].values, df


# ---------------------------------------------------------------------------
# Fetch 3-day weather + air quality forecast from Open-Meteo
# ---------------------------------------------------------------------------

def fetch_weather_forecast():
    """
    Returns a DataFrame indexed by timestamp with hourly forecasted values
    for the next 3 days (72 rows minimum).

    Columns: timestamp, pm2_5, pm10, no2, o3, co, so2,
             temperature, humidity, pressure, wind_speed
    """
    aq_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide",
        "timezone": "Asia/Karachi",
        "forecast_days": 4,   # fetch 4 days so we always have 72+ hours
    }
    w_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Karachi",
        "forecast_days": 4,
    }

    try:
        aq_resp = requests.get(OPENMETEO_AIR_QUALITY_URL, params=aq_params, timeout=10)
        aq_resp.raise_for_status()
        aq = aq_resp.json()["hourly"]

        w_resp = requests.get(OPENMETEO_WEATHER_URL, params=w_params, timeout=10)
        w_resp.raise_for_status()
        w = w_resp.json()["hourly"]

        # Build a time-indexed dict for fast lookup
        w_map = {t: i for i, t in enumerate(w["time"])}

        rows = []
        for i, t in enumerate(aq["time"]):
            wi = w_map.get(t, 0)
            rows.append({
                "timestamp": t,   # string like "2025-06-01T14:00"
                "pm2_5":        float(aq["pm2_5"][i] or 0),
                "pm10":         float(aq["pm10"][i] or 0),
                "no2":          float(aq["nitrogen_dioxide"][i] or 0),
                "o3":           float(aq["ozone"][i] or 0),
                "co":           float(aq["carbon_monoxide"][i] or 0),
                "so2":          float(aq["sulphur_dioxide"][i] or 0),
                "temperature":  float(w["temperature_2m"][wi] or 0),
                "humidity":     float(w["relative_humidity_2m"][wi] or 0),
                "pressure":     float(w["surface_pressure"][wi] or 0),
                "wind_speed":   float(w["wind_speed_10m"][wi] or 0),
            })

        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        print(f"Warning: could not fetch weather forecast ({e}). "
              "Weather features will be held constant.")
        return None


def _lookup_weather(forecast_df, target_dt):
    """
    Given the forecast DataFrame, return the weather row closest to target_dt.
    target_dt must be timezone-aware or naive (both sides treated as UTC).
    """
    if forecast_df is None or forecast_df.empty:
        return None

    # Build the key string the same way Open-Meteo formats it
    key = target_dt.strftime("%Y-%m-%dT%H:00")
    row = forecast_df[forecast_df["timestamp"] == key]

    if row.empty:
        # Fallback: pick the nearest available hour
        try:
            times = pd.to_datetime(forecast_df["timestamp"])
            target_naive = pd.Timestamp(target_dt).tz_localize(None) \
                if target_dt.tzinfo else pd.Timestamp(target_dt)
            idx = (times - target_naive).abs().argmin()
            row = forecast_df.iloc[[idx]]
        except Exception:
            return None

    return row.iloc[0]


# ---------------------------------------------------------------------------
# Single-step prediction
# ---------------------------------------------------------------------------

def _predict_one(artifact, row_values):
    """
    Run one inference step.
    row_values: 1-D numpy array of shape (n_features,)
    Returns a scalar float AQI prediction.
    """
    model = artifact["model"]
    scaler = artifact["scaler"]

    X = row_values.reshape(1, -1)

    if scaler is not None:
        X = scaler.transform(X)

    model_type = type(model).__name__
    if model_type == "Sequential":
        X = X.reshape((1, 1, X.shape[1]))
        pred = float(model.predict(X, verbose=0).flatten()[0])
    else:
        pred = float(model.predict(X)[0])

    return max(0.0, min(500.0, pred))


# ---------------------------------------------------------------------------
# 72-step iterative forecast  (THE CORE FIX)
# ---------------------------------------------------------------------------

def generate_forecast(artifact, latest_features_df, feature_cols):
    """
    Produce 72 hourly AQI forecasts (3 days ahead) by calling the 1-hour
    model iteratively.

    At each step we:
      1. Update time features (hour, day_of_week, month).
      2. Update weather / pollutant features from Open-Meteo's 3-day forecast.
      3. Update AQI lag and rolling features using the growing prediction history.
      4. Run the model, collect the predicted AQI.
      5. Append that prediction to the history for the next step.
    """
    # --- Fetch the 3-day weather forecast once up front ---
    weather_df = fetch_weather_forecast()

    # --- Seed history with observed AQI values ---
    # We need up to 24 values for aqi_lag_24h / aqi_rolling_24h
    aqi_history = list(latest_features_df["aqi"].values[-24:])

    # --- Base row (last known observation) ---
    base_row = latest_features_df[feature_cols].values[-1].copy().astype(float)

    # --- Base timestamp ---
    try:
        base_time = pd.to_datetime(latest_features_df["timestamp"].iloc[-1])
        if base_time.tzinfo is None:
            base_time = base_time.tz_localize(timezone.utc)
    except Exception:
        base_time = datetime.now(timezone.utc)

    forecasts = []

    for step in range(1, FORECAST_STEPS + 1):
        row = base_row.copy()
        future_time = base_time + timedelta(hours=step)

        # --- 1. Time features ---
        _set(row, feature_cols, "hour",        future_time.hour)
        _set(row, feature_cols, "day_of_week", future_time.weekday())
        _set(row, feature_cols, "month",       future_time.month)

        # --- 2. Forecasted weather / pollutant features ---
        weather_row = _lookup_weather(weather_df, future_time)
        if weather_row is not None:
            for col in ["pm2_5", "pm10", "no2", "o3", "co", "so2",
                        "temperature", "humidity", "pressure", "wind_speed"]:
                if col in weather_row and col in feature_cols:
                    _set(row, feature_cols, col, float(weather_row[col]))

        # --- 3. AQI lag features (from growing history) ---
        for lag in LAG_HOURS:
            col = f"aqi_lag_{lag}h"
            if col in feature_cols and len(aqi_history) >= lag:
                _set(row, feature_cols, col, aqi_history[-lag])

        # --- 4. AQI rolling features ---
        for window in ROLLING_WINDOWS:
            col = f"aqi_rolling_{window}h"
            if col in feature_cols and len(aqi_history) >= 1:
                window_vals = aqi_history[-window:]   # as many as available
                _set(row, feature_cols, col, round(sum(window_vals) / len(window_vals), 2))

        # --- 5. AQI change rate ---
        if "aqi_change_rate" in feature_cols and len(aqi_history) >= 2:
            prev = aqi_history[-2]
            curr = aqi_history[-1]
            _set(row, feature_cols, "aqi_change_rate",
                 round((curr - prev) / max(abs(prev), 1), 4))

        # --- 6. Predict ---
        predicted_aqi = _predict_one(artifact, row)

        forecasts.append({
            "horizon_hours":  step,
            "predicted_aqi":  round(predicted_aqi, 2),
            "forecast_time":  future_time.isoformat(),
        })

        # Append prediction to history for next step's lag features
        aqi_history.append(predicted_aqi)
        if len(aqi_history) > 24:
            aqi_history = aqi_history[-24:]

    return forecasts


def _set(row, feature_cols, col, value):
    """Helper: set a value in the row array by column name."""
    if col in feature_cols:
        row[feature_cols.index(col)] = value


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

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

    shap_values = None

    try:
        if model_type in ["RandomForestRegressor", "XGBRegressor", "GradientBoostingRegressor"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
        elif model_type == "Ridge":
            explainer = shap.LinearExplainer(model, bg_input)
            shap_values = explainer.shap_values(X_input)
        # LSTM: skip (KernelExplainer is too slow for production)

        if shap_values is None:
            return None, None

        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "shap_value": np.abs(shap_values[-1] if len(shap_values.shape) > 1
                                 else shap_values[0])
        }).sort_values("shap_value", ascending=False)

        return shap_values, shap_df

    except Exception as e:
        print(f"SHAP generation failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# AQI helpers
# ---------------------------------------------------------------------------

def get_aqi_category(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


def get_aqi_color(aqi):
    if aqi <= 50:   return "#00e400"
    if aqi <= 100:  return "#ffff00"
    if aqi <= 150:  return "#ff7e00"
    if aqi <= 200:  return "#ff0000"
    if aqi <= 300:  return "#8f3f97"
    return "#7e0023"