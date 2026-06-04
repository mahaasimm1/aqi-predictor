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

FORECAST_STEPS = 72  # 3 days × 24 hours


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
        raise ValueError("No best model found. Run training pipeline first.")
    artifact = pickle.loads(fs.get(model_doc["gridfs_id"]).read())
    return artifact, model_doc


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_latest_features(feature_cols, db=None, n=50):
    if db is None:
        db = get_db()
    records = list(db[FEATURES_COLLECTION].find(
        {}, sort=[("timestamp", -1)], limit=n
    ))
    if not records:
        raise ValueError("No features in MongoDB. Run feature pipeline first.")
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Add derived features so the row matches training schema
    df = _add_inference_features(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_cols].values, df


def _add_inference_features(df):
    """
    Mirror of add_features() in training_pipeline — must stay in sync.
    Any feature added to training must also be computed here.
    """
    df = df.copy()

    # Log transforms
    for col in ["pm2_5", "pm10", "no2", "co", "so2"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # Wind interactions
    df["pm2_5_x_wind"] = df["pm2_5"] / (df["wind_speed"] + 1)
    df["pm10_x_wind"]  = df["pm10"]  / (df["wind_speed"] + 1)

    # Momentum
    if "aqi_lag_3h" in df.columns:
        df["aqi_momentum_3h"] = (df["aqi"] - df["aqi_lag_3h"]) / 3.0
    else:
        df["aqi_momentum_3h"] = 0.0
    if "aqi_lag_6h" in df.columns:
        df["aqi_momentum_6h"] = (df["aqi"] - df["aqi_lag_6h"]) / 6.0
    else:
        df["aqi_momentum_6h"] = 0.0

    # lag_48h
    df["aqi_lag_48h"] = df["aqi"].shift(48) if len(df) > 48 else df["aqi"]

    # Peak traffic hour flag
    df["peak_traffic_hour"] = df["hour"].between(15, 20).astype(int)

    # Rolling volatility
    df["aqi_std_6h"] = (
        df["aqi"].shift(1).rolling(window=6, min_periods=2).std().round(3)
    )
    df["aqi_std_6h"] = df["aqi_std_6h"].fillna(0.0)

    # Cyclical time
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ---------------------------------------------------------------------------
# 3-day weather forecast from Open-Meteo
# ---------------------------------------------------------------------------

def fetch_weather_forecast():
    aq_params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide",
        "timezone": "Asia/Karachi", "forecast_days": 4,
    }
    w_params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "timezone": "Asia/Karachi", "forecast_days": 4,
    }
    try:
        aq = requests.get(OPENMETEO_AIR_QUALITY_URL, params=aq_params, timeout=10).json()["hourly"]
        w  = requests.get(OPENMETEO_WEATHER_URL,     params=w_params,  timeout=10).json()["hourly"]
        w_map = {t: i for i, t in enumerate(w["time"])}
        rows = []
        for i, t in enumerate(aq["time"]):
            wi = w_map.get(t, 0)
            rows.append({
                "timestamp":   t,
                "pm2_5":       float(aq["pm2_5"][i] or 0),
                "pm10":        float(aq["pm10"][i] or 0),
                "no2":         float(aq["nitrogen_dioxide"][i] or 0),
                "o3":          float(aq["ozone"][i] or 0),
                "co":          float(aq["carbon_monoxide"][i] or 0),
                "so2":         float(aq["sulphur_dioxide"][i] or 0),
                "temperature": float(w["temperature_2m"][wi] or 0),
                "humidity":    float(w["relative_humidity_2m"][wi] or 0),
                "pressure":    float(w["surface_pressure"][wi] or 0),
                "wind_speed":  float(w["wind_speed_10m"][wi] or 0),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Warning: weather forecast fetch failed ({e}). "
              "Weather features held constant.")
        return None


def _lookup_weather(forecast_df, target_dt):
    if forecast_df is None or forecast_df.empty:
        return None
    key = target_dt.strftime("%Y-%m-%dT%H:00")
    row = forecast_df[forecast_df["timestamp"] == key]
    if row.empty:
        try:
            times = pd.to_datetime(forecast_df["timestamp"])
            naive = pd.Timestamp(target_dt).tz_localize(None) \
                    if target_dt.tzinfo else pd.Timestamp(target_dt)
            row = forecast_df.iloc[[(times - naive).abs().argmin()]]
        except Exception:
            return None
    return row.iloc[0]


# ---------------------------------------------------------------------------
# Single prediction step
# ---------------------------------------------------------------------------

def _predict_one(artifact, row_values):
    model  = artifact["model"]
    scaler = artifact["scaler"]
    X = row_values.reshape(1, -1)
    if scaler is not None:
        X = scaler.transform(X)
    if type(model).__name__ == "Sequential":
        X = X.reshape((1, 1, X.shape[1]))
        pred = float(model.predict(X, verbose=0).flatten()[0])
    else:
        pred = float(model.predict(X)[0])
    return max(0.0, min(400.0, round(pred, 2)))


# ---------------------------------------------------------------------------
# 72-step iterative forecast
# ---------------------------------------------------------------------------

def _set(row, feature_cols, col, value):
    if col in feature_cols:
        row[feature_cols.index(col)] = float(value)


def generate_forecast(artifact, latest_features_df, feature_cols):
    weather_df   = fetch_weather_forecast()
    aqi_history  = list(latest_features_df["aqi"].values[-24:])
    base_row     = latest_features_df[feature_cols].values[-1].copy().astype(float)

    try:
        base_time = pd.to_datetime(latest_features_df["timestamp"].iloc[-1])
        if base_time.tzinfo is None:
            base_time = base_time.tz_localize(timezone.utc)
    except Exception:
        base_time = datetime.now(timezone.utc)

    forecasts = []

    for step in range(1, FORECAST_STEPS + 1):
        row         = base_row.copy()
        future_time = base_time + timedelta(hours=step)

        # --- 1. Forecasted weather / pollutants ---
        wr = _lookup_weather(weather_df, future_time)
        if wr is not None:
            raw = {}
            for col in ["pm2_5", "pm10", "no2", "o3", "co", "so2",
                        "temperature", "humidity", "pressure", "wind_speed"]:
                if col in wr:
                    raw[col] = float(wr[col])
                    _set(row, feature_cols, col, raw[col])

            # Derived features from forecasted weather
            pm25 = raw.get("pm2_5", 0)
            pm10 = raw.get("pm10",  0)
            hum  = raw.get("humidity",   0)
            wind = raw.get("wind_speed", 0)
            _set(row, feature_cols, "pm2_5_x_humidity", pm25 * hum / 100)
            _set(row, feature_cols, "pm10_x_humidity",  pm10 * hum / 100)
            _set(row, feature_cols, "pm2_5_x_wind",     pm25 / (wind + 1))
            _set(row, feature_cols, "pm10_x_wind",      pm10 / (wind + 1))
            _set(row, feature_cols, "log_pm2_5", np.log1p(pm25))
            _set(row, feature_cols, "log_pm10",  np.log1p(pm10))
            _set(row, feature_cols, "log_co",    np.log1p(raw.get("co",  0)))
            _set(row, feature_cols, "log_so2",   np.log1p(raw.get("so2", 0)))

        # --- 2. Cyclical time features ---
        _set(row, feature_cols, "hour_sin",  np.sin(2 * np.pi * future_time.hour     / 24))
        _set(row, feature_cols, "hour_cos",  np.cos(2 * np.pi * future_time.hour     / 24))
        _set(row, feature_cols, "month_sin", np.sin(2 * np.pi * future_time.month    / 12))
        _set(row, feature_cols, "month_cos", np.cos(2 * np.pi * future_time.month    / 12))
        _set(row, feature_cols, "day_of_week", future_time.weekday())

        # --- 3. AQI lag features ---
        for lag in LAG_HOURS:
            if len(aqi_history) >= lag:
                _set(row, feature_cols, f"aqi_lag_{lag}h", aqi_history[-lag])

        # --- 4. AQI rolling features ---
        for window in ROLLING_WINDOWS:
            vals = aqi_history[-window:]
            if vals:
                _set(row, feature_cols, f"aqi_rolling_{window}h",
                     round(sum(vals) / len(vals), 2))

        # --- 5. AQI momentum, volatility, peak hour, lag_48h ---
        if len(aqi_history) >= 2:
            prev = aqi_history[-2]
            curr = aqi_history[-1]
            _set(row, feature_cols, "aqi_change_rate",
                 round((curr - prev) / max(abs(prev), 1), 4))
        if len(aqi_history) >= 3:
            _set(row, feature_cols, "aqi_momentum_3h",
                 round((aqi_history[-1] - aqi_history[-3]) / 3.0, 4))
        if len(aqi_history) >= 6:
            _set(row, feature_cols, "aqi_momentum_6h",
                 round((aqi_history[-1] - aqi_history[-6]) / 6.0, 4))
            _set(row, feature_cols, "aqi_std_6h",
                 round(float(np.std(aqi_history[-6:])), 3))
        if len(aqi_history) >= 48:
            _set(row, feature_cols, "aqi_lag_48h", aqi_history[-48])
        _set(row, feature_cols, "peak_traffic_hour",
             1 if 15 <= future_time.hour <= 20 else 0)

        # --- 6. Predict ---
        predicted_aqi = _predict_one(artifact, row)
        forecasts.append({
            "horizon_hours": step,
            "predicted_aqi": predicted_aqi,
            "forecast_time": future_time.isoformat(),
        })

        aqi_history.append(predicted_aqi)
        if len(aqi_history) > 24:
            aqi_history = aqi_history[-24:]

    return forecasts


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def generate_shap_values(artifact, X, feature_cols, background_X=None):
    model      = artifact["model"]
    scaler     = artifact["scaler"]
    model_type = type(model).__name__

    if scaler is not None:
        X_input = scaler.transform(X)
        bg      = scaler.transform(background_X) if background_X is not None else X_input
    else:
        X_input = X
        bg      = background_X if background_X is not None else X_input

    shap_values = None
    try:
        if model_type in ["RandomForestRegressor", "XGBRegressor",
                          "GradientBoostingRegressor"]:
            shap_values = shap.TreeExplainer(model).shap_values(X_input)
        elif model_type == "Ridge":
            shap_values = shap.LinearExplainer(model, bg).shap_values(X_input)
        elif model_type == "StackingRegressor":
            # Use the best base learner for SHAP (first tree-based one found)
            for _, est in model.estimators_:
                etype = type(est).__name__
                if etype in ["RandomForestRegressor", "XGBRegressor",
                             "GradientBoostingRegressor"]:
                    shap_values = shap.TreeExplainer(est).shap_values(X_input)
                    break

        if shap_values is None:
            return None, None

        vals = shap_values[-1] if shap_values.ndim > 1 else shap_values[0]
        shap_df = pd.DataFrame({
            "feature":    feature_cols,
            "shap_value": np.abs(vals),
        }).sort_values("shap_value", ascending=False)

        return shap_values, shap_df

    except Exception as e:
        print(f"SHAP failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# AQI helpers
# ---------------------------------------------------------------------------

def get_aqi_category(aqi):
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"


def get_aqi_color(aqi):
    if aqi <= 50:  return "#00e400"
    if aqi <= 100: return "#ffff00"
    if aqi <= 150: return "#ff7e00"
    if aqi <= 200: return "#ff0000"
    if aqi <= 300: return "#8f3f97"
    return "#7e0023"