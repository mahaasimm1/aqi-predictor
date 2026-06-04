import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient
from config.settings import (
    OPENMETEO_AIR_QUALITY_URL, OPENMETEO_WEATHER_URL,
    AIR_QUALITY_PARAMS, WEATHER_PARAMS,
    LAT, LON, CITY, MONGO_URI, MONGO_DB_NAME,
    FEATURES_COLLECTION, LAG_HOURS, ROLLING_WINDOWS
)


def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


def safe_val(value, fallback=0.0):
    """Explicit None check — same fix as backfill.py."""
    return value if value is not None else fallback


def fetch_current_data():
    aq_params = {
        "latitude": LAT, "longitude": LON,
        "hourly": AIR_QUALITY_PARAMS,
        "timezone": "Asia/Karachi",
        "forecast_days": 1
    }
    w_params = {
        "latitude": LAT, "longitude": LON,
        "hourly": WEATHER_PARAMS,
        "timezone": "Asia/Karachi",
        "forecast_days": 1
    }
    aq_response = requests.get(OPENMETEO_AIR_QUALITY_URL, params=aq_params)
    aq_response.raise_for_status()

    w_response = requests.get(OPENMETEO_WEATHER_URL, params=w_params)
    w_response.raise_for_status()

    return aq_response.json(), w_response.json()


def parse_current_hour(aq_data, w_data):
    now          = datetime.now(timezone.utc)
    current_hour = now.strftime("%Y-%m-%dT%H:00")

    aq_hourly = aq_data["hourly"]
    w_hourly  = w_data["hourly"]

    try:
        idx = aq_hourly["time"].index(current_hour)
    except ValueError:
        idx = 0

    try:
        w_idx = w_hourly["time"].index(current_hour)
    except ValueError:
        w_idx = 0

    aqi = aq_hourly["us_aqi"][idx]
    if aqi is None:
        aqi = 0.0

    record = {
        "timestamp":   datetime.fromisoformat(aq_hourly["time"][idx]).replace(tzinfo=timezone.utc),
        "city":        CITY,
        "lat":         LAT,
        "lon":         LON,
        "aqi":         float(aqi),
        "pm2_5":       safe_val(aq_hourly["pm2_5"][idx]),
        "pm10":        safe_val(aq_hourly["pm10"][idx]),
        "no2":         safe_val(aq_hourly["nitrogen_dioxide"][idx]),
        "o3":          safe_val(aq_hourly["ozone"][idx]),
        "co":          safe_val(aq_hourly["carbon_monoxide"][idx]),
        "so2":         safe_val(aq_hourly["sulphur_dioxide"][idx]),
        # THE FIX: safe_val instead of 'or 0'
        "temperature": safe_val(w_hourly["temperature_2m"][w_idx]),
        "humidity":    safe_val(w_hourly["relative_humidity_2m"][w_idx]),
        "pressure":    safe_val(w_hourly["surface_pressure"][w_idx]),
        "wind_speed":  safe_val(w_hourly["wind_speed_10m"][w_idx]),
        "wind_deg":    safe_val(w_hourly["wind_direction_10m"][w_idx]),
        "visibility":  safe_val(w_hourly["visibility"][w_idx]),
    }

    return record


def get_aqi_category(aqi):
    if aqi <= 50:   return "good"
    if aqi <= 100:  return "moderate"
    if aqi <= 150:  return "unhealthy_sensitive"
    if aqi <= 200:  return "unhealthy"
    if aqi <= 300:  return "very_unhealthy"
    return "hazardous"


def compute_features(record, db):
    ts = record["timestamp"]
    record["hour"]         = ts.hour
    record["day_of_week"]  = ts.weekday()
    record["day_of_month"] = ts.day
    record["month"]        = ts.month

    collection = db[FEATURES_COLLECTION]
    recent = list(collection.find(
        {"city": CITY},
        sort=[("timestamp", -1)],
        limit=max(LAG_HOURS + ROLLING_WINDOWS)
    ))

    aqi_values = [r["aqi"] for r in recent if "aqi" in r]

    for lag in LAG_HOURS:
        record[f"aqi_lag_{lag}h"] = aqi_values[lag - 1] if len(aqi_values) >= lag else record["aqi"]

    for window in ROLLING_WINDOWS:
        if len(aqi_values) >= window:
            record[f"aqi_rolling_{window}h"] = round(sum(aqi_values[:window]) / window, 2)
        else:
            record[f"aqi_rolling_{window}h"] = record["aqi"]

    if len(aqi_values) >= 1:
        prev = aqi_values[0]
        record["aqi_change_rate"] = round((record["aqi"] - prev) / max(prev, 1), 4)
    else:
        record["aqi_change_rate"] = 0.0

    record["aqi_category"] = get_aqi_category(record["aqi"])
    return record


def store_features(record, db):
    db[FEATURES_COLLECTION].update_one(
        {"timestamp": record["timestamp"], "city": record["city"]},
        {"$set": record},
        upsert=True
    )
    temp = record.get("temperature", "?")
    hum  = record.get("humidity", "?")
    print(f"Stored: {record['city']} @ {record['timestamp']} | "
          f"AQI={record['aqi']} | temp={temp}°C | humidity={hum}%")


def run_feature_pipeline():
    print(f"Running feature pipeline for {CITY}...")
    db = get_db()

    print("Fetching data from Open-Meteo...")
    aq_data, w_data = fetch_current_data()

    record = parse_current_hour(aq_data, w_data)
    record = compute_features(record, db)
    store_features(record, db)

    print("Feature pipeline complete!")
    return record


if __name__ == "__main__":
    run_feature_pipeline()