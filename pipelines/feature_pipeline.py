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


# MongoDB connection

def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


# Fetch current air quality and weather from Open-Meteo

def fetch_current_data():
    # Air quality
    aq_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": AIR_QUALITY_PARAMS,
        "timezone": "Asia/Karachi",
        "forecast_days": 1
    }
    aq_response = requests.get(OPENMETEO_AIR_QUALITY_URL, params=aq_params)
    aq_response.raise_for_status()
    aq_data = aq_response.json()

    # Weather
    w_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": WEATHER_PARAMS,
        "timezone": "Asia/Karachi",
        "forecast_days": 1
    }
    w_response = requests.get(OPENMETEO_WEATHER_URL, params=w_params)
    w_response.raise_for_status()
    w_data = w_response.json()

    return aq_data, w_data


# Parse current hour data from API response

def parse_current_hour(aq_data, w_data):
    now = datetime.now(timezone.utc)
    current_hour = now.strftime("%Y-%m-%dT%H:00")

    aq_hourly = aq_data["hourly"]
    w_hourly = w_data["hourly"]

    # Find index for current hour
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
        aqi = 0

    record = {
        "timestamp": datetime.fromisoformat(aq_hourly["time"][idx]).replace(tzinfo=timezone.utc),
        "city": CITY,
        "lat": LAT,
        "lon": LON,
        "aqi": float(aqi),
        "pm2_5": aq_hourly["pm2_5"][idx] or 0,
        "pm10": aq_hourly["pm10"][idx] or 0,
        "no2": aq_hourly["nitrogen_dioxide"][idx] or 0,
        "o3": aq_hourly["ozone"][idx] or 0,
        "co": aq_hourly["carbon_monoxide"][idx] or 0,
        "so2": aq_hourly["sulphur_dioxide"][idx] or 0,
        "temperature": w_hourly["temperature_2m"][w_idx] or 0,
        "humidity": w_hourly["relative_humidity_2m"][w_idx] or 0,
        "pressure": w_hourly["surface_pressure"][w_idx] or 0,
        "wind_speed": w_hourly["wind_speed_10m"][w_idx] or 0,
        "wind_deg": w_hourly["wind_direction_10m"][w_idx] or 0,
        "visibility": w_hourly["visibility"][w_idx] or 0,
    }

    return record


# Get AQI category

def get_aqi_category(aqi):
    if aqi <= 50:
        return "good"
    elif aqi <= 100:
        return "moderate"
    elif aqi <= 150:
        return "unhealthy_sensitive"
    elif aqi <= 200:
        return "unhealthy"
    elif aqi <= 300:
        return "very_unhealthy"
    else:
        return "hazardous"


# Compute time-based, lag, and rolling features

def compute_features(record, db):
    ts = record["timestamp"]

    record["hour"] = ts.hour
    record["day_of_week"] = ts.weekday()
    record["day_of_month"] = ts.day
    record["month"] = ts.month

    collection = db[FEATURES_COLLECTION]
    recent = list(collection.find(
        {"city": CITY},
        sort=[("timestamp", -1)],
        limit=max(LAG_HOURS + ROLLING_WINDOWS)
    ))

    aqi_values = [r["aqi"] for r in recent if "aqi" in r]

    for lag in LAG_HOURS:
        if len(aqi_values) >= lag:
            record[f"aqi_lag_{lag}h"] = aqi_values[lag - 1]
        else:
            record[f"aqi_lag_{lag}h"] = record["aqi"]

    for window in ROLLING_WINDOWS:
        if len(aqi_values) >= window:
            record[f"aqi_rolling_{window}h"] = round(sum(aqi_values[:window]) / window, 2)
        else:
            record[f"aqi_rolling_{window}h"] = record["aqi"]

    if len(aqi_values) >= 1:
        prev_aqi = aqi_values[0]
        record["aqi_change_rate"] = round((record["aqi"] - prev_aqi) / max(prev_aqi, 1), 4)
    else:
        record["aqi_change_rate"] = 0.0

    record["aqi_category"] = get_aqi_category(record["aqi"])

    return record


# Store features in MongoDB

def store_features(record, db):
    collection = db[FEATURES_COLLECTION]
    collection.update_one(
        {"timestamp": record["timestamp"], "city": record["city"]},
        {"$set": record},
        upsert=True
    )
    print(f"Stored features for {record['city']} at {record['timestamp']} | AQI: {record['aqi']}")


# Main

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