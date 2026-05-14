import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient
from config.settings import (
    OPENWEATHER_API_KEY, AIR_POLLUTION_URL, WEATHER_URL,
    LAT, LON, CITY, MONGO_URI, MONGO_DB_NAME,
    FEATURES_COLLECTION, LAG_HOURS, ROLLING_WINDOWS
)


# MongoDB Connection

def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


# ─── Fetch Raw Data ───────────────────────────────────────────────────────────

def fetch_air_pollution():
    """Fetch current air pollution data from OpenWeather."""
    params = {
        "lat": LAT,
        "lon": LON,
        "appid": OPENWEATHER_API_KEY
    }
    response = requests.get(AIR_POLLUTION_URL, params=params)
    response.raise_for_status()
    data = response.json()
    return data["list"][0]


def fetch_weather():
    """Fetch current weather data from OpenWeather."""
    params = {
        "lat": LAT,
        "lon": LON,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(WEATHER_URL, params=params)
    response.raise_for_status()
    return response.json()


# ─── Parse Raw Data ───────────────────────────────────────────────────────────

def parse_raw_data(pollution_data, weather_data):
    """Parse raw API responses into a flat dictionary."""
    components = pollution_data["components"]
    dt = datetime.fromtimestamp(pollution_data["dt"], tz=timezone.utc)

    record = {
        "timestamp": dt,
        "city": CITY,
        "lat": LAT,
        "lon": LON,

        # AQI (OpenWeather uses 1-5 scale, we convert to US AQI equivalent)
        "aqi_openweather": pollution_data["main"]["aqi"],

        # Pollutants (μg/m³)
        "pm2_5": components.get("pm2_5", 0),
        "pm10": components.get("pm10", 0),
        "no2": components.get("no2", 0),
        "o3": components.get("o3", 0),
        "co": components.get("co", 0),
        "so2": components.get("so2", 0),
        "no": components.get("no", 0),
        "nh3": components.get("nh3", 0),

        # Weather
        "temperature": weather_data["main"]["temp"],
        "humidity": weather_data["main"]["humidity"],
        "pressure": weather_data["main"]["pressure"],
        "wind_speed": weather_data["wind"]["speed"],
        "wind_deg": weather_data["wind"].get("deg", 0),
        "visibility": weather_data.get("visibility", 0),
        "weather_desc": weather_data["weather"][0]["description"],
    }

    # Convert OpenWeather 1-5 AQI to approximate US AQI
    record["aqi"] = convert_to_us_aqi(record["aqi_openweather"], record["pm2_5"])
    # Cap AQI at 400 and fix data errors where AQI is impossibly high vs PM2.5
    if record["aqi"] == 500 and record["pm2_5"] < 50:
        record["aqi"] = convert_to_us_aqi(record["aqi_openweather"], record["pm2_5"])
        record["aqi"] = min(record["aqi"], int(record["pm2_5"] * 2.5))

    return record


def convert_to_us_aqi(ow_aqi, pm2_5):
    """
    Convert OpenWeather AQI (1-5) to approximate US AQI using PM2.5.
    Uses EPA breakpoints for PM2.5.
    """
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm2_5 <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm2_5 - bp_lo) + aqi_lo
            return round(aqi)
    return 500  # hazardous cap


# ─── Feature Engineering ─────────────────────────────────────────────────────

def compute_features(record, db):
    """Add time-based, lag, and rolling features to the record."""
    ts = record["timestamp"]

    # Time-based features
    record["hour"] = ts.hour
    record["day_of_week"] = ts.weekday()
    record["day_of_month"] = ts.day
    record["month"] = ts.month

    # Fetch recent records for lag/rolling features
    collection = db[FEATURES_COLLECTION]
    recent = list(collection.find(
        {"city": CITY},
        sort=[("timestamp", -1)],
        limit=max(LAG_HOURS + ROLLING_WINDOWS)
    ))

    aqi_values = [r["aqi"] for r in recent if "aqi" in r]

    # Lag features
    for lag in LAG_HOURS:
        if len(aqi_values) >= lag:
            record[f"aqi_lag_{lag}h"] = aqi_values[lag - 1]
        else:
            record[f"aqi_lag_{lag}h"] = record["aqi"]  # fallback to current

    # Rolling average features
    for window in ROLLING_WINDOWS:
        if len(aqi_values) >= window:
            record[f"aqi_rolling_{window}h"] = round(
                sum(aqi_values[:window]) / window, 2
            )
        else:
            record[f"aqi_rolling_{window}h"] = record["aqi"]

    # AQI change rate (vs 1 hour ago)
    if len(aqi_values) >= 1:
        prev_aqi = aqi_values[0]
        record["aqi_change_rate"] = round(
            (record["aqi"] - prev_aqi) / max(prev_aqi, 1), 4
        )
    else:
        record["aqi_change_rate"] = 0.0

    # AQI category
    record["aqi_category"] = get_aqi_category(record["aqi"])

    return record


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


# ─── Store Features ───────────────────────────────────────────────────────────

def store_features(record, db):
    """Store feature record in MongoDB, avoid duplicates by timestamp."""
    collection = db[FEATURES_COLLECTION]

    # Use timestamp + city as unique key
    collection.update_one(
        {
            "timestamp": record["timestamp"],
            "city": record["city"]
        },
        {"$set": record},
        upsert=True
    )
    print(f"✅ Stored features for {record['city']} at {record['timestamp']} | AQI: {record['aqi']}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_feature_pipeline():
    print(f"🚀 Running feature pipeline for {CITY}...")

    # Connect to MongoDB
    db = get_db()

    # Fetch raw data
    print("📡 Fetching air pollution data...")
    pollution_data = fetch_air_pollution()

    print("🌤  Fetching weather data...")
    weather_data = fetch_weather()

    # Parse
    record = parse_raw_data(pollution_data, weather_data)

    # Engineer features
    record = compute_features(record, db)

    # Store
    store_features(record, db)

    print("✅ Feature pipeline complete!")
    return record


if __name__ == "__main__":
    run_feature_pipeline()