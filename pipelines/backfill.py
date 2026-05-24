import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from config.settings import (
    OPENMETEO_AIR_QUALITY_URL, OPENMETEO_WEATHER_URL,
    AIR_QUALITY_PARAMS, WEATHER_PARAMS,
    LAT, LON, CITY, MONGO_URI, MONGO_DB_NAME,
    FEATURES_COLLECTION, LAG_HOURS, ROLLING_WINDOWS, TRAINING_DAYS
)


# MongoDB connection

def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


# Fetch historical air quality data

def fetch_historical_air_quality(start_date, end_date):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": AIR_QUALITY_PARAMS,
        "timezone": "Asia/Karachi",
        "start_date": start_date,
        "end_date": end_date
    }
    response = requests.get(OPENMETEO_AIR_QUALITY_URL, params=params)
    response.raise_for_status()
    return response.json()


# Fetch historical weather data

def fetch_historical_weather(start_date, end_date):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": WEATHER_PARAMS,
        "timezone": "Asia/Karachi",
        "start_date": start_date,
        "end_date": end_date
    }
    response = requests.get(OPENMETEO_WEATHER_URL, params=params)
    response.raise_for_status()
    return response.json()


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


# Parse API responses into records

def parse_records(aq_data, w_data):
    aq_hourly = aq_data["hourly"]
    w_hourly = w_data["hourly"]

    w_time_map = {t: i for i, t in enumerate(w_hourly["time"])}

    records = []
    for i, time_str in enumerate(aq_hourly["time"]):
        aqi = aq_hourly["us_aqi"][i]
        if aqi is None:
            continue

        aqi = float(aqi)
        if aqi > 400:
            continue

        w_idx = w_time_map.get(time_str, 0)

        dt = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)

        record = {
            "timestamp": dt,
            "city": CITY,
            "lat": LAT,
            "lon": LON,
            "aqi": aqi,
            "pm2_5": aq_hourly["pm2_5"][i] or 0,
            "pm10": aq_hourly["pm10"][i] or 0,
            "no2": aq_hourly["nitrogen_dioxide"][i] or 0,
            "o3": aq_hourly["ozone"][i] or 0,
            "co": aq_hourly["carbon_monoxide"][i] or 0,
            "so2": aq_hourly["sulphur_dioxide"][i] or 0,
            "temperature": w_hourly["temperature_2m"][w_idx] or 0,
            "humidity": w_hourly["relative_humidity_2m"][w_idx] or 0,
            "pressure": w_hourly["surface_pressure"][w_idx] or 0,
            "wind_speed": w_hourly["wind_speed_10m"][w_idx] or 0,
            "wind_deg": w_hourly["wind_direction_10m"][w_idx] or 0,
            "visibility": w_hourly["visibility"][w_idx] or 0,
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
            "day_of_month": dt.day,
            "month": dt.month,
            "aqi_category": get_aqi_category(aqi),
        }
        records.append(record)

    return records


# Compute lag and rolling features

def compute_lag_and_rolling(records):
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    aqi_series = df["aqi"]

    for lag in LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = aqi_series.shift(lag).bfill()

    for window in ROLLING_WINDOWS:
        df[f"aqi_rolling_{window}h"] = (
            aqi_series.shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .round(2)
        )

    df["aqi_change_rate"] = (
        (aqi_series - aqi_series.shift(1)) / aqi_series.shift(1).replace(0, 1)
    ).fillna(0).round(4)

    return df.to_dict(orient="records")


# Store records in MongoDB

def store_records(records, db):
    collection = db[FEATURES_COLLECTION]
    inserted = 0
    skipped = 0

    for record in records:
        result = collection.update_one(
            {"timestamp": record["timestamp"], "city": record["city"]},
            {"$set": record},
            upsert=True
        )
        if result.upserted_id:
            inserted += 1
        else:
            skipped += 1

    print(f"Inserted: {inserted} | Skipped: {skipped}")


# Main backfill runner

def run_backfill(days=TRAINING_DAYS):
    print(f"Starting backfill for {CITY} — last {days} days...")
    db = get_db()

    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"Fetching air quality data from {start_date} to {end_date}...")
    aq_data = fetch_historical_air_quality(start_date, end_date)

    print("Fetching weather data...")
    w_data = fetch_historical_weather(start_date, end_date)

    print("Parsing records...")
    records = parse_records(aq_data, w_data)
    print(f"Parsed {len(records)} records")

    print("Computing lag and rolling features...")
    records = compute_lag_and_rolling(records)

    print("Storing in MongoDB...")
    store_records(records, db)

    print(f"Backfill complete. Total records: {len(records)}")


if __name__ == "__main__":
    run_backfill()