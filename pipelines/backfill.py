import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from config.settings import (
    OPENWEATHER_API_KEY, AIR_POLLUTION_HISTORY_URL, WEATHER_URL,
    LAT, LON, CITY, MONGO_URI, MONGO_DB_NAME,
    FEATURES_COLLECTION, LAG_HOURS, ROLLING_WINDOWS, TRAINING_DAYS
)


# MongoDB Connection

def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


# Fetch historical air pollution data for a time range

def fetch_historical_air_pollution(start_dt, end_dt):
    start_unix = int(start_dt.timestamp())
    end_unix = int(end_dt.timestamp())

    params = {
        "lat": LAT,
        "lon": LON,
        "start": start_unix,
        "end": end_unix,
        "appid": OPENWEATHER_API_KEY
    }
    response = requests.get(AIR_POLLUTION_HISTORY_URL, params=params)
    response.raise_for_status()
    return response.json()["list"]


# Fetch weather data for a specific timestamp
# OpenWeather free tier does not support historical weather
# so we use current weather as a proxy for all historical records

def fetch_current_weather():
    params = {
        "lat": LAT,
        "lon": LON,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(WEATHER_URL, params=params)
    response.raise_for_status()
    return response.json()


# Convert OpenWeather 1-5 AQI to approximate US AQI using PM2.5 breakpoints

def convert_to_us_aqi(pm2_5):
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
    return 500


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


# Parse a single historical pollution entry into a flat record

def parse_historical_record(entry, weather_data):
    components = entry["components"]
    dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
    pm2_5 = components.get("pm2_5", 0)
    aqi = convert_to_us_aqi(pm2_5)

    return {
        "timestamp": dt,
        "city": CITY,
        "lat": LAT,
        "lon": LON,
        "aqi_openweather": entry["main"]["aqi"],
        "aqi": aqi,
        "pm2_5": pm2_5,
        "pm10": components.get("pm10", 0),
        "no2": components.get("no2", 0),
        "o3": components.get("o3", 0),
        "co": components.get("co", 0),
        "so2": components.get("so2", 0),
        "no": components.get("no", 0),
        "nh3": components.get("nh3", 0),
        "temperature": weather_data["main"]["temp"],
        "humidity": weather_data["main"]["humidity"],
        "pressure": weather_data["main"]["pressure"],
        "wind_speed": weather_data["wind"]["speed"],
        "wind_deg": weather_data["wind"].get("deg", 0),
        "visibility": weather_data.get("visibility", 0),
        "weather_desc": weather_data["weather"][0]["description"],
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "day_of_month": dt.day,
        "month": dt.month,
        "is_weekend": int(dt.weekday() >= 5),
        "is_peak_hour": int(dt.hour in [7, 8, 9, 17, 18, 19]),
        "aqi_category": get_aqi_category(aqi),
    }


# Compute lag and rolling features across the full sorted list of records

def compute_lag_and_rolling(records):
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    aqi_series = df["aqi"]

    for lag in LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = aqi_series.shift(lag).fillna(method="bfill")

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


# Store all records in MongoDB, skip duplicates

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

    print(f"Inserted: {inserted} | Skipped (already exist): {skipped}")


# Main backfill runner

def run_backfill(days=TRAINING_DAYS):
    print(f"Starting backfill for {CITY} — last {days} days...")

    db = get_db()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    print("Fetching current weather (used as proxy for historical weather)...")
    weather_data = fetch_current_weather()

    print(f"Fetching historical pollution data from {start_dt.date()} to {end_dt.date()}...")
    pollution_list = fetch_historical_air_pollution(start_dt, end_dt)
    print(f"Fetched {len(pollution_list)} raw records")

    print("Parsing records...")
    records = [parse_historical_record(entry, weather_data) for entry in pollution_list]

    print("Computing lag and rolling features...")
    records = compute_lag_and_rolling(records)

    print("Storing in MongoDB...")
    store_records(records, db)

    print(f"Backfill complete. Total records processed: {len(records)}")


if __name__ == "__main__":
    run_backfill()