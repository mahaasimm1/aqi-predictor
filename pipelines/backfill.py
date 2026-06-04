import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from config.settings import (
    OPENMETEO_AIR_QUALITY_URL,
    AIR_QUALITY_PARAMS, WEATHER_PARAMS,
    LAT, LON, CITY, MONGO_URI, MONGO_DB_NAME,
    FEATURES_COLLECTION, LAG_HOURS, ROLLING_WINDOWS, TRAINING_DAYS
)

# Historical weather must use the archive endpoint — the forecast endpoint
# only has data for recent/future hours and returns None for older dates.
OPENMETEO_WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


def fetch_historical_air_quality(start_date, end_date):
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": AIR_QUALITY_PARAMS,
        "timezone": "Asia/Karachi",
        "start_date": start_date, "end_date": end_date
    }
    r = requests.get(OPENMETEO_AIR_QUALITY_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_historical_weather(start_date, end_date):
    """
    Use the archive endpoint for historical weather.
    The standard forecast endpoint (api.open-meteo.com) only carries
    recent/future data and returns None for historical hours.
    """
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,visibility",
        "timezone": "Asia/Karachi",
        "start_date": start_date, "end_date": end_date
    }
    r = requests.get(OPENMETEO_WEATHER_ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def safe_val(value, fallback=0.0):
    """Explicit None check — keeps genuine 0.0, only replaces actual None."""
    return value if value is not None else fallback


def get_aqi_category(aqi):
    if aqi <= 50:   return "good"
    if aqi <= 100:  return "moderate"
    if aqi <= 150:  return "unhealthy_sensitive"
    if aqi <= 200:  return "unhealthy"
    if aqi <= 300:  return "very_unhealthy"
    return "hazardous"


def parse_records(aq_data, w_data):
    aq_hourly  = aq_data["hourly"]
    w_hourly   = w_data["hourly"]
    w_time_map = {t: i for i, t in enumerate(w_hourly["time"])}

    # Sanity check
    non_null = sum(1 for t in w_hourly["temperature_2m"] if t is not None)
    print(f"  Weather: {non_null}/{len(w_hourly['temperature_2m'])} non-null temperatures")
    print(f"  Sample temps: {w_hourly['temperature_2m'][:5]}")

    records = []
    for i, time_str in enumerate(aq_hourly["time"]):
        aqi = aq_hourly["us_aqi"][i]
        if aqi is None:
            continue
        aqi = float(aqi)
        if aqi > 400:
            continue

        w_idx = w_time_map.get(time_str, 0)
        dt    = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)

        records.append({
            "timestamp":    dt,
            "city":         CITY,
            "lat":          LAT,
            "lon":          LON,
            "aqi":          aqi,
            "pm2_5":        safe_val(aq_hourly["pm2_5"][i]),
            "pm10":         safe_val(aq_hourly["pm10"][i]),
            "no2":          safe_val(aq_hourly["nitrogen_dioxide"][i]),
            "o3":           safe_val(aq_hourly["ozone"][i]),
            "co":           safe_val(aq_hourly["carbon_monoxide"][i]),
            "so2":          safe_val(aq_hourly["sulphur_dioxide"][i]),
            "temperature":  safe_val(w_hourly["temperature_2m"][w_idx]),
            "humidity":     safe_val(w_hourly["relative_humidity_2m"][w_idx]),
            "pressure":     safe_val(w_hourly["surface_pressure"][w_idx]),
            "wind_speed":   safe_val(w_hourly["wind_speed_10m"][w_idx]),
            "wind_deg":     safe_val(w_hourly["wind_direction_10m"][w_idx]),
            "visibility":   safe_val(w_hourly["visibility"][w_idx]),
            "hour":         dt.hour,
            "day_of_week":  dt.weekday(),
            "day_of_month": dt.day,
            "month":        dt.month,
            "aqi_category": get_aqi_category(aqi),
        })

    return records


def compute_lag_and_rolling(records):
    df  = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    aqi = df["aqi"]

    for lag in LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = aqi.shift(lag).bfill()

    for window in ROLLING_WINDOWS:
        df[f"aqi_rolling_{window}h"] = (
            aqi.shift(1).rolling(window=window, min_periods=1).mean().round(2)
        )

    df["aqi_change_rate"] = (
        (aqi - aqi.shift(1)) / aqi.shift(1).replace(0, 1)
    ).fillna(0).round(4)

    return df.to_dict(orient="records")


def store_records(records, db):
    collection       = db[FEATURES_COLLECTION]
    inserted = updated = 0
    for record in records:
        result = collection.update_one(
            {"timestamp": record["timestamp"], "city": record["city"]},
            {"$set": record},
            upsert=True
        )
        if result.upserted_id:
            inserted += 1
        else:
            updated += 1
    print(f"Inserted: {inserted} | Updated (overwritten): {updated}")


def run_backfill(days=TRAINING_DAYS):
    print(f"Starting backfill for {CITY} — last {days} days...")
    db = get_db()

    end_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"Date range: {start_date} to {end_date}")

    print("Fetching air quality data (Open-Meteo AQ)...")
    aq_data = fetch_historical_air_quality(start_date, end_date)

    print("Fetching weather data (Open-Meteo Archive)...")
    w_data = fetch_historical_weather(start_date, end_date)

    print("Parsing records...")
    records = parse_records(aq_data, w_data)
    print(f"Parsed {len(records)} valid records")

    print("Computing lag and rolling features...")
    records = compute_lag_and_rolling(records)

    print("Storing in MongoDB...")
    store_records(records, db)

    # Verification
    df = pd.DataFrame(records)
    print(f"\nVerification:")
    for col in ["temperature", "humidity", "pressure", "wind_speed"]:
        zero_pct = (df[col] == 0).mean() * 100
        mean_val = df[col].mean()
        print(f"  {col:12s}  mean={mean_val:.2f}  zeros={zero_pct:.1f}%")

    print(f"\nBackfill complete — {len(records)} records stored.")


if __name__ == "__main__":
    run_backfill()