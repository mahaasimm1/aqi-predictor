import sys
sys.path.append('.')
from pymongo import MongoClient
from config.settings import MONGO_URI, MONGO_DB_NAME
from datetime import datetime, timezone, timedelta
import numpy as np

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

predictions = list(db['predictions'].find(
    {},
    {"_id": 0, "timestamp": 1, "current_aqi": 1, "forecasts": 1}
))

actuals = list(db['aqi_features'].find(
    {},
    {"_id": 0, "timestamp": 1, "aqi": 1}
))

actual_map = {r["timestamp"]: r["aqi"] for r in actuals}

errors = []
for pred in predictions:
    pred_time = pred["timestamp"]
    forecasts = pred.get("forecasts", [])

    for forecast in forecasts:
        horizon = forecast.get("horizon_hours", 24)
        predicted_aqi = forecast.get("predicted_aqi")
        future_time = pred_time + timedelta(hours=horizon)

        # Find closest actual within 30 minutes
        closest_actual = None
        min_diff = timedelta(minutes=30)
        for actual_time, actual_aqi in actual_map.items():
            diff = abs(actual_time.replace(tzinfo=None) - future_time.replace(tzinfo=None))
            if diff < min_diff:
                min_diff = diff
                closest_actual = actual_aqi

        if closest_actual is not None:
            error = abs(predicted_aqi - closest_actual)
            errors.append({
                "horizon": horizon,
                "predicted": predicted_aqi,
                "actual": closest_actual,
                "error": error
            })

if errors:
    print(f"Forecast validations found: {len(errors)}")
    for h in [24, 48, 72]:
        h_errors = [e["error"] for e in errors if e["horizon"] == h]
        if h_errors:
            print(f"+{h}h MAE: {np.mean(h_errors):.2f} | samples: {len(h_errors)}")
    print(f"\nSample predictions vs actuals:")
    for e in errors[:5]:
        print(f"  +{e['horizon']}h | predicted: {e['predicted']} | actual: {e['actual']} | error: {e['error']:.1f}")
else:
    print("No forecast validations found")