import sys
sys.path.append('.')
from pymongo import MongoClient
from config.settings import MONGO_URI, MONGO_DB_NAME
from datetime import datetime, timezone, timedelta
import numpy as np

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

print("=" * 50)
print("FULL SYSTEM HEALTH CHECK")
print("=" * 50)

# 1. Feature collection
print("\n--- Feature Collection ---")
total_features = db['aqi_features'].count_documents({})
print(f"Total records: {total_features}")

latest_feature = db['aqi_features'].find_one(sort=[("timestamp", -1)])
oldest_feature = db['aqi_features'].find_one(sort=[("timestamp", 1)])
print(f"Oldest record: {oldest_feature['timestamp']}")
print(f"Latest record: {latest_feature['timestamp']}")

hours_since_last = (datetime.now(timezone.utc) - latest_feature['timestamp'].replace(tzinfo=timezone.utc)).total_seconds() / 3600
print(f"Hours since last feature collected: {hours_since_last:.1f}")
if hours_since_last > 2:
    print("WARNING: Feature pipeline may not be running correctly")
else:
    print("OK: Feature pipeline is collecting data")

# 2. Check for duplicate timestamps
print("\n--- Duplicate Check ---")
pipeline = [
    {"$group": {"_id": "$timestamp", "count": {"$sum": 1}}},
    {"$match": {"count": {"$gt": 1}}}
]
duplicates = list(db['aqi_features'].aggregate(pipeline))
print(f"Duplicate timestamps: {len(duplicates)}")
if duplicates:
    print("WARNING: Duplicates found")
else:
    print("OK: No duplicates")

# 3. Check for missing hours
print("\n--- Gap Check (last 7 days) ---")
since = datetime.now(timezone.utc) - timedelta(days=7)
recent = list(db['aqi_features'].find(
    {"timestamp": {"$gte": since}},
    {"_id": 0, "timestamp": 1}
))
recent_times = sorted([r['timestamp'] for r in recent])
gaps = []
for i in range(1, len(recent_times)):
    diff = (recent_times[i] - recent_times[i-1]).total_seconds() / 3600
    if diff > 2:
        gaps.append({
            "from": recent_times[i-1],
            "to": recent_times[i],
            "gap_hours": round(diff, 1)
        })
print(f"Records in last 7 days: {len(recent_times)}")
print(f"Expected (~168 hours): 168")
print(f"Gaps > 2 hours found: {len(gaps)}")
for g in gaps[:5]:
    print(f"  Gap: {g['from']} to {g['to']} ({g['gap_hours']}h)")

# 4. AQI data quality
print("\n--- AQI Data Quality ---")
zero_aqi = db['aqi_features'].count_documents({"aqi": {"$lt": 5}})
high_aqi = db['aqi_features'].count_documents({"aqi": {"$gt": 400}})
null_aqi = db['aqi_features'].count_documents({"aqi": None})
print(f"Near-zero AQI records (< 5): {zero_aqi}")
print(f"Extreme AQI records (> 400): {high_aqi}")
print(f"Null AQI records: {null_aqi}")
if zero_aqi > 0 or high_aqi > 0:
    print("WARNING: Bad AQI values exist")
else:
    print("OK: AQI values look clean")

# 5. Model registry
print("\n--- Model Registry ---")
models = list(db['models'].find({}, {"_id": 0, "name": 1, "model_type": 1, "is_best": 1, "metrics": 1, "trained_at": 1}))
for m in models:
    flag = "BEST" if m.get("is_best") else "    "
    print(f"[{flag}] {m['model_type']} | RMSE: {m['metrics']['rmse']} | trained: {m.get('trained_at', 'unknown')}")

best_models = [m for m in models if m.get("is_best")]
if len(best_models) == 0:
    print("WARNING: No best model marked")
elif len(best_models) > 1:
    print("WARNING: Multiple models marked as best")
else:
    print("OK: Exactly one best model")

# 6. Training metrics history
print("\n--- Training History (last 3 runs) ---")
from itertools import groupby
all_metrics = list(db['model_metrics'].find(
    {}, {"_id": 0},
    sort=[("logged_at", -1)],
    limit=12
))
if all_metrics:
    dates = {}
    for m in all_metrics:
        date = m['logged_at'].strftime("%Y-%m-%d %H:%M")
        if date not in dates:
            dates[date] = []
        dates[date].append(m)
    for date, runs in list(dates.items())[:3]:
        print(f"\nRun at {date}:")
        for r in sorted(runs, key=lambda x: x['metrics']['rmse']):
            print(f"  {r['model_name']}: RMSE {r['metrics']['rmse']:.2f}")

# 7. Predictions collection
print("\n--- Predictions Collection ---")
total_preds = db['predictions'].count_documents({})
print(f"Total predictions stored: {total_preds}")
latest_pred = db['predictions'].find_one(sort=[("timestamp", -1)])
if latest_pred:
    hours_since_pred = (datetime.now(timezone.utc) - latest_pred['timestamp'].replace(tzinfo=timezone.utc)).total_seconds() / 3600
    print(f"Latest prediction: {latest_pred['timestamp']} ({hours_since_pred:.1f}h ago)")
    print(f"Latest predicted AQI: {latest_pred['current_aqi']}")

# 8. Monitoring logs
print("\n--- Monitoring Logs (last 5) ---")
monitoring = list(db['monitoring_logs'].find(
    {}, {"_id": 0},
    sort=[("timestamp", -1)],
    limit=5
))
if monitoring:
    for m in monitoring:
        drift = "DRIFT DETECTED" if m.get("drift_detected") else "OK"
        print(f"{m['timestamp']} | MAE: {m.get('mae', 'N/A')} | Drift ratio: {m.get('drift_ratio', 'N/A')} | {drift}")
else:
    print("No monitoring logs found")

print("\n" + "=" * 50)
print("HEALTH CHECK COMPLETE")
print("=" * 50)