from flask import Blueprint, jsonify, current_app, request
from config.settings import FEATURES_COLLECTION
from datetime import datetime, timezone, timedelta

history_bp = Blueprint("history", __name__)

@history_bp.route("/history")
def history():
    db = current_app.config["DB"]
    days = int(request.args.get("days", 7))
    since = datetime.now(timezone.utc) - timedelta(days=days)

    collection = db[FEATURES_COLLECTION]
    records = list(collection.find(
        {"timestamp": {"$gte": since}},
        {"_id": 0, "timestamp": 1, "aqi": 1, "pm2_5": 1,
         "pm10": 1, "no2": 1, "o3": 1, "aqi_category": 1},
        sort=[("timestamp", 1)]
    ))

    for r in records:
        r["timestamp"] = r["timestamp"].isoformat()

    return jsonify({
        "city": "karachi",
        "days": days,
        "count": len(records),
        "data": records
    })