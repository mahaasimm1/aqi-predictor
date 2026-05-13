import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import FEATURES_COLLECTION
from datetime import datetime, timezone, timedelta


def get_recent_features(db, days=7):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    collection = db[FEATURES_COLLECTION]
    records = list(collection.find(
        {"timestamp": {"$gte": since}},
        {"_id": 0},
        sort=[("timestamp", 1)]
    ))
    return records


def get_latest_record(db):
    collection = db[FEATURES_COLLECTION]
    record = collection.find_one(sort=[("timestamp", -1)])
    if record:
        record.pop("_id", None)
    return record