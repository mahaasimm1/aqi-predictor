import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from config.settings import (
    MONGO_URI, MONGO_DB_NAME, PREDICTIONS_COLLECTION,
    FEATURES_COLLECTION, MONITORING_COLLECTION,
    DRIFT_THRESHOLD, MONITORING_WINDOW_DAYS
)


def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


def load_prediction_vs_actual(db):
    since = datetime.now(timezone.utc) - timedelta(days=MONITORING_WINDOW_DAYS)

    predictions = list(db[PREDICTIONS_COLLECTION].find(
        {"timestamp": {"$gte": since}},
        {"_id": 0, "timestamp": 1, "current_aqi": 1}
    ))

    if not predictions:
        print("No predictions found in monitoring window.")
        return [], []

    actuals = list(db[FEATURES_COLLECTION].find(
        {"timestamp": {"$gte": since}},
        {"_id": 0, "timestamp": 1, "aqi": 1}
    ))

    actual_map = {r["timestamp"]: r["aqi"] for r in actuals}

    y_pred = []
    y_actual = []

    for pred in predictions:
        actual_aqi = actual_map.get(pred["timestamp"])
        if actual_aqi is not None:
            y_pred.append(pred["current_aqi"])
            y_actual.append(actual_aqi)

    return y_pred, y_actual


def compute_drift_metrics(y_pred, y_actual):
    y_pred = np.array(y_pred)
    y_actual = np.array(y_actual)

    mae = float(np.mean(np.abs(y_pred - y_actual)))
    rmse = float(np.sqrt(np.mean((y_pred - y_actual) ** 2)))
    mean_actual = float(np.mean(y_actual))
    drift_ratio = mae / max(mean_actual, 1)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_actual_aqi": round(mean_actual, 4),
        "drift_ratio": round(drift_ratio, 4),
        "drift_detected": drift_ratio > DRIFT_THRESHOLD,
        "sample_size": len(y_pred)
    }


def log_metrics(metrics, db):
    db[MONITORING_COLLECTION].insert_one({
        "timestamp": datetime.now(timezone.utc),
        **metrics
    })
    print(f"Metrics logged: {metrics}")


def open_github_issue(metrics):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")

    if not token or not repo:
        print("GitHub token or repo not set, skipping issue creation.")
        return

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    body = {
        "title": f"Model Drift Detected — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "body": (
            f"**Drift detected in AQI predictions.**\n\n"
            f"- MAE: {metrics['mae']}\n"
            f"- RMSE: {metrics['rmse']}\n"
            f"- Drift Ratio: {metrics['drift_ratio']} (threshold: {DRIFT_THRESHOLD})\n"
            f"- Sample Size: {metrics['sample_size']}\n\n"
            f"Consider retraining the model or checking the data pipeline."
        ),
        "labels": ["monitoring", "drift"]
    }

    response = requests.post(url, json=body, headers=headers)
    if response.status_code == 201:
        print(f"GitHub issue created: {response.json()['html_url']}")
    else:
        print(f"Failed to create issue: {response.status_code}")


def run_monitoring():
    print("Starting monitoring run...")
    db = get_db()

    y_pred, y_actual = load_prediction_vs_actual(db)

    if not y_pred:
        print("Not enough matched prediction/actual pairs.")
        return

    metrics = compute_drift_metrics(y_pred, y_actual)
    log_metrics(metrics, db)

    if metrics["drift_detected"]:
        print(f"Drift detected! Ratio: {metrics['drift_ratio']} > {DRIFT_THRESHOLD}")
        open_github_issue(metrics)
    else:
        print(f"No drift detected. Ratio: {metrics['drift_ratio']}")

    print("Monitoring complete.")


if __name__ == "__main__":
    run_monitoring()