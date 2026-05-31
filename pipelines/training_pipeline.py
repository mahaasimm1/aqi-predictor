import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pymongo import MongoClient
import gridfs
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config.settings import (
    MONGO_URI, MONGO_DB_NAME, FEATURES_COLLECTION,
    MODELS_COLLECTION, METRICS_COLLECTION,
    LAG_HOURS, ROLLING_WINDOWS
)

# The model predicts AQI 1 hour ahead.
# At inference time we call it iteratively for each future hour,
# updating lag/rolling features and forecasted weather each step.
FORECAST_HORIZON = 1


def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


def load_features(db):
    collection = db[FEATURES_COLLECTION]
    records = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} records from MongoDB")
    return df


def build_features_and_target(df):
    """
    Build X and y where:
      - X at row i = all features observed at time T
      - y at row i = AQI at time T + 1h  (1-step-ahead target)

    Weather features (pm2_5, temperature, etc.) are the CURRENT observed values.
    At inference time we replace them with Open-Meteo 3-day forecasts.
    Lag/rolling features summarise recent AQI history seen at time T.
    """
    feature_cols = (
        ["pm2_5", "pm10", "no2", "o3", "co", "so2",
         "temperature", "humidity", "pressure", "wind_speed",
         "hour", "day_of_week", "month", "aqi_change_rate"]
        + [f"aqi_lag_{lag}h" for lag in LAG_HOURS]
        + [f"aqi_rolling_{w}h" for w in ROLLING_WINDOWS]
    )

    feature_cols = [c for c in feature_cols if c in df.columns]

    # Target: AQI one hour later
    df = df.copy()
    df["target"] = df["aqi"].shift(-FORECAST_HORIZON)
    df = df.dropna(subset=["target"] + feature_cols).reset_index(drop=True)

    X = df[feature_cols].values
    y = df["target"].values

    return X, y, feature_cols


def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def train_ridge(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_s = np.clip(X_train_s, -10, 10)
    X_test_s = np.clip(X_test_s, -10, 10)

    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    search = GridSearchCV(Ridge(), param_grid, cv=5,
                          scoring="neg_root_mean_squared_error")
    search.fit(X_train_s, y_train)
    model = search.best_estimator_
    print(f"Best Ridge alpha: {model.alpha}")

    preds = np.clip(model.predict(X_test_s), 0, 500)
    metrics = evaluate(y_test, preds, "Ridge")
    return {"model": model, "scaler": scaler, "metrics": metrics, "name": "ridge"}


def train_random_forest(X_train, y_train, X_test, y_test):
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [4, 8, 16],
        "max_features": ["sqrt", "log2"]
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist, n_iter=20, cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best RF params: {search.best_params_}")

    preds = np.clip(model.predict(X_test), 0, 500)
    metrics = evaluate(y_test, preds, "RandomForest")
    return {"model": model, "scaler": None, "metrics": metrics, "name": "random_forest"}


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_samples_split": [5, 10],
        "subsample": [0.8, 1.0]
    }
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_dist, n_iter=20, cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best GB params: {search.best_params_}")

    preds = np.clip(model.predict(X_test), 0, 500)
    metrics = evaluate(y_test, preds, "GradientBoosting")
    return {"model": model, "scaler": None, "metrics": metrics, "name": "gradient_boosting"}


def train_xgboost(X_train, y_train, X_test, y_test):
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [3, 5, 7],
        "gamma": [0, 0.1, 0.2],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0]
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_dist, n_iter=30, cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best XGB params: {search.best_params_}")

    preds = np.clip(model.predict(X_test), 0, 500)
    metrics = evaluate(y_test, preds, "XGBoost")
    return {"model": model, "scaler": None, "metrics": metrics, "name": "xgboost"}


def train_lstm(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_r = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_r = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, X_train_s.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train_r, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    preds = np.clip(model.predict(X_test_r).flatten(), 0, 500)
    metrics = evaluate(y_test, preds, "LSTM")
    return {"model": model, "scaler": scaler, "metrics": metrics, "name": "lstm"}


def save_model(best, feature_cols, db):
    fs = gridfs.GridFS(db)

    artifact = {
        "model": best["model"],
        "scaler": best["scaler"],
        "feature_cols": feature_cols,
        "forecast_horizon": FORECAST_HORIZON,
    }

    model_bytes = pickle.dumps(artifact)
    model_name = f"aqi_{best['name']}_model"

    old = db[MODELS_COLLECTION].find_one({"name": model_name})
    if old and "gridfs_id" in old:
        try:
            fs.delete(old["gridfs_id"])
        except Exception:
            pass

    gridfs_id = fs.put(model_bytes, filename=model_name)

    db[MODELS_COLLECTION].update_one(
        {"name": model_name},
        {"$set": {
            "name": model_name,
            "model_type": best["name"],
            "gridfs_id": gridfs_id,
            "metrics": best["metrics"],
            "feature_cols": feature_cols,
            "forecast_horizon": FORECAST_HORIZON,
            "trained_at": datetime.now(timezone.utc),
            "is_best": True
        }},
        upsert=True
    )

    # Mark all other models as not best
    db[MODELS_COLLECTION].update_many(
        {"name": {"$ne": model_name}},
        {"$set": {"is_best": False}}
    )

    print(f"Saved best model: {model_name}")


def log_metrics(results, db):
    timestamp = datetime.now(timezone.utc)
    for result in results:
        db[METRICS_COLLECTION].insert_one({
            "model_name": result["name"],
            "metrics": result["metrics"],
            "logged_at": timestamp
        })


def run_training_pipeline():
    print("Starting training pipeline...")
    db = get_db()

    df = load_features(db)
    if len(df) < 100:
        print("Not enough data to train. Run backfill first.")
        return

    X, y, feature_cols = build_features_and_target(df)
    print(f"Feature matrix: {X.shape} | Target: {y.shape}")
    print(f"Features used: {feature_cols}")

    # Chronological split — never shuffle time-series data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("\nTraining models...")
    results = [
        train_ridge(X_train, y_train, X_test, y_test),
        train_random_forest(X_train, y_train, X_test, y_test),
        train_gradient_boosting(X_train, y_train, X_test, y_test),
        train_xgboost(X_train, y_train, X_test, y_test),
        train_lstm(X_train, y_train, X_test, y_test),
    ]

    best = min(results, key=lambda r: r["metrics"]["rmse"])
    print(f"\nBest model: {best['name']} (RMSE: {best['metrics']['rmse']})")

    save_model(best, feature_cols, db)
    log_metrics(results, db)

    print("Training pipeline complete.")


if __name__ == "__main__":
    run_training_pipeline()