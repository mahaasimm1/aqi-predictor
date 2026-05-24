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
    LAG_HOURS, ROLLING_WINDOWS, FORECAST_HOURS
)


# MongoDB connection

def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


# Load features from MongoDB

def load_features(db):
    collection = db[FEATURES_COLLECTION]
    records = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} records from MongoDB")
    return df


# Build feature matrix and target

def build_features_and_target(df, forecast_horizon=1):
    feature_cols = (
        ["pm2_5", "pm10", "no2", "o3", "co", "so2",
         "temperature", "humidity", "pressure", "wind_speed",
         "hour", "day_of_week", "month", "aqi_change_rate"]
        + [f"aqi_lag_{lag}h" for lag in LAG_HOURS]
        + [f"aqi_rolling_{w}h" for w in ROLLING_WINDOWS]
    )

    feature_cols = [c for c in feature_cols if c in df.columns]

    # Single target: AQI forecast_horizon hours ahead
    df["target"] = df["aqi"].shift(-forecast_horizon)
    df = df.dropna(subset=["target"] + feature_cols)

    X = df[feature_cols].values
    y = df["target"].values  # 1D array

    return X, y, feature_cols


# Evaluate predictions

def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


# Train Ridge Regression

def train_ridge(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = np.clip(X_train_scaled, -10, 10)
    X_test_scaled = np.clip(X_test_scaled, -10, 10)

    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    search = GridSearchCV(Ridge(), param_grid, cv=5,
                         scoring="neg_root_mean_squared_error")
    search.fit(X_train_scaled, y_train)
    model = search.best_estimator_
    print(f"Best Ridge alpha: {model.alpha}")

    preds = model.predict(X_test_scaled)
    preds = np.clip(preds, 0, 500)
    metrics = evaluate(y_test, preds, "Ridge")

    return {"model": model, "scaler": scaler, "metrics": metrics, "name": "ridge"}


# Train Random Forest

def train_random_forest(X_train, y_train, X_test, y_test):
    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [4, 8, 16],
        "max_features": ["sqrt", "log2"]
    }

    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=20, cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best RF params: {search.best_params_}")

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 500)
    metrics = evaluate(y_test, preds, "RandomForest")

    return {"model": model, "scaler": None, "metrics": metrics, "name": "random_forest"}


# Train Gradient Boosting

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_samples_split": [5, 10],
        "subsample": [0.8, 1.0]
    }

    base_model = GradientBoostingRegressor(random_state=42)
    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=20, cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best GB params: {search.best_params_}")

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 500)
    metrics = evaluate(y_test, preds, "GradientBoosting")

    return {"model": model, "scaler": None, "metrics": metrics, "name": "gradient_boosting"}


# Train XGBoost

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

    base_model = XGBRegressor(random_state=42, verbosity=0)
    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=30, cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best XGB params: {search.best_params_}")

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 500)
    metrics = evaluate(y_test, preds, "XGBoost")

    return {"model": model, "scaler": None, "metrics": metrics, "name": "xgboost"}


# Train LSTM

def train_lstm(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_r = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_r = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train_r, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    preds = model.predict(X_test_r).flatten()
    preds = np.clip(preds, 0, 500)
    metrics = evaluate(y_test, preds, "LSTM")

    return {"model": model, "scaler": scaler, "metrics": metrics, "name": "lstm"}


# Save best model to MongoDB GridFS

def save_model(best, feature_cols, db):
    fs = gridfs.GridFS(db)

    artifact = {
        "model": best["model"],
        "scaler": best["scaler"],
        "feature_cols": feature_cols,
        "is_arima": False
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
            "trained_at": datetime.now(timezone.utc),
            "is_best": True
        }},
        upsert=True
    )

    db[MODELS_COLLECTION].update_many(
        {"name": {"$ne": model_name}},
        {"$set": {"is_best": False}}
    )

    print(f"Saved best model: {model_name}")


# Log metrics for all models

def log_metrics(results, db):
    collection = db[METRICS_COLLECTION]
    timestamp = datetime.now(timezone.utc)
    for result in results:
        collection.insert_one({
            "model_name": result["name"],
            "metrics": result["metrics"],
            "logged_at": timestamp
        })


# Main training pipeline

def run_training_pipeline():
    print("Starting training pipeline...")
    db = get_db()

    df = load_features(db)

    if len(df) < 100:
        print("Not enough data to train. Run backfill first.")
        return

    X, y, feature_cols = build_features_and_target(df, forecast_horizon=1)
    print(f"Feature matrix: {X.shape} | Target: {y.shape}")
    print(f"Features used: {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("\nTraining models...")
    ridge_result = train_ridge(X_train, y_train, X_test, y_test)
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    gb_result = train_gradient_boosting(X_train, y_train, X_test, y_test)
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
    lstm_result = train_lstm(X_train, y_train, X_test, y_test)

    results = [ridge_result, rf_result, gb_result, xgb_result, lstm_result]

    best = min(results, key=lambda r: r["metrics"]["rmse"])
    print(f"\nBest model: {best['name']} (RMSE: {best['metrics']['rmse']})")

    save_model(best, feature_cols, db)
    log_metrics(results, db)

    for result in results:
        db[MODELS_COLLECTION].update_one(
            {"name": f"aqi_{result['name']}_model"},
            {"$set": {
                "trained_at": datetime.now(timezone.utc),
                "metrics": result["metrics"]
            }},
            upsert=True
        )

    print("Training pipeline complete.")


if __name__ == "__main__":
    run_training_pipeline()