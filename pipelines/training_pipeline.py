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
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, StackingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (TimeSeriesSplit, RandomizedSearchCV,
                                      GridSearchCV, cross_val_score)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config.settings import (
    MONGO_URI, MONGO_DB_NAME, FEATURES_COLLECTION,
    MODELS_COLLECTION, METRICS_COLLECTION,
    LAG_HOURS, ROLLING_WINDOWS
)

FORECAST_HORIZON = 1


def get_db():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB_NAME]


def load_features(db):
    records = list(db[FEATURES_COLLECTION].find({}, {"_id": 0}))
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} records from MongoDB")
    return df


def clean_data(df):
    weather_cols = [c for c in ["temperature", "humidity", "pressure", "wind_speed"]
                    if c in df.columns]
    all_zero   = (df[weather_cols] == 0).all(axis=1)
    zero_count = all_zero.sum()
    df.loc[all_zero, weather_cols] = np.nan
    df[weather_cols] = df[weather_cols].ffill().bfill()
    print(f"  Imputed {zero_count} zero-weather rows | {len(df)} records kept")
    df = df[df["aqi"] > 0].reset_index(drop=True)
    return df


def add_features(df):
    
    df = df.copy()

    # Log transforms
    for col in ["pm2_5", "pm10", "no2", "co", "so2"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # Wind interactions
    if "pm2_5" in df.columns and "wind_speed" in df.columns:
        df["pm2_5_x_wind"] = df["pm2_5"] / (df["wind_speed"] + 1)
    if "pm10" in df.columns and "wind_speed" in df.columns:
        df["pm10_x_wind"]  = df["pm10"]  / (df["wind_speed"] + 1)

    # Momentum
    if "aqi_lag_3h" in df.columns:
        df["aqi_momentum_3h"] = (df["aqi"] - df["aqi_lag_3h"]) / 3.0
    if "aqi_lag_6h" in df.columns:
        df["aqi_momentum_6h"] = (df["aqi"] - df["aqi_lag_6h"]) / 6.0

    # lag_48h (same-time two days ago)
    df["aqi_lag_48h"] = df["aqi"].shift(48)

    # peak traffic hour flag (EDA showed systematic residual error)
    df["peak_traffic_hour"] = df["hour"].between(15, 20).astype(int)

    # rolling volatility — std of last 6h AQI
    df["aqi_std_6h"] = (
        df["aqi"].shift(1).rolling(window=6, min_periods=2).std().round(3)
    )

    # Cyclical time
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def build_features_and_target(df):
    feature_cols = (
        ["pm2_5", "pm10", "no2", "o3", "co", "so2",
         "log_pm2_5", "log_pm10", "log_no2", "log_co", "log_so2",
         "temperature", "humidity", "pressure", "wind_speed",
         "pm2_5_x_wind", "pm10_x_wind",
         "hour_sin", "hour_cos", "month_sin", "month_cos",
         "peak_traffic_hour",
         "aqi_momentum_3h", "aqi_momentum_6h",
         "aqi_std_6h"]
        + [f"aqi_lag_{lag}h" for lag in LAG_HOURS]
        + ["aqi_lag_48h"]
        + [f"aqi_rolling_{w}h" for w in ROLLING_WINDOWS]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.copy()
    df["target"] = df["aqi"].shift(-FORECAST_HORIZON)
    df = df.dropna(subset=["target"] + feature_cols).reset_index(drop=True)

    X = df[feature_cols].values.astype(float)
    y = df["target"].values.astype(float)

    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    return X, y, feature_cols


def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {model_name:40s}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def tscv_score(model, X, y, n_splits=5):
    """
    TimeSeriesSplit cross-validation.
    More reliable estimate than a single holdout split —
    averages RMSE, MAE, and R² over 5 chronological train/test windows.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmses, maes, r2s = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        preds = np.clip(model.predict(X_te), 0, 500)

        rmses.append(np.sqrt(mean_squared_error(y_te, preds)))
        maes.append(mean_absolute_error(y_te, preds))
        r2s.append(r2_score(y_te, preds))

    return {
        "rmse": round(float(np.mean(rmses)), 4),
        "mae":  round(float(np.mean(maes)),  4),
        "r2":   round(float(np.mean(r2s)),   4),
        "rmse_std": round(float(np.std(rmses)), 4),
        "mae_std":  round(float(np.std(maes)),  4),
        "r2_std":   round(float(np.std(r2s)),   4),
    }


def train_ridge(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    Xtr = np.clip(scaler.fit_transform(X_train), -10, 10)
    Xte = np.clip(scaler.transform(X_test),      -10, 10)
    search = GridSearchCV(Ridge(),
                          {"alpha": [0.01, 0.1, 1, 10, 100, 1000]},
                          cv=TimeSeriesSplit(5),
                          scoring="neg_root_mean_squared_error")
    search.fit(Xtr, y_train)
    model  = search.best_estimator_
    preds  = np.clip(model.predict(Xte), 0, 500)
    metrics = evaluate(y_test, preds, "Ridge")
    return {"model": model, "scaler": scaler, "metrics": metrics, "name": "ridge"}


def train_random_forest(X_train, y_train, X_test, y_test):
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [200, 300, 500],
         "max_depth": [15, 20, None],
         "min_samples_split": [2, 5],
         "min_samples_leaf": [1, 2],
         "max_features": ["sqrt", 0.4, 0.5]},
        n_iter=30, cv=TimeSeriesSplit(5),
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    model  = search.best_estimator_
    print(f"    RF: {search.best_params_}")
    preds  = np.clip(model.predict(X_test), 0, 500)
    metrics = evaluate(y_test, preds, "RandomForest")

    # TimeSeriesSplit CV score on full dataset
    cv = tscv_score(model, np.vstack([X_train, X_test]),
                        np.concatenate([y_train, y_test]))
    print(f"    RF  CV  RMSE={cv['rmse']} ± {cv['rmse_std']}  "
          f"MAE={cv['mae']} ± {cv['mae_std']}  R2={cv['r2']} ± {cv['r2_std']}")
    metrics["rmse_cv"] = cv["rmse"]
    metrics["mae_cv"]  = cv["mae"]
    metrics["r2_cv"]   = cv["r2"]

    return {"model": model, "scaler": None, "metrics": metrics, "name": "random_forest"}


def train_extra_trees(X_train, y_train, X_test, y_test):
    search = RandomizedSearchCV(
        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [200, 300, 500],
         "max_depth": [15, 20, None],
         "min_samples_split": [2, 5],
         "min_samples_leaf": [1, 2],
         "max_features": ["sqrt", 0.4, 0.5]},
        n_iter=20, cv=TimeSeriesSplit(5),
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    model  = search.best_estimator_
    print(f"    ET: {search.best_params_}")
    preds  = np.clip(model.predict(X_test), 0, 500)
    metrics = evaluate(y_test, preds, "ExtraTrees")
    cv = tscv_score(model, np.vstack([X_train, X_test]),
                        np.concatenate([y_train, y_test]))
    print(f"    ET  CV  RMSE={cv['rmse']} ± {cv['rmse_std']}  "
          f"MAE={cv['mae']} ± {cv['mae_std']}  R2={cv['r2']} ± {cv['r2_std']}")
    metrics["rmse_cv"] = cv["rmse"]
    metrics["mae_cv"]  = cv["mae"]
    metrics["r2_cv"]   = cv["r2"]
    return {"model": model, "scaler": None, "metrics": metrics, "name": "extra_trees"}


def train_xgboost(X_train, y_train, X_test, y_test):
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0, tree_method="hist"),
        {"n_estimators": [300, 500, 700],
         "max_depth": [3, 4, 5],
         "learning_rate": [0.01, 0.05, 0.1],
         "subsample": [0.7, 0.8, 0.9],
         "colsample_bytree": [0.7, 0.8, 1.0],
         "min_child_weight": [1, 3, 5],
         "gamma": [0, 0.05, 0.1],
         "reg_alpha": [0, 0.1],
         "reg_lambda": [0.5, 1.0, 2.0]},
        n_iter=50, cv=TimeSeriesSplit(5),
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    model  = search.best_estimator_
    print(f"    XGB: {search.best_params_}")
    preds  = np.clip(model.predict(X_test), 0, 500)
    metrics = evaluate(y_test, preds, "XGBoost")
    cv = tscv_score(model, np.vstack([X_train, X_test]),
                        np.concatenate([y_train, y_test]))
    print(f"    XGB CV  RMSE={cv['rmse']} ± {cv['rmse_std']}  "
          f"MAE={cv['mae']} ± {cv['mae_std']}  R2={cv['r2']} ± {cv['r2_std']}")
    metrics["rmse_cv"] = cv["rmse"]
    metrics["mae_cv"]  = cv["mae"]
    metrics["r2_cv"]   = cv["r2"]
    return {"model": model, "scaler": None, "metrics": metrics, "name": "xgboost"}


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": [200, 300, 500],
         "max_depth": [3, 4, 5],
         "learning_rate": [0.01, 0.05, 0.1],
         "subsample": [0.7, 0.8, 0.9],
         "max_features": ["sqrt", 0.5]},
        n_iter=30, cv=TimeSeriesSplit(5),
        scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    model  = search.best_estimator_
    print(f"    GB: {search.best_params_}")
    preds  = np.clip(model.predict(X_test), 0, 500)
    metrics = evaluate(y_test, preds, "GradientBoosting")
    cv = tscv_score(model, np.vstack([X_train, X_test]),
                        np.concatenate([y_train, y_test]))
    print(f"    GB  CV  RMSE={cv['rmse']} ± {cv['rmse_std']}  "
          f"MAE={cv['mae']} ± {cv['mae_std']}  R2={cv['r2']} ± {cv['r2_std']}")
    metrics["rmse_cv"] = cv["rmse"]
    metrics["mae_cv"]  = cv["mae"]
    metrics["r2_cv"]   = cv["r2"]
    return {"model": model, "scaler": None, "metrics": metrics, "name": "gradient_boosting"}


def train_stacking(X_train, y_train, X_test, y_test,
                   rf_result, et_result, gb_result):
    stack = StackingRegressor(
        estimators=[
            ("rf", rf_result["model"]),
            ("et", et_result["model"]),
            ("gb", gb_result["model"]),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=1,          # n_jobs=1 avoids joblib multiprocess issues
        passthrough=False,
    )
    stack.fit(X_train, y_train)
    preds   = np.clip(stack.predict(X_test), 0, 500)
    metrics = evaluate(y_test, preds, "Stacking (RF+ET+GB->Ridge)")

    # Also report TimeSeriesCV score for stacking
    tscv = TimeSeriesSplit(n_splits=5)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv = tscv_score(stack, X_all, y_all)
    print(f"    Stack CV  RMSE={cv['rmse']} ± {cv['rmse_std']}  "
          f"MAE={cv['mae']} ± {cv['mae_std']}  R2={cv['r2']} ± {cv['r2_std']}")
    metrics["rmse_cv"] = cv["rmse"]
    metrics["mae_cv"]  = cv["mae"]
    metrics["r2_cv"]   = cv["r2"]

    return {"model": stack, "scaler": None, "metrics": metrics, "name": "stacking"}


def save_model(best, feature_cols, db):
    fs = gridfs.GridFS(db)
    artifact = {
        "model":            best["model"],
        "scaler":           best["scaler"],
        "feature_cols":     feature_cols,
        "forecast_horizon": FORECAST_HORIZON,
        "predict_delta":    False,
    }
    model_name = f"aqi_{best['name']}_model"
    old = db[MODELS_COLLECTION].find_one({"name": model_name})
    if old and "gridfs_id" in old:
        try:
            fs.delete(old["gridfs_id"])
        except Exception:
            pass
    gridfs_id = fs.put(pickle.dumps(artifact), filename=model_name)
    db[MODELS_COLLECTION].update_one(
        {"name": model_name},
        {"$set": {
            "name":             model_name,
            "model_type":       best["name"],
            "gridfs_id":        gridfs_id,
            "metrics":          best["metrics"],   # holdout metrics
            "metrics_cv": {                        # TimeSeriesCV metrics
                "rmse": best["metrics"].get("rmse_cv"),
                "mae":  best["metrics"].get("mae_cv"),
                "r2":   best["metrics"].get("r2_cv"),
            },
            "feature_cols":     feature_cols,
            "forecast_horizon": FORECAST_HORIZON,
            "trained_at":       datetime.now(timezone.utc),
            "is_best":          True,
        }},
        upsert=True
    )
    db[MODELS_COLLECTION].update_many(
        {"name": {"$ne": model_name}},
        {"$set": {"is_best": False}}
    )
    print(f"Saved: {model_name}")


def log_metrics(results, db):
    ts = datetime.now(timezone.utc)
    for r in results:
        db[METRICS_COLLECTION].insert_one({
            "model_name": r["name"],
            "metrics":    r["metrics"],
            "logged_at":  ts,
        })


def run_training_pipeline():
    print("=" * 60)
    print("Training pipeline")
    print("=" * 60)
    db = get_db()

    df = load_features(db)
    if len(df) < 100:
        print("Not enough data.")
        return

    print("\nCleaning")
    df = clean_data(df)

    print("\nFeature engineering")
    df = add_features(df)

    print("\n--- Building feature matrix ---")
    X, y, feature_cols = build_features_and_target(df)

    # Chronological split for final held-out evaluation
    split   = int(len(X) * 0.8)
    X_train = X[:split];  y_train = y[:split]
    X_test  = X[split:];  y_test  = y[split:]
    print(f"  Train={len(X_train)}, Test={len(X_test)}\n")

    print("Training (TimeSeriesCV hyperparameter search)")
    ridge_r = train_ridge(X_train, y_train, X_test, y_test)
    rf_r    = train_random_forest(X_train, y_train, X_test, y_test)
    et_r    = train_extra_trees(X_train, y_train, X_test, y_test)
    xgb_r   = train_xgboost(X_train, y_train, X_test, y_test)
    gb_r    = train_gradient_boosting(X_train, y_train, X_test, y_test)

    print("\nStacking")
    stack_r = train_stacking(X_train, y_train, X_test, y_test,
                             rf_r, et_r, gb_r)

    results = [ridge_r, rf_r, et_r, xgb_r, gb_r, stack_r]

    # Pick winner by CV R² if available (more reliable), else holdout RMSE
    def sort_key(r):
        if "r2_cv" in r["metrics"]:
            return -r["metrics"]["r2_cv"]   # higher is better
        return r["metrics"]["rmse"]          # lower is better

    best = min(results, key=sort_key)

    print(f"\n{'='*60}")
    print(f"All model results (holdout | TimeSeriesCV):")
    print(f"  {'Model':20s}  {'RMSE':>8}  {'MAE':>8}  {'R2':>8}  "
          f"{'CV_RMSE':>10}  {'CV_MAE':>8}  {'CV_R2':>8}")
    print(f"  {'-'*80}")
    for r in results:
        m = r["metrics"]
        cv_str = (f"  {m.get('rmse_cv','n/a'):>10}  {m.get('mae_cv','n/a'):>8}  {m.get('r2_cv','n/a'):>8}"
                  if "r2_cv" in m else "  n/a")
        print(f"  {r['name']:20s}  {m['rmse']:>8}  {m['mae']:>8}  {m['r2']:>8}{cv_str}")
    print(f"\nWinner : {best['name']}")
    print(f"Holdout")
    print(f"  RMSE : {best['metrics']['rmse']}")
    print(f"  MAE  : {best['metrics']['mae']}")
    print(f"  R2   : {best['metrics']['r2']}")
    if "r2_cv" in best["metrics"]:
        print(f"TimeSeriesCV (5 folds)")
        print(f"  RMSE : {best['metrics']['rmse_cv']}")
        print(f"  MAE  : {best['metrics']['mae_cv']}")
        print(f"  R2   : {best['metrics']['r2_cv']}")
    print("=" * 60)

    save_model(best, feature_cols, db)
    log_metrics(results, db)
    print("\nDone.")


if __name__ == "__main__":
    run_training_pipeline()