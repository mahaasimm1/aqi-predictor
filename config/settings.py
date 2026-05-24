import os
from dotenv import load_dotenv

load_dotenv()

# --- Location ---
CITY = os.getenv("CITY", "karachi")
LAT = float(os.getenv("LAT", 24.8607))
LON = float(os.getenv("LON", 67.0011))

# Open-Meteo API (no key required)
OPENMETEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENMETEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_PARAMS = "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide,us_aqi"
WEATHER_PARAMS = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,visibility"

# --- MongoDB ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "aqi_predictor")

# Collections
FEATURES_COLLECTION = "aqi_features"
MODELS_COLLECTION = "models"
METRICS_COLLECTION = "model_metrics"
PREDICTIONS_COLLECTION = "predictions"
MONITORING_COLLECTION = "monitoring_logs"

# --- AQI Thresholds ---
AQI_THRESHOLDS = {
    "good": (0, 50),
    "moderate": (51, 100),
    "unhealthy_sensitive": (101, 150),
    "unhealthy": (151, 200),
    "very_unhealthy": (201, 300),
    "hazardous": (301, 500)
}
AQI_ALERT_THRESHOLD = int(os.getenv("AQI_ALERT_THRESHOLD", 150))

# --- AQI Colors (for dashboard) ---
AQI_COLORS = {
    "good": "#00e400",
    "moderate": "#ffff00",
    "unhealthy_sensitive": "#ff7e00",
    "unhealthy": "#ff0000",
    "very_unhealthy": "#8f3f97",
    "hazardous": "#7e0023"
}

# --- Model Settings ---
FORECAST_HOURS = list(range(1, 73))  # 1 to 72 hours
LOOKBACK_HOURS = 24                   # how many past hours to use as features
TRAINING_DAYS = 90                    # how many days of history to train on
MODEL_VERSION = "v1"

# --- Flask ---
FLASK_PORT = int(os.getenv("PORT", os.getenv("FLASK_PORT", 5000)))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# --- Feature Engineering ---
LAG_HOURS = [1, 2, 3, 6, 12, 24]    # lag features to compute
ROLLING_WINDOWS = [3, 6, 12, 24]     # rolling average windows

# --- Monitoring ---
DRIFT_THRESHOLD = 0.15                # 15% drift triggers alert
MONITORING_WINDOW_DAYS = 7           # look at last 7 days for drift