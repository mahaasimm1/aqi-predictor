# AQI Predictor

An end-to-end serverless Air Quality Index (AQI) forecasting system for Karachi, Pakistan. The system predicts AQI for the next 3 days at hourly resolution using machine learning, with real-time data ingestion, a cloud feature store, and an interactive dashboard.

**Live Dashboard:** https://aqi-predictor-7xvbhqndwxemfcrmqcqz44.streamlit.app/

---

## What it does

- Fetches hourly pollutant and weather data from the Open-Meteo API
- Engineers and stores time-series features in MongoDB Atlas
- Trains five ML models and selects the best using TimeSeriesCV
- Serves 72-hour iterative forecasts (1-hour-ahead model called 72 times) via a Flask API
- Displays current AQI, 3-day forecast, pollutant breakdown, SHAP importance, and 7-day history on a Streamlit dashboard

---

## Model Performance

| Model | Holdout RMSE | Holdout R² | CV RMSE | CV R² |
|---|---|---|---|---|
| Ridge | 12.99 | 0.404 | — | — |
| Random Forest | 8.30 | 0.757 | 7.28 | 0.832 |
| Extra Trees | 8.09 | 0.769 | 7.19 | 0.838 |
| **XGBoost (winner)** | **8.64** | **0.737** | **6.06** | **0.876** |
| Gradient Boosting | 8.44 | 0.749 | 6.35 | 0.868 |
| Stacking | 8.70 | 0.733 | 6.87 | 0.849 |

CV = TimeSeriesCV with 5 chronological folds. XGBoost selected as best model.

---

## Project Structure

```
aqi-predictor/
├── pipelines/
│   ├── backfill.py           # Seeds 90 days of historical data into MongoDB
│   └── feature_pipeline.py   # Runs hourly via GitHub Actions, appends latest record
├── models/
│   └── model_utils.py        # Inference: 72-step forecast, SHAP, AQI helpers
├── pipelines/
│   └── training_pipeline.py  # Trains 5 models, TimeSeriesCV, saves best to MongoDB
├── backend/
│   ├── app.py                # Flask app, loads model from MongoDB at startup
│   └── routes/
│       └── predict.py        # /predict endpoint — runs 72-step forecast
├── frontend/
│   └── dashboard.py          # Streamlit dashboard
├── notebooks/
│   ├── eda.ipynb             # Exploratory data analysis (13 cells, all outputs)
│   └── shap_analysis.ipynb   # SHAP interpretability analysis (8 plots)
├── config/
│   └── settings.py           # All config: MongoDB URI, API URLs, feature lists
├── monitoring/
│   └── monitor.py            # Drift detection, logs alerts to MongoDB
└── .github/
    └── workflows/            # GitHub Actions: hourly feature pipeline, weekly monitor
```

---

## Architecture

```
Open-Meteo API
      |
      v
feature_pipeline.py (GitHub Actions, hourly)
      |
      v
MongoDB Atlas — aqi_features collection (Feature Store)
      |
      v
training_pipeline.py (Github Actions, daily)
      |
      v
MongoDB Atlas — models collection + GridFS (Model Registry)
      |
      v
Flask API (Render) ←——— Streamlit Dashboard (Streamlit Cloud)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Data source | Open-Meteo Air Quality + Archive APIs |
| Feature store | MongoDB Atlas (`aqi_features` collection) |
| Model registry | MongoDB Atlas + GridFS (`models` collection) |
| ML models | scikit-learn, XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Backend API | Flask |
| Frontend | Streamlit |
| Pipeline orchestration | GitHub Actions (hourly cron) |
| API deployment | Render |
| Dashboard deployment | Streamlit Community Cloud |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/aqi-predictor.git
cd aqi-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the root directory:

```
MONGO_URI=mongodb+srv://your-connection-string
MONGO_DB_NAME=aqi_predictor
```

### 4. Run the backfill (first time only)

Seeds the last 90 days of data into MongoDB:

```bash
python pipelines/backfill.py
```

### 5. Train the model

```bash
python pipelines/training_pipeline.py
```

This trains 5 models with TimeSeriesCV hyperparameter search, selects the best by CV RMSE, and saves the artifact to MongoDB GridFS.

### 6. Run the backend

```bash
python backend/app.py
```

API available at `http://localhost:5001/predict`

### 7. Run the dashboard

```bash
streamlit run frontend/dashboard.py
```

---

## Automated Pipeline

The feature pipeline runs automatically every hour via GitHub Actions. Set the following secrets in your GitHub repository settings:

- `MONGO_URI` — your MongoDB Atlas connection string
- `MONGO_DB_NAME` — database name (default: `aqi_predictor`)

The workflow file is at `.github/workflows/feature_pipeline.yml`.

---

## Notebooks

**EDA** (`notebooks/eda.ipynb`) — Data quality checks, AQI distribution, temporal patterns, feature correlations, pollutant distributions, lag feature analysis, weather analysis, interaction feature validation, and a full summary of findings used to drive feature engineering decisions.

**SHAP Analysis** (`notebooks/shap_analysis.ipynb`) — Global feature importance, beeswarm summary plot, dependency plots for top features, waterfall explanations for individual predictions, time-of-day SHAP patterns, and feature group breakdown.
