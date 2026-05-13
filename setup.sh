#!/bin/bash

# Create all directories
mkdir -p .github/workflows
mkdir -p pipelines
mkdir -p backend/routes
mkdir -p backend/services
mkdir -p frontend/components
mkdir -p models
mkdir -p monitoring
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p tests
mkdir -p config
mkdir -p scripts
mkdir -p docs

# GitHub files
touch .github/workflows/feature_pipeline.yml
touch .github/workflows/training_pipeline.yml
touch .github/workflows/monitoring.yml
touch .github/PULL_REQUEST_TEMPLATE.md

# Pipeline files
touch pipelines/__init__.py
touch pipelines/feature_pipeline.py
touch pipelines/backfill.py
touch pipelines/training_pipeline.py

# Backend files
touch backend/__init__.py
touch backend/app.py
touch backend/routes/__init__.py
touch backend/routes/predict.py
touch backend/routes/history.py
touch backend/routes/health.py
touch backend/services/__init__.py
touch backend/services/model_service.py
touch backend/services/feature_service.py

# Frontend files
touch frontend/dashboard.py
touch frontend/components/__init__.py
touch frontend/components/aqi_gauge.py
touch frontend/components/forecast_chart.py
touch frontend/components/pollutant_chart.py
touch frontend/components/shap_plot.py
touch frontend/components/alert_banner.py
touch frontend/components/history_chart.py

# Model files
touch models/__init__.py
touch models/model_utils.py
touch models/ridge_model.py
touch models/random_forest_model.py
touch models/lstm_model.py

# Monitoring files
touch monitoring/__init__.py
touch monitoring/drift_detector.py
touch monitoring/alert_service.py
touch monitoring/metrics_logger.py

# Notebook files
touch notebooks/eda.ipynb
touch notebooks/model_experiments.ipynb
touch notebooks/shap_analysis.ipynb

# Test files
touch tests/__init__.py
touch tests/test_feature_pipeline.py
touch tests/test_model_utils.py
touch tests/test_routes.py

# Config files
touch config/__init__.py
touch config/settings.py

# Script files
touch scripts/setup_hopsworks.py
touch scripts/run_backfill.sh

# Docs
touch docs/architecture.md
touch docs/api_reference.md
touch docs/report.md

# Root files
touch .env.example
touch requirements.txt
touch Procfile

echo "✅ All files and folders created successfully!"
