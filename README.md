
# Citi Bike Forecasting Project

End-to-end time-series pipeline for forecasting NYC Citi Bike hourly ride counts by station/region.
Includes data ingestion → feature engineering → model training/evaluation → batch inference → a Streamlit dashboard.

<p align="center"> <img src="assets/citibike_banner.png" alt="Citi Bike Forecasting" width="800"> </p>

✨ Highlights

Data: pulls official Citi Bike trip data from https://citibikenyc.com/system-data

Features: rich lag features (1, 24, 48, …, 672) + rolling means, trend, DoW/holiday, weather (optional)

Models: baseline (naive/mean), LightGBM (28-day lag), and feature-selected LightGBM (≤10 features)

Metrics: MAE, RMSE, MAPE, per-location and overall

Batch inference: monthly/hourly forecasts written to data/predictions/...

Streamlit app: interactive exploration + MAE/MAPE by location with caching

Optional: MLflow experiment tracking, GitHub Actions for scheduled runs


🧱 1. Project Structure

cciti-bike-forecasting/
├── app/                     # Streamlit app (app.py + components)
├── data/
│   ├── raw/                 # downloaded CSV/Parquet
│   ├── features/            # engineered features
│   └── predictions/         # batch predictions (partitioned)
├── models/                  # saved LightGBM artifacts
├── notebooks/               # EDA and experiments
├── src/
│   ├── ingest.py            # download & local cache
│   ├── transform.py         # clean & feature engineer
│   ├── train.py             # train/evaluate models
│   ├── predict.py           # batch inference
│   ├── metrics.py           # MAE/RMSE/MAPE helpers
│   ├── utils.py             # shared utilities/logging
│   └── config.py            # paths, constants, settings
├── requirements.txt
├── .env.example
└── README.md



🧱 Architecture (at a glance)
            ┌──────────────────────┐
            │  System Data (CSV/ZIP│
            │  https://citibike... │
            └─────────┬────────────┘
                      │  ingest.py
               raw/   ▼
            ┌──────────────────────┐
            │  Bronze: raw files   │
            └─────────┬────────────┘
                      │  transform.py
           features/  ▼
            ┌──────────────────────┐
            │ Silver: cleaned +    │
            │ engineered features   │
            └─────────┬────────────┘
                      │  train.py
             models/  ▼
            ┌──────────────────────┐
            │  LightGBM models     │
            │  + metrics           │
            └─────────┬────────────┘
                      │  predict.py
       predictions/   ▼
            ┌──────────────────────┐
            │  Batch forecasts     │
            │  (year/month/day/hr) │
            └─────────┬────────────┘
                      │  app.py
                      ▼
            ┌──────────────────────┐
            │  Streamlit dashboard │
            └──────────────────────┘



🔧 Setup

Prereqs: Python 3.10+, Git, (optional) make
# clone your repo
git clone https://github.com/<you>/citi-bike-forecasting.git
cd citi-bike-forecasting

# create env + install deps
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env

🗂️ Data

Primary source: Citi Bike System Data → https://citibikenyc.com/system-data

Typical files: monthly trip CSVs (YYYYMM-citibike-tripdata.csv.zip)

Optional: weather (NOAA/Open-Meteo). If you don’t have an API key, keep weather disabled.


🚀 Quickstart (Local)
1) Ingest data

Downloads the months you specify and stores them under data/raw/.
python -m src.ingest --start 2023-01 --end 2024-01
# examples:
# python -m src.ingest --months 2023-01 2023-02 2023-03

2) Transform & feature engineering

Cleans data, aggregates rides to hourly counts per location, and builds lag/rolling features.
python -m src.transform --freq H --top_k_locations 3 --with-weather false
Generated artifacts go to data/features/.

3) Train models

Trains three variants and writes models + metrics to models/ and notebooks/metrics.csv.
# baseline + LightGBM (28-day lag) + LightGBM (≤10 features)
python -m src.train --locations 43 79 162 --horizon_hours 24 --mlflow false
Key features (for the 10-feature variant) include: lag_1, lag_24, lag_48, lag_72, lag_96, same_hour_4wk_avg, dow, hour, rolling_mean_24, rolling_std_24.

4) Batch inference

Writes partitioned predictions to data/predictions/model=<id>/location_id=<id>/year=YYYY/month=MM/....
python -m src.predict --year 2024 --locations 43 79 162 --model lgbm_28lag

5) Launch Streamlit dashboard

Two tabs: Features/Predictions and Metrics, with caching.
streamlit run app/app.py

📊 Evaluation

MAE (Mean Absolute Error)

RMSE

MAPE

Example (after training):
location_id, model, split, MAE, RMSE, MAPE
43, lgbm_28lag, test, 4.87, 6.11, 9.8
79, lgbm_10feat, test, 5.02, 6.45, 10.3
...
The Streamlit app displays per-location metrics and side-by-side comparisons of the two LightGBM variants.

⚙️ Configuration
src/config.py (or .env) controls defaults:
# .env.example
DATA_DIR=data
RAW_DIR=data/raw
FEATURE_DIR=data/features
PRED_DIR=data/predictions
MODELS_DIR=models

TOP_K_LOCATIONS=3
FREQ=H
WITH_WEATHER=false

# MLflow (optional)
MLFLOW_TRACKING_URI=
MLFLOW_EXPERIMENT_NAME=citibike-forecasting

🧪 Useful Commands
# Rebuild features quickly for a shorter slice
python -m src.transform --start 2023-01-01 --end 2023-02-28 --freq H

# Train only for one location
python -m src.train --locations 43

# Predict one month only
python -m src.predict --year 2024 --month 01 --locations 43

# Export metrics table
python -m src.train --export-metrics metrics/metrics.csv

🧰 MLflow (Optional)
Enable experiment tracking:
# .env
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME=citibike-forecasting

Run:
mlflow ui
# in another terminal
python -m src.train --mlflow true
You’ll see runs with MAE reductions vs baseline, plus params & artifacts.

⏰ Scheduling (Optional, GitHub Actions)

Create .github/workflows/pipeline.yml:
name: citibike-forecasting
on:
  schedule:
    - cron: "0 2 * * *"  # daily 02:00 UTC
  workflow_dispatch: {}

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -r requirements.txt
      - run: |
          python -m src.ingest --end now --start 2024-01
          python -m src.transform --freq H --top_k_locations 3
          python -m src.train --locations 43 79 162 --horizon_hours 24
          python -m src.predict --year 2024 --locations 43 79 162 --model lgbm_28lag
      - name: Commit artifacts
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add data/models data/predictions -A || true
          git commit -m "chore: update predictions" || true
          git push || true

🧱 Implementation Notes

Aggregation level: hourly counts per start_station_id (or a chosen mapping to “location”).

Missing hours are forward-filled; outliers can be capped via IQR or quantile winsorization.

Feature set:

Lags: 1, 24, 48, 72, 96, 168, 336, 672

Rolling stats: mean/std over 24/168h

same_hour_4wk_avg (avg of the same hour across the last 4 weeks)

Calendar: hour, dayofweek, weekend, holiday flag

(optional) weather: temp, precip, wind

🧹 Troubleshooting

Slow training: reduce locations (--locations 43) or limit months in ingest.

Streamlit cache stale: the app invalidates cache on hour change; you can force refresh with the sidebar toggle.

Different station IDs: schema can differ across years; transform.py maps/normalizes columns.

