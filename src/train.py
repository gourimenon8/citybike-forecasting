# src/train.py

"""
Driver script to train baseline, full‑lag, and top‑10 LightGBM models
with MLflow logging & registration against Hopsworks.
"""

import os
import sys
import pandas as pd
import numpy as np
import hopsworks
import mlflow

from sklearn.model_selection import TimeSeriesSplit
from utils import clean_metric_name
from modeling import make_lags, train_baseline, train_full_lgbm, train_top10_lgbm

def main():
    # 1️⃣ Load Hopsworks API key
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the HOPSWORKS_API_KEY environment variable")

    # 2️⃣ Connect to Hopsworks & read feature group
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("citibike_daily_rides", version=1)
    agg_df = fg.read()

    # 3️⃣ Pivot to wide format, fill missing
    daily = (
        agg_df
        .pivot(index="date", columns="start_station_name", values="ride_count")
        .fillna(0)
    )
    daily.index = pd.to_datetime(daily.index)

    # 4️⃣ Create lag features (28 days)
    X, y = make_lags(daily, n_lags=28)

    # 5️⃣ Train/test split (last fold)
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 6️⃣ Configure MLflow tracking with Basic Auth
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI") 
    if not mlflow_uri: 
        raise RuntimeError("MLFLOW_TRACKING_URI environment variable not set")
    mlflow.set_tracking_uri(mlflow_uri)

    # 7️⃣ Baseline model
    print("▶️  Training baseline model")
    train_baseline(y_train, y_test, clean_metric_name)

    # 8️⃣ Full‑lag LightGBM models
    print("▶️  Training full‑lag LightGBM models")
    full_models = train_full_lgbm(X_train, X_test, y_train, y_test, clean_metric_name)

    # 9️⃣ Top‑10 LightGBM models + register best
    print("▶️  Training top‑10 LightGBM models + registering best")
    train_top10_lgbm(full_models, X_train, X_test, y_train, y_test, clean_metric_name)

    print("✅ All models trained, logged, and registered.")

if __name__ == "__main__":
    main()
