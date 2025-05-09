# src/inference.py

"""
Load best model from Hopsworks Model Registry via MLflow and predict next‑day rides.
"""

import os
import sys
import pandas as pd
import numpy as np
import hopsworks
import mlflow
from datetime import timedelta
from mlflow.tracking import MlflowClient

def main():
    # 1️⃣ Load Hopsworks API key
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the HOPSWORKS_API_KEY environment variable")

    # 2️⃣ Connect to Hopsworks & read feature group
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("citibike_daily_rides", version=1)
    df = fg.read()

    # 3️⃣ Pivot + last 28 days
    daily = (
        df
        .pivot(index="date", columns="start_station_name", values="ride_count")
        .fillna(0)
    )
    daily.index = pd.to_datetime(daily.index)
    last28 = daily.tail(28)
    if len(last28) < 28:
        raise RuntimeError("Need at least 28 days of data for inference")

    # 4️⃣ Build lag features for next day
    next_day = last28.index[-1] + timedelta(days=1)
    tmp = last28.copy()
    tmp.loc[next_day] = np.nan
    lag_dict = {
        f"{c}_lag{l}": tmp[c].shift(l).astype("float32")
        for c in daily.columns for l in range(1, 29)
    }
    Xp = pd.DataFrame(lag_dict).iloc[[-1]].astype("float32")

    # 5️⃣ Configure MLflow tracking with Basic Auth
    # Security fix: Don't expose API key in URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI") 
    if not mlflow_uri: 
        raise RuntimeError("MLFLOW_TRACKING_URI environment variable not set")

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    # 6️⃣ Load the latest top‑10 model from the Model Registry
    #    (Assumes you registered under name "citibike_best_model")
    model_uri = f"models:/citibike_best_model/1"
    model = mlflow.pyfunc.load_model(model_uri)

    # 7️⃣ Predict
    preds = model.predict(Xp)
    if preds.shape != (1,):
        raise ValueError(f"Expected prediction shape (1,), got {preds.shape}")
    print(next_day)
    forecast = pd.Series([preds[0]], index=["total_bikes"], name=next_day.strftime("%Y-%m-%d"))
    print("\n📈 Forecast for next day (total bikes):")
    print(forecast.round(1))

    # 8️⃣ Save locally
    out = f"predictions_{next_day.strftime('%Y%m%d')}.csv"
    forecast.to_frame("predicted_rides").to_csv(out)
    print(f"\n✅ Saved forecast to {out}")

if __name__ == "__main__":
    main()
