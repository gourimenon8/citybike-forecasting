# src/inference.py

"""
Load best model from Hopsworks Model Registry via MLflow and predict next‑day rides.
"""

# src/inference.py

import os
import pandas as pd
import numpy as np
import hopsworks
import mlflow
from datetime import timedelta
import re

def clean_fn(name: str) -> str:
    """Sanitize station name to create a valid MLflow model name."""
    return re.sub(r"[^\w]+", "_", name)

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

    # 3️⃣ Pivot data and get last 28 days
    daily = (
        df.pivot(index="date", columns="start_station_name", values="ride_count")
          .fillna(0)
    )
    daily.index = pd.to_datetime(daily.index)
    last28 = daily.tail(28)
    if len(last28) < 28:
        raise RuntimeError("Need at least 28 days of data for inference")

    # 4️⃣ Build lag features for the next day
    next_day = last28.index[-1] + timedelta(days=1)
    tmp = last28.copy()
    tmp.loc[next_day] = np.nan
    lag_dict = {
        f"{c}_lag{l}": tmp[c].shift(l).astype("float32")
        for c in daily.columns for l in range(1, 29)
    }
    Xp = pd.DataFrame(lag_dict).iloc[[-1]].astype("float32")

    # 5️⃣ Configure MLflow tracking
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI environment variable not set")
    mlflow.set_tracking_uri(mlflow_uri)

    # 6️⃣ Inference: load and predict for each station
    forecast_values = {}
    station_names = Xp.columns.str.extract(r"(.+)_lag\d+")[0].unique()

    for station in station_names:
        cleaned_name = clean_fn(station)
        model_name = f"citibike_model_{cleaned_name}"
        try:
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/1")
            station_lags = [col for col in Xp.columns if col.startswith(f"{station}_lag")]
            pred = model.predict(Xp[station_lags])[0]
            forecast_values[station] = pred
            print(f"✅ Predicted {pred:.1f} rides for '{station}'")
        except Exception as e:
            forecast_values[station] = None
            print(f"⚠️ Could not load or predict for station '{station}': {e}")

    # 7️⃣ Print and save forecast
    forecast = pd.Series(forecast_values, name=next_day.strftime("%Y-%m-%d"))
    forecast = forecast.dropna()  # drop failed stations
    print("\n📈 Forecast for next day:")
    print(forecast.round(1))

    # 8️⃣ Save to CSV
    out = f"predictions_{next_day.strftime('%Y%m%d')}.csv"
    forecast.to_frame("predicted_rides").to_csv(out)
    print(f"\n✅ Saved forecast to {out}")

if __name__ == "__main__":
    main()
