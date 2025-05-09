"""
Script to load best model from MLflow and predict next-day rides
"""

"""
Load best model from MLflow and predict next-day rides
"""

import hopsworks
import mlflow
import pandas as pd
import numpy as np
from datetime import timedelta
from mlflow.tracking import MlflowClient

if __name__ == "__main__":
    # 1. Feature data
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group("citibike_daily_rides", version=1)
    df = fg.read()

    # 2. Pivot + last 28 days
    daily = df.pivot(index="date", columns="start_station_name", values="ride_count").fillna(0)
    daily.index = pd.to_datetime(daily.index)
    last28 = daily.tail(28)
    if len(last28) < 28:
        raise RuntimeError("Need 28 days of data for inference")

    # 3. Build lag features
    next_day = last28.index[-1] + timedelta(days=1)
    tmp = last28.copy()
    tmp.loc[next_day] = np.nan
    lag_dict = {
        f"{c}_lag{l}": tmp[c].shift(l)
        for c in daily.columns for l in range(1, 29)
    }
    Xp = pd.DataFrame(lag_dict).iloc[[-1]]

    # 4. MLflow client
    mlflow.set_tracking_uri("https://c.app.hopsworks.ai:443/hopsworks-api/api/project/1215708/mlflow")
    client = MlflowClient()
    exp = client.get_experiment_by_name("citibike_forecasting")
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
    top_run = next(r for r in runs if r.data.params.get("model_type")=="lightgbm_top10")
    model = mlflow.sklearn.load_model(f"runs:/{top_run.info.run_id}/model")

    # 5. Predict
    input_feats = model.metadata.get_input_schema().input_names()
    preds = model.predict(Xp[input_feats])
    forecast = pd.Series(preds, index=daily.columns, name=next_day.strftime("%Y-%m-%d"))
    print("\n📈 Forecast for next day:")
    print(forecast.round(1))

    # 6. Save
    out = f"predictions_{next_day.strftime('%Y%m%d')}.csv"
    forecast.to_frame("predicted_rides").to_csv(out)
    print(f"\n✅ Saved forecast to {out}")

