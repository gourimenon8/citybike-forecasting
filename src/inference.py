# src/inference.py

"""
Load best model from Hopsworks Model Registry via MLflow and predict next‑day rides.
"""

import os
import json
import hopsworks
import mlflow
import pandas as pd
from utils import make_lags, clean_model_name

# 1️⃣ Connect to Hopsworks
print("🔌 Connecting to Hopsworks...")
api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise RuntimeError("HOPSWORKS_API_KEY not set")

project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

# 2️⃣ Load data
fg = fs.get_feature_group("citibike_daily_rides", version=1)
agg_df = fg.read()
pivot = (
    agg_df
    .pivot(index="date", columns="start_station_name", values="ride_count")
    .fillna(0)
)
pivot.index = pd.to_datetime(pivot.index)

# 3️⃣ Create lag features
X_all, _ = make_lags(pivot, n_lags=28)
X_tomorrow = X_all.iloc[[-1]]  # latest row

# 4️⃣ Load models + predict
registry = project.get_model_registry()
stations = ["1 Ave & E 68 St", "8 Ave & W 31 St", "Broadway & W 25 St", "University Pl & E 14 St", "W 21 St & 6 Ave"]

for station in stations:
    print(f"\n🔍 Loading model: {clean_model_name(station)}")
    try:
        # Get latest version of model
        model_name = f"citibike_model_{clean_model_name(station)}"
        model = registry.get_model(model_name, version=6)
        model_dir = model.download()
        model = mlflow.sklearn.load_model(model_dir)

        # Load top-10 features
        top10_path = os.path.join(model_dir, "top10_features.json")
        if not os.path.exists(top10_path):
            raise FileNotFoundError(f"{top10_path} not found")

        with open(top10_path) as f:
            top10 = json.load(f)

        X_selected = X_tomorrow[top10]
        pred = model.predict(X_selected)[0]
        print(f"📈 Predicted rides tomorrow at '{station}': {pred:.0f}")

    except Exception as e:
        print(f"⚠️ Could not load model or predict for '{station}': {e}")

print("✅ Inference complete.")
