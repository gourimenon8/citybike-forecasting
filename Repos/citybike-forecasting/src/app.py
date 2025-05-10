import os
import json
import streamlit as st
import pandas as pd
import mlflow
import hopsworks

from utils import clean_model_name, make_lags
from sklearn.metrics import mean_absolute_error

st.title("🚲 Forecasting Citi Bike Rides")
st.markdown("Predicting tomorrow's ride counts for NYC's busiest stations.")

# Load Hopsworks feature data
st.write("🔌 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()
fg = fs.get_feature_group("citibike_daily_rides", version=1)
df = fg.read()
df["date"] = pd.to_datetime(df["date"])

# Pivot to wide format and create lag features
pivot = df.pivot(index="date", columns="start_station_name", values="ride_count").fillna(0)
X_all, _ = make_lags(pivot, n_lags=28)
X_latest = X_all.iloc[[-1]]  # Last row (latest day)

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.tracking.MlflowClient()

# Predict for each top station
top_stations = [
    "1 Ave & E 68 St",
    "8 Ave & W 31 St",
    "Broadway & W 25 St",
    "University Pl & E 14 St",
    "W 21 St & 6 Ave",
]

for station in top_stations:
    try:
        model_name = f"citibike_model_{clean_model_name(station)}"
        st.write(f"🔍 Loading model: `{model_name}`")

        # Get latest version
        latest = client.get_latest_versions(model_name, stages=["None"])[0]

        # Load model
        model = mlflow.pyfunc.load_model(latest.source)

        # Load top-10 features
        top10_path = os.path.join(latest.source, "top10_features.json")
        with open(top10_path, "r") as f:
            top10 = json.load(f)

        # Prepare input
        input_data = X_latest[top10]

        # Predict
        pred = model.predict(input_data)[0]
        st.metric(label=f"📈 Prediction for '{station}'", value=int(round(pred)))

    except Exception as e:
        st.warning(f"⚠️ Could not predict for '{station}': {e}")

st.success("✅ Forecast complete.")
