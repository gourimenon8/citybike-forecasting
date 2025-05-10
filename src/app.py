# src/app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
from datetime import timedelta

# —————————————————————————————————————————————
# 1) disable Hopsworks “model serving” client so that hopsworks.login()
#    won’t try to initialize kServe, which fails outside their cloud
os.environ["HOPSWORKS_DISABLE_SERVING"] = "true"

import hopsworks

# —————————————————————————————————————————————
st.title("Citi Bike Ride Forecasting")

# 2) Log in & fetch your feature group
st.info("🔌 Connecting to Hopsworks…")
project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()
fg = fs.get_feature_group("citibike_daily_rides", version=1)
df = fg.read()
st.success("✅ Feature data loaded")

# 3) pivot into wide form & take last 28 days
daily = (
    df
    .pivot(index="date", columns="start_station_name", values="ride_count")
    .fillna(0)
)
daily.index = pd.to_datetime(daily.index)
last28 = daily.tail(28)

# 4) build tomorrow’s lag features
next_day = last28.index[-1] + timedelta(days=1)
tmp = last28.copy()
tmp.loc[next_day] = np.nan
# each column → 28 lag columns
Xp = pd.concat(
    {
        f"{station}_lag{lag}": tmp[station].shift(lag).astype("float32")
        for station in daily.columns
        for lag in range(1, 29)
    },
    axis=1,
).iloc[[-1]].astype("float32")

# 5) loop through each station's model
st.info("⏳ Loading models & generating predictions…")
results = []
for station in daily.columns:
    clean = station.replace(" ", "_").replace("&", "").replace("/", "_")
    model_name = f"citibike_model_{clean}"
    st.write(f"🔍 Loading model `{model_name}`")
    try:
        # get latest registered version
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name)
        uri = versions[0].source  # e.g. "models:/citibike_model_X/3"
        model = mlflow.pyfunc.load_model(uri)
        pred = model.predict(Xp)[0]
        results.append({"station": station, "prediction": round(pred)})
        st.success(f"📈 {station}: {pred:.0f} rides tomorrow")
    except Exception as e:
        st.error(f"⚠️ {station}: could not load/predict ({e})")

if results:
    st.subheader("Summary Table")
    st.table(pd.DataFrame(results).set_index("station"))
else:
    st.error("❌ No predictions could be made.")
