import os
import streamlit as st
import pandas as pd
import hopsworks
import joblib
import json

from utils import clean_model_name, make_lags
from datetime import datetime

st.set_page_config(page_title="Citi Bike Forecast", page_icon="🚲")
st.title("🚲 Citi Bike Forecast")

# ✅ Read API key
api_key = os.environ.get("HOPSWORKS_API_KEY")
if not api_key:
    st.error("HOPSWORKS_API_KEY not set.")
    st.stop()

# ✅ Connect to Hopsworks
project = hopsworks.login(api_key_value=api_key, project="gourimenon8")
fs = project.get_feature_store()

# ✅ Read feature data
fg = fs.get_feature_group(name="citibike_aggregates", version=1)
df = fg.read()

# ✅ Lag features
pivot = df.pivot(index="date", columns="station_name", values="rides")
X_all, _ = make_lags(pivot, n_lags=28)
X_pred = X_all.tail(1)

model_registry = project.get_model_registry()
stations = list(pivot.columns)

for station in stations:
    cleaned = clean_model_name(station)
    st.subheader(f"📍 {station}")
    try:
        model = model_registry.get_model(f"citibike_model_{cleaned}", version=1)
        dir = model.download()
        
        with open(os.path.join(dir, "top10_features.json")) as f:
            top10 = json.load(f)

        lgb_model = joblib.load(os.path.join(dir, "model.pkl"))
        pred = lgb_model.predict(X_pred[top10])[0]

        st.metric("Predicted Rides Tomorrow", f"{int(pred)}")
    except Exception as e:
        st.error(f"❌ Could not load model for {station}: {e}")
