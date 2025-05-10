# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import mlflow
from datetime import timedelta

st.set_page_config(page_title="Citi Bike Forecast", layout="centered")
st.title("🚲 Citi Bike Ride Forecasting App")
st.markdown("Tomorrow's predicted rides per station will be displayed here.")

try:
    st.subheader("🔗 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_feature_group("citibike_daily_rides", version=1)
    df = fg.read()
    st.success(f"Loaded {df.shape[0]} rows of feature data.")

    # Pivot data
    daily = df.pivot(index="date", columns="start_station_name", values="ride_count").fillna(0)
    daily.index = pd.to_datetime(daily.index)
    last28 = daily.tail(28)
    st.write("Last 28 days shape:", last28.shape)

    if len(last28) < 28:
        st.warning("Not enough data for inference (need 28 days).")
    else:
        next_day = last28.index[-1] + timedelta(days=1)
        tmp = last28.copy()
        tmp.loc[next_day] = np.nan

        lag_dict = {
            f"{c}_lag{l}": tmp[c].shift(l).astype("float32")
            for c in daily.columns for l in range(1, 29)
        }
        Xp = pd.DataFrame(lag_dict).iloc[[-1]].astype("float32")
        st.write("Input shape for model:", Xp.shape)

        # Load model
        mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])
        model = mlflow.pyfunc.load_model("models:/citibike_best_model/1")
        preds = model.predict(Xp)
        st.success(f"Prediction complete: {preds[0]:.1f} rides")
        st.metric(label=f"Prediction for {next_day.strftime('%Y-%m-%d')}", value=f"{preds[0]:.1f}")
        st.bar_chart(pd.Series([preds[0]], index=[next_day.strftime('%Y-%m-%d')]))

except Exception as e:
    st.error(f"❌ An error occurred: {e}")
