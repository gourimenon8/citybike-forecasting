# app.py

import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import mlflow
from datetime import timedelta

# Page setup
st.set_page_config(page_title="Citi Bike Forecast", layout="centered")
st.title("🚲 Citi Bike Ride Forecasting App")
st.markdown("Tomorrow's predicted rides for NYC's busiest Citi Bike stations.")

# Start app
try:
    st.info("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    fg = fs.get_feature_group("citibike_daily_rides", version=1)
    df = fg.read()
    st.success("✅ Feature data loaded from Hopsworks")

    # Prepare time series input
    st.info("Processing input data for inference...")
    daily = (
        df.pivot(index="date", columns="start_station_name", values="ride_count")
          .fillna(0)
    )
    daily.index = pd.to_datetime(daily.index)
    last28 = daily.tail(28)

    if len(last28) < 28:
        st.warning("🚧 Not enough data for inference (need at least 28 days).")
    else:
        next_day = last28.index[-1] + timedelta(days=1)
        tmp = last28.copy()
        tmp.loc[next_day] = np.nan
        lag_dict = {
            f"{c}_lag{l}": tmp[c].shift(l).astype("float32")
            for c in daily.columns for l in range(1, 29)
        }
        Xp = pd.DataFrame(lag_dict).iloc[[-1]].astype("float32")
        st.success(f"✅ Prepared lag features for {next_day.date()}")

        # Load model from MLflow
        st.info("Loading model from MLflow registry...")
        mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])
        model = mlflow.pyfunc.load_model("models:/citibike_best_model/1")
        preds = model.predict(Xp)

        st.success("✅ Prediction complete")
        forecast = pd.Series(preds, index=daily.columns)
        forecast_df = forecast.to_frame(name="Predicted Rides").T
        forecast_df.index = [next_day.strftime("%Y-%m-%d")]

        # Display forecast
        st.subheader("📈 Forecast for Next Day")
        st.dataframe(forecast_df.style.format("{:.0f}"))
        st.bar_chart(forecast)

except Exception as e:
    st.error(f"❌ Error: {e}")
