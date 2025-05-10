import os
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import plotly.express as px
from datetime import datetime, timedelta

# Disable Hopsworks "model serving" client
os.environ["HOPSWORKS_DISABLE_SERVING"] = "true"
import hopsworks

# Set page configuration
st.set_page_config(page_title="Citi Bike Monitoring Dashboard", layout="wide")
st.title("📊 Model Monitoring: Citi Bike Prediction")

# Set MLflow credentials
mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])
os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]

# Function to clean station names for model names
def clean_model_name(station):
    return station.replace(" ", "_").replace("&", "").replace("/", "_")

# Sidebar navigation
page = st.sidebar.radio("Choose a view:", ["Model MAE Comparison", "Prediction Viewer"])

try:
    # Connect to Hopsworks and load data
    @st.cache_resource
    def get_hopsworks_connection():
        project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
        fs = project.get_feature_store()
        return project, fs

    @st.cache_data(ttl=3600)
    def load_feature_data(_fs):
        fg = _fs.get_feature_group("citibike_daily_rides", version=1)
        df = fg.read()
        return df
    
    project, fs = get_hopsworks_connection()
    df = load_feature_data(fs)
    
    # Pivot the data
    daily = (
        df
        .pivot(index="date", columns="start_station_name", values="ride_count")
        .fillna(0)
    )
    daily.index = pd.to_datetime(daily.index)
    
    # Get list of all stations
    all_stations = sorted(daily.columns.tolist())

    if page == "Model MAE Comparison":
        st.subheader("🔢 MAE Across Trained Models")

        # Load run data from MLflow
        runs_df = mlflow.search_runs(experiment_ids=["0"], filter_string="", output_format="pandas")
        
        if runs_df.empty:
            st.warning("No runs found in MLflow. Have you trained any models yet?")
        else:
            # Get columns for display
            st.write("Available run columns:", [col for col in runs_df.columns if col.startswith("metrics.") or col.startswith("tags.")])
            
            # Extract metrics for each station
            mae_cols = [col for col in runs_df.columns if col.startswith("metrics.mae_")]
            
            if mae_cols:
                # Create a dataframe with model info and metrics
                metrics_data = []
                
                for _, run in runs_df.iterrows():
                    run_name = run.get("tags.mlflow.runName", "Unknown Run")
                    model_type = run.get("params.model_type", "Unknown Type")
                    
                    for col in mae_cols:
                        if pd.notna(run[col]):
                            # Extract station name from metric name 
                            station_name = col.replace("metrics.mae_", "").replace("_", " ")
                            
                            metrics_data.append({
                                "Model": run_name,
                                "Type": model_type,
                                "Station": station_name,
                                "MAE": run[col]
                            })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df = metrics_df.sort_values("MAE")
                    
                    # Show metrics table
                    st.dataframe(metrics_df.reset_index(drop=True))
                    
                    # Plot MAE by model type
                    fig = px.bar(
                        metrics_df, 
                        x="Model", 
                        y="MAE", 
                        color="Type",
                        title="MAE by Model",
                        barmode="group"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot MAE by station
                    fig = px.bar(
                        metrics_df,
                        x="Station",
                        y="MAE",
                        color="Type",
                        title="MAE by Station",
                        barmode="group"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No MAE metrics found in the runs.")
            else:
                st.warning("No MAE metrics found in the MLflow runs.")

    elif page == "Prediction Viewer":
        st.subheader("📊 Predicted Ride Counts")

        # First try to load predictions from file
        try:
            pred_df = pd.read_csv("./data/predictions/predictions.csv", parse_dates=["datetime"])
            file_predictions = True
            
            # Get station identifier
            station_id_col = "start_station_id" if "start_station_id" in pred_df.columns else "station"
            pred_df[station_id_col] = pred_df[station_id_col].astype(str)

            # Get prediction column
            pred_col = "predicted_ride_count" if "predicted_ride_count" in pred_df.columns else "prediction"
            
            # Display station options
            station_options = pred_df[station_id_col].unique()
            station = st.selectbox("Select a Station:", sorted(station_options))

            # Display the predictions as a line chart
            st.line_chart(
                pred_df[pred_df[station_id_col] == station][[pred_df.columns[0], pred_col]].set_index(pred_df.columns[0])
            )
            
        except FileNotFoundError:
            file_predictions = False
            st.warning("Prediction file not found. Showing historical data and 7-day average prediction.")
            
            # If no prediction file, show historical data with 7-day average
            selected_station = st.selectbox("Select a station:", all_stations)
            
            # Get data for the last 30 days
            end_date = daily.index.max()
            start_date = end_date - timedelta(days=30)
            
            station_data = daily.loc[daily.index >= start_date, selected_station]
            
            # Create a simple 7-day moving average prediction
            rolling_avg = station_data.rolling(window=7).mean()
            
            # Create a DataFrame for display
            forecast_df = pd.DataFrame({
                "Date": station_data.index,
                "Actual": station_data.values,
                "7-Day Average": rolling_avg.values
            })
            
            # Plot the data
            fig = px.line(
                forecast_df,
                x="Date",
                y=["Actual", "7-Day Average"],
                title=f"Historical Rides and 7-Day Average for {selected_station}",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download as CSV
            st.download_button(
                "Download Data as CSV",
                forecast_df.to_csv(index=False),
                file_name=f"station_data_{selected_station.replace(' ', '_')}.csv",
                mime="text/csv"
            )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check your Hopsworks and MLflow credentials.")