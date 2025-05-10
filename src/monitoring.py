import os
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Disable Hopsworks model serving client
os.environ["HOPSWORKS_DISABLE_SERVING"] = "true"
import hopsworks

# Page configuration
st.set_page_config(
    page_title="Citi Bike Monitoring Dashboard",
    page_icon="🚲",
    layout="wide"
)

# Set MLflow tracking URI from secrets
mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])
os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]

# Connect to Hopsworks
@st.cache_resource
def get_hopsworks_connection():
    st.info("🔌 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    return project, fs

# Cache the feature data loading
@st.cache_data(ttl=3600)
def load_feature_data(fs):
    fg = fs.get_feature_group("citibike_daily_rides", version=1)
    df = fg.read()
    return df

# Clean station names for model retrieval
def clean_model_name(station):
    return station.replace(" ", "_").replace("&", "").replace("/", "_")

# Title and description
st.title("🚲 Citi Bike Model Monitoring Dashboard")
st.markdown("Track model performance and predictions for the Citi Bike forecasting system")

# Sidebar navigation
page = st.sidebar.radio("Choose a view:", ["Model Performance", "Station Predictions", "Feature Importance"])

try:
    # Connect to services
    project, fs = get_hopsworks_connection()
    df = load_feature_data(fs)
    
    # Pivot the data
    daily = (
        df
        .pivot(index="date", columns="start_station_name", values="ride_count")
        .fillna(0)
    )
    daily.index = pd.to_datetime(daily.index)
    
    # Get list of available stations
    all_stations = sorted(daily.columns.tolist())
    
    if page == "Model Performance":
        st.header("📊 Model Performance Metrics")
        
        # Get experiments from MLflow
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        exp_names = [exp.name for exp in experiments]
        
        selected_exp = st.selectbox("Select Experiment:", exp_names)
        exp = client.get_experiment_by_name(selected_exp)
        
        if exp:
            # Get runs from selected experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            
            if not runs.empty:
                # Clean up column names for display
                metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
                param_cols = [col for col in runs.columns if col.startswith('params.')]
                
                # Create tabs for different views
                tabs = st.tabs(["Metrics", "Parameters", "Run Details"])
                
                with tabs[0]:
                    st.subheader("Model Metrics")
                    
                    # Extract and display metrics
                    if metric_cols:
                        metrics_df = runs[['run_id', 'tags.mlflow.runName'] + metric_cols].copy()
                        metrics_df.columns = [col.replace('metrics.', '') if col.startswith('metrics.') else col for col in metrics_df.columns]
                        metrics_df.rename(columns={'tags.mlflow.runName': 'Run Name'}, inplace=True)
                        
                        st.dataframe(metrics_df.set_index('Run Name').drop('run_id', axis=1))
                        
                        # Plot MAE by model type
                        if 'mae' in metrics_df.columns:
                            fig = px.bar(
                                metrics_df, 
                                x='Run Name', 
                                y='mae', 
                                title="Mean Absolute Error by Model",
                                labels={'mae': 'MAE', 'Run Name': 'Model'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No metrics found for this experiment")
                
                with tabs[1]:
                    st.subheader("Model Parameters")
                    
                    if param_cols:
                        params_df = runs[['tags.mlflow.runName'] + param_cols].copy()
                        params_df.columns = [col.replace('params.', '') if col.startswith('params.') else col for col in params_df.columns]
                        params_df.rename(columns={'tags.mlflow.runName': 'Run Name'}, inplace=True)
                        
                        st.dataframe(params_df.set_index('Run Name'))
                    else:
                        st.info("No parameters found for this experiment")
                
                with tabs[2]:
                    st.subheader("Run Details")
                    st.dataframe(runs[['run_id', 'tags.mlflow.runName', 'start_time', 'end_time', 'status']])
            else:
                st.warning("No runs found for this experiment")
        else:
            st.warning("No experiment found with this name")
            
    elif page == "Station Predictions":
        st.header("🔮 Prediction Performance by Station")
        
        # Select stations to analyze
        selected_stations = st.multiselect(
            "Select stations to analyze:",
            options=all_stations,
            default=all_stations[:3] if len(all_stations) >= 3 else all_stations
        )
        
        if not selected_stations:
            st.warning("Please select at least one station")
        else:
            # Date range for analysis
            date_range = st.slider(
                "Select date range:",
                min_value=daily.index.min().to_pydatetime(),
                max_value=daily.index.max().to_pydatetime(),
                value=(daily.index.max() - timedelta(days=30), daily.index.max().to_pydatetime())
            )
            
            # Filter data based on selections
            mask = (daily.index >= date_range[0]) & (daily.index <= date_range[1])
            filtered_data = daily.loc[mask, selected_stations]
            
            # Get prediction results if available
            st.subheader("Historical Rides vs Recent Predictions")
            
            # Load models and generate recent predictions
            with st.spinner("Loading models and generating predictions..."):
                client = mlflow.tracking.MlflowClient()
                
                prediction_results = []