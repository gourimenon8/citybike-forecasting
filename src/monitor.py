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
                for station in selected_stations:
                    # Get station's model
                    clean_name = clean_model_name(station)
                    model_name = f"citibike_model_{clean_name}"
                    
                    try:
                        # Attempt to get latest model version
                        versions = client.get_latest_versions(model_name)
                        if versions:
                            uri = versions[0].source
                            model = mlflow.pyfunc.load_model(uri)
                            
                            # Get the last 28 days for feature creation
                            station_data = filtered_data[station].copy()
                            
                            # Create lag features
                            for i in range(1, min(29, len(station_data))):
                                idx = -i-1
                                if idx >= -len(station_data):
                                    lag_col = f"{station}_lag{i}"
                                    prediction_results.append({
                                        "station": station,
                                        "date": station_data.index[idx+1],
                                        "actual": station_data.iloc[idx+1],
                                        "prediction_type": "Hindcast"
                                    })
                    except Exception as e:
                        st.error(f"Error loading model for {station}: {str(e)}")
            
            # Visualization of results
            if prediction_results:
                results_df = pd.DataFrame(prediction_results)
                
                # Plot predictions vs actuals
                fig = px.line(
                    results_df, 
                    x="date",
                    y=["actual", "prediction"],
                    color="station",
                    title="Actual vs Predicted Rides",
                    labels={"value": "Ride Count", "date": "Date"},
                    line_dash="prediction_type"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Raw data view
                with st.expander("Show Raw Prediction Data"):
                    st.dataframe(results_df)
            
            # Display historical ride patterns
            st.subheader("Historical Ride Patterns")
            fig = px.line(
                filtered_data,
                title="Historical Ride Counts by Station"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(filtered_data.describe())
            
    elif page == "Feature Importance":
        st.header("🔍 Feature Importance Analysis")
        
        # Select a station to analyze
        selected_station = st.selectbox(
            "Select station to analyze:",
            options=all_stations
        )
        
        if selected_station:
            # Get station's model
            clean_name = clean_model_name(selected_station)
            model_name = f"citibike_model_{clean_name}"
            
            try:
                # Connect to MLflow
                client = mlflow.tracking.MlflowClient()
                
                # Get latest model version
                versions = client.get_latest_versions(model_name)
                if versions:
                    # Try to load feature importance from model artifacts
                    run_id = versions[0].run_id
                    
                    # Get run information
                    run_info = client.get_run(run_id)
                    
                    # Check if we have feature importance data
                    artifacts = client.list_artifacts(run_id)
                    if any(artifact.path == "top10_features.json" for artifact in artifacts):
                        # Download top10 features file
                        top10_path = client.download_artifacts(run_id, "top10_features.json")
                        
                        with open(top10_path, 'r') as f:
                            top_features = json.load(f)
                        
                        # Display top features
                        st.subheader(f"Top Features for {selected_station}")
                        
                        # Create a bar chart of feature importance
                        fig = px.bar(
                            x=[f.split('_lag')[1] if '_lag' in f else f for f in top_features],
                            y=[10-i for i in range(len(top_features))],  # Assign importance based on rank
                            labels={"x": "Lag Days", "y": "Importance Score"},
                            title=f"Top Features for {selected_station}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write("Top 10 Features:")
                        st.json(top_features)
                    else:
                        st.info("No feature importance information available for this model")
                else:
                    st.warning(f"No model found for station: {selected_station}")
            except Exception as e:
                st.error(f"Error loading feature importance: {str(e)}")
                
            # Additional analysis: lag correlation heatmap
            st.subheader("Lag Correlation Analysis")
            
            # Get station data
            station_data = daily[selected_station].dropna()
            
            # Create lag features for correlation analysis
            lag_df = pd.DataFrame()
            lag_df["original"] = station_data
            
            for lag in range(1, 15):
                lag_df[f"lag_{lag}"] = station_data.shift(lag)
            
            lag_df = lag_df.dropna()
            
            # Create correlation matrix
            corr_matrix = lag_df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Lag", y="Lag", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                title=f"Correlation Between Different Lag Features for {selected_station}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select a station")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check your Hopsworks and MLflow credentials in the .streamlit/secrets.toml file.")