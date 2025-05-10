import os
import json
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
st.set_page_config(
    page_title="Citi Bike Ride Forecast",
    page_icon="🚲"
)

# Set MLflow credentials
mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])
os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]
os.environ["MLFLOW_TRACKING_URI"] = st.secrets["MLFLOW_TRACKING_URI"]

# Page header
st.title("🚲 Citi Bike Station Forecasting")

# Cache the Hopsworks connection
@st.cache_resource
def get_hopsworks_connection():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    return project, fs

# Cache the feature data loading
@st.cache_data(ttl=3600)
def load_feature_data(_fs):
    fg = _fs.get_feature_group("citibike_daily_rides", version=1)
    df = fg.read()
    return df

# Function to clean station names for model names
def clean_model_name(station):
    return station.replace(" ", "_").replace("&", "").replace("/", "_")

# Function to create lag features
def make_lags(df, stations, n_lags=28):
    next_day = df.index.max() + timedelta(days=1)
    tmp = df.copy()
    tmp.loc[next_day] = np.nan
    
    # Create lag features only for selected stations
    X = pd.concat({
        f"{station}_lag{lag}": tmp[station].shift(lag).astype("float32")
        for station in stations
        for lag in range(1, n_lags + 1)
    }, axis=1)
    
    return X.iloc[[-1]].astype("float32")

# Use model registry directly from Hopsworks instead of MLflow
@st.cache_resource
def get_model_registry():
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    return project.get_model_registry()

# Check if local model exists and load it
def load_local_model(station):
    clean_name = clean_model_name(station)
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            f"citibike_model_{clean_name}")
    
    if os.path.exists(model_dir):
        st.info(f"Loading local model for {station}...")
        try:
            # Load the model with MLflow
            model = mlflow.pyfunc.load_model(model_dir)
            
            # Load top10 features
            top10_path = os.path.join(model_dir, "top10_features.json")
            if os.path.exists(top10_path):
                with open(top10_path, 'r') as f:
                    top10_features = json.load(f)
                return model, top10_features
            else:
                st.warning(f"top10_features.json not found for {station}, using all features")
                return model, None
                
        except Exception as e:
            st.warning(f"Error loading local model: {e}")
            return None, None
    return None, None

# Main app
try:
    project, fs = get_hopsworks_connection()
    df = load_feature_data(fs)
    model_registry = get_model_registry()
    
    # Pivot the data
    daily = (
        df
        .pivot(index="date", columns="start_station_name", values="ride_count")
        .fillna(0)
    )
    daily.index = pd.to_datetime(daily.index)
    
    # Get list of all stations
    all_stations = sorted(daily.columns.tolist())
    
    # Station selection
    st.sidebar.header("Station Selection")
    selected_stations = st.sidebar.multiselect(
        "Select stations to forecast:",
        options=all_stations,
        default=[all_stations[0]] if all_stations else []
    )
    
    if not selected_stations:
        st.warning("Please select at least one station")
        st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Historical Data", "Forecasts"])
    
    with tab1:
        st.header("Historical Ride Data")
        
        # Simple date filter (last 7/30/90 days)
        days_filter = st.radio(
            "Show data for the last:",
            options=["7 days", "30 days", "90 days"]
        )
        
        days_back = int(days_filter.split()[0])
        cutoff_date = daily.index.max() - timedelta(days=days_back)
        
        # Filter data
        filtered_data = daily.loc[daily.index >= cutoff_date, selected_stations]
        
        # Plot
        fig = px.line(
            filtered_data,
            labels={"value": "Number of Rides"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Ride Forecasts")
        
        # Create predictions
        predictions = []
        
        # Create lag features for prediction
        X_tomorrow = make_lags(daily, selected_stations)
        
        # Get tomorrow's date as string
        tomorrow_str = (daily.index.max() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Load models and make predictions
        for station in selected_stations:
            try:
                # Try local model first
                model, top10_features = load_local_model(station)
                
                if model is not None:
                    # Make prediction with local model
                    if top10_features:
                        X_selected = X_tomorrow[top10_features]
                    else:
                        X_selected = X_tomorrow
                    
                    pred = model.predict(X_selected)[0]
                    pred_rounded = round(float(pred))
                    
                    predictions.append({
                        "Station": station,
                        "Predicted Rides": pred_rounded,
                        "Source": "Local model"
                    })
                    st.success(f"Prediction for {station} successful using local model!")
                    continue
                
                # If local model fails, try Hopsworks
                clean_name = clean_model_name(station)
                model_name = f"citibike_model_{clean_name}"
                
                try:
                    st.info(f"Loading model for {station} from Hopsworks...")
                    # Try to get a specific version (like in inference.py)
                    try:
                        model = model_registry.get_model(name=model_name, version=6)
                    except:
                        # Fall back to latest version
                        model = model_registry.get_model(name=model_name)
                        
                    model_dir = model.download()
                    
                    # Load the downloaded model with MLflow
                    loaded_model = mlflow.pyfunc.load_model(model_dir)
                    
                    # Try to get top10 features from the downloaded model
                    top10_path = os.path.join(model_dir, "top10_features.json")
                    if os.path.exists(top10_path):
                        with open(top10_path) as f:
                            top10_features = json.load(f)
                            X_selected = X_tomorrow[top10_features]
                    else:
                        # Fallback to using all features
                        X_selected = X_tomorrow
                        
                    # Make prediction
                    pred = loaded_model.predict(X_selected)[0]
                    pred_rounded = round(float(pred))
                    
                    predictions.append({
                        "Station": station,
                        "Predicted Rides": pred_rounded,
                        "Source": "Hopsworks model"
                    })
                    st.success(f"Prediction for {station} successful using Hopsworks model!")
                    
                except Exception as e:
                    st.warning(f"Error getting model from Hopsworks: {str(e)}")
                    
                    # Fallback: Simple average prediction as demo
                    station_data = daily[station].values
                    if len(station_data) > 0:
                        # Calculate average of last 7 days
                        recent_avg = daily[station].iloc[-7:].mean()
                        predictions.append({
                            "Station": station,
                            "Predicted Rides": round(recent_avg),
                            "Source": "Fallback (7-day avg)"
                        })
                        st.info(f"Used fallback prediction for {station} (7-day average)")
                
            except Exception as e:
                st.warning(f"Error predicting for {station}: {str(e)}")
        
        # Show predictions
        if predictions:
            pred_df = pd.DataFrame(predictions)
            
            # Simple bar chart
            st.subheader(f"Predicted Rides for {tomorrow_str}")
            fig = px.bar(
                pred_df,
                x="Station",
                y="Predicted Rides",
                text="Predicted Rides",
                color="Source",
                title=f"Predicted Rides for {tomorrow_str}"
            )
            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data as table
            st.subheader("Prediction Details")
            st.dataframe(pred_df, use_container_width=True)
            
            # Download as CSV
            st.download_button(
                label="Download Predictions as CSV",
                data=pred_df.to_csv(index=False),
                file_name=f"citibike_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No predictions could be made.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please check your Hopsworks and MLflow credentials.")